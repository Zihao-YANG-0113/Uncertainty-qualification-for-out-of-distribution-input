import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# =========================================================
# 0) Repro / metrics
# =========================================================
def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def auroc_id_vs_ood(id_scores, ood_scores):
    """Higher score => more ID-like."""
    y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    y_score = np.concatenate([id_scores, ood_scores])
    return float(roc_auc_score(y_true, y_score))

@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, logits = model(x)
        pred = logits.argmax(dim=1)
        total += y.size(0)
        correct += (pred == y).sum().item()
    return 100.0 * correct / total

def msp_confidence_from_logits(logits):
    probs = torch.softmax(logits, dim=1)
    return probs.max(dim=1)[0].detach().cpu().numpy().reshape(-1)

# =========================================================
# 1) Model: CIFAR ResNet-18 (correct stride placement)
# =========================================================
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64,  2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride=s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)         # CIFAR: 4x4 feature map
        feat = out.view(out.size(0), -1)   # (N,512)
        logits = self.linear(feat)
        return feat, logits

# =========================================================
# 2) Numerically stable SPD inverse
# =========================================================
def spd_inverse(A):
    try:
        L = torch.linalg.cholesky(A)
        return torch.cholesky_inverse(L)
    except RuntimeError:
        return torch.inverse(A)

# =========================================================
# 3) Extract feats/logits (device-safe) + optional L2 norm
# =========================================================
@torch.no_grad()
def extract_all(model, loader, device, l2_normalize=True):
    model.eval()
    feats, logits_list, labels_list = [], [], []
    for x, y in tqdm(loader, desc="Extracting"):
        x, y = x.to(device), y.to(device)
        feat, logits = model(x)
        if l2_normalize:
            feat = F.normalize(feat, p=2, dim=1)
        feats.append(feat)
        logits_list.append(logits)
        labels_list.append(y)
    return torch.cat(feats, 0), torch.cat(logits_list, 0), torch.cat(labels_list, 0)

# =========================================================
# 4) LLLA Fitter + RMD (Relative Mahalanobis) - FIXED VERSION
# =========================================================
class LLLA_Fitter:
    def __init__(self, num_classes, tau=1e-4, jitter=1e-6):
        self.num_classes = int(num_classes)
        self.tau = float(tau)
        self.jitter = float(jitter)

        # LLLA components
        self.posterior_cov = None   # (d,d)
        
        # Class-conditional Mahalanobis components
        self.class_means = None     # (K,d)
        self.shared_cov_inv = None  # (d,d)

        # Background (Global) components for RMD
        self.global_mean = None     # (d,)
        self.global_cov_inv = None  # (d,d)

        # ID statistics for normalization
        # 使用标准 Mahalanobis (min_k) 的统计量
        self.maha_id_mean = None
        self.maha_id_std = None

    def fit(self, features, logits, labels):
        device = features.device
        N, d = features.shape
        K = logits.shape[1]
        assert K == self.num_classes

        print(f"[-] Fit LLLA + RMD: N={N}, d={d}, device={device}")

        # --- 1. LLLA: weighted Hessian proxy ---
        probs = torch.softmax(logits, dim=1)
        r = 1.0 - (probs * probs).sum(dim=1)
        r = r.clamp_min(1e-4)
        Phi_w = features * torch.sqrt(r).unsqueeze(1)
        H = Phi_w.t().matmul(Phi_w)

        precision = H + self.tau * torch.eye(d, device=device)
        precision = precision + self.jitter * torch.eye(d, device=device)
        self.posterior_cov = spd_inverse(precision)

        # --- 2. Class-Conditional Gaussian (Foreground) ---
        self.class_means = torch.zeros(self.num_classes, d, device=device)
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.any():
                self.class_means[c] = features[mask].mean(dim=0)

        # Shared covariance (within-class)
        mu_per = self.class_means[labels]
        Xc = features - mu_per
        cov = Xc.t().matmul(Xc) / max(N - 1, 1)
        cov = cov + self.jitter * torch.eye(d, device=device)
        self.shared_cov_inv = spd_inverse(cov)

        # --- 3. Global Gaussian (Background) for RMD ---
        self.global_mean = features.mean(dim=0)
        X_global = features - self.global_mean
        cov_global = X_global.t().matmul(X_global) / max(N - 1, 1)
        cov_global = cov_global + self.jitter * torch.eye(d, device=device)
        self.global_cov_inv = spd_inverse(cov_global)

        # --- 4. Compute ID Mahalanobis Statistics (using min_k, not pred) ---
        # 计算每个样本到所有类的马氏距离，取最小值
        all_d_cls = []
        for c in range(self.num_classes):
            diff = features - self.class_means[c]
            d_c = (diff.matmul(self.shared_cov_inv) * diff).sum(dim=1)
            all_d_cls.append(d_c)
        all_d_cls = torch.stack(all_d_cls, dim=1)  # (N, K)
        min_d_cls = all_d_cls.min(dim=1)[0]  # (N,)

        # D_global
        diff_glob = features - self.global_mean
        d_glob = (diff_glob.matmul(self.global_cov_inv) * diff_glob).sum(dim=1)

        # 标准 Mahalanobis (不是 RMD): 只用 min_k D_class
        # 这个统计量用于归一化
        self.maha_id_mean = float(min_d_cls.mean().item())
        self.maha_id_std  = float(min_d_cls.std(unbiased=False).item())
        if self.maha_id_std < 1e-6:
            self.maha_id_std = 1.0

        print(f"[Info] Min-class Maha stats on ID: mean={self.maha_id_mean:.4f}, std={self.maha_id_std:.4f}")
        
        # 也计算 RMD 统计量用于对比
        rmd_scores = min_d_cls - d_glob
        rmd_mean = float(rmd_scores.mean().item())
        rmd_std = float(rmd_scores.std(unbiased=False).item())
        print(f"[Info] RMD (min_k) stats on ID: mean={rmd_mean:.4f}, std={rmd_std:.4f}")
        
        print("[-] Fit complete.")

# =========================================================
# 5) Predictor: LLLA + Mahalanobis (Multiple Variants)
# =========================================================
class Predictor:
    def __init__(self, fitter: LLLA_Fitter, alpha=1.0, beta=0.0, 
                 maha_mode="min_class",  # "min_class", "pred_class", "rmd_min", "rmd_pred"
                 normalize_maha=True):
        self.fitter = fitter
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.maha_mode = maha_mode
        self.normalize_maha = bool(normalize_maha)
        self.pi = 3.1415926535

    def _ensure_device(self, device):
        if self.fitter.posterior_cov.device != device:
            self.fitter.posterior_cov = self.fitter.posterior_cov.to(device)
        if self.fitter.class_means.device != device:
            self.fitter.class_means = self.fitter.class_means.to(device)
        if self.fitter.shared_cov_inv.device != device:
            self.fitter.shared_cov_inv = self.fitter.shared_cov_inv.to(device)
        if self.fitter.global_mean.device != device:
            self.fitter.global_mean = self.fitter.global_mean.to(device)
        if self.fitter.global_cov_inv.device != device:
            self.fitter.global_cov_inv = self.fitter.global_cov_inv.to(device)

    def llla_var(self, features):
        self._ensure_device(features.device)
        tmp = features.matmul(self.fitter.posterior_cov)
        v = (tmp * features).sum(dim=1, keepdim=True)
        return self.alpha * v

    def compute_maha_score(self, features, logits):
        """
        Compute Mahalanobis-based score based on mode.
        Returns a score where HIGHER = more OOD-like (more uncertain)
        """
        self._ensure_device(features.device)
        N = features.shape[0]
        K = self.fitter.num_classes
        
        # Compute D_class for all classes
        all_d_cls = []
        for c in range(K):
            diff = features - self.fitter.class_means[c]
            d_c = (diff.matmul(self.fitter.shared_cov_inv) * diff).sum(dim=1)
            all_d_cls.append(d_c)
        all_d_cls = torch.stack(all_d_cls, dim=1)  # (N, K)
        
        # Compute D_global
        diff_glob = features - self.fitter.global_mean
        d_glob = (diff_glob.matmul(self.fitter.global_cov_inv) * diff_glob).sum(dim=1)
        
        if self.maha_mode == "min_class":
            # Standard Mahalanobis: min_k D_k(x)
            # Higher = further from all classes = more OOD
            score = all_d_cls.min(dim=1)[0]
            
        elif self.maha_mode == "pred_class":
            # Predicted class Mahalanobis
            pred = logits.argmax(dim=1)
            score = all_d_cls[torch.arange(N, device=features.device), pred]
            
        elif self.maha_mode == "rmd_min":
            # RMD with min_k: min_k(D_k - D_0) = min_k(D_k) - D_0
            min_d_cls = all_d_cls.min(dim=1)[0]
            score = min_d_cls - d_glob
            
        elif self.maha_mode == "rmd_pred":
            # RMD with predicted class
            pred = logits.argmax(dim=1)
            d_pred = all_d_cls[torch.arange(N, device=features.device), pred]
            score = d_pred - d_glob
            
        elif self.maha_mode == "neg_rmd_min":
            # Negative RMD: -RMD = D_0 - min_k(D_k)
            # 当 OOD 在背景分布中也远离时，D_0 大，-RMD 大
            min_d_cls = all_d_cls.min(dim=1)[0]
            score = d_glob - min_d_cls
            
        else:
            raise ValueError(f"Unknown maha_mode: {self.maha_mode}")
        
        return score.unsqueeze(1)  # (N, 1)

    def maha_norm_score(self, features, logits):
        """
        Returns normalized Mahalanobis score.
        Higher = more OOD-like
        """
        raw_score = self.compute_maha_score(features, logits).squeeze(1)
        
        if not self.normalize_maha:
            return raw_score.unsqueeze(1)
        
        # Normalize using ID statistics
        m = self.fitter.maha_id_mean
        s = self.fitter.maha_id_std
        z = (raw_score - m) / s
        
        # 只惩罚比典型 ID 更远的样本
        z = torch.clamp(z, min=0.0)
        return z.unsqueeze(1)

    @torch.no_grad()
    def conf(self, features, logits):
        self._ensure_device(features.device)
        
        v_llla = self.llla_var(features)
        
        # Mahalanobis penalty
        geo = self.beta * self.maha_norm_score(features, logits) if self.beta != 0.0 else 0.0
        
        sigma_total = v_llla + geo
        kappa = 1.0 / torch.sqrt(1.0 + (self.pi / 8.0) * sigma_total)
        
        mod_logits = logits * kappa
        probs = torch.softmax(mod_logits, dim=1)
        conf = probs.max(dim=1)[0]
        return conf.detach().cpu().numpy().reshape(-1)
    
    @torch.no_grad()
    def get_maha_scores(self, features, logits):
        """Get raw Mahalanobis scores for analysis"""
        return self.compute_maha_score(features, logits).squeeze(1).cpu().numpy()

# =========================================================
# 6) AutoBeta: match magnitudes on ID
# =========================================================
def auto_beta_from_id(fitter, feat_id, logits_id, alpha=1.0, maha_mode="min_class", 
                      normalize_maha=True, stat="median"):
    """Choose beta so that stat(llla_var) ≈ beta * stat(maha_norm) on ID."""
    pred0 = Predictor(fitter, alpha=alpha, beta=0.0, maha_mode=maha_mode, 
                      normalize_maha=normalize_maha)

    device = feat_id.device
    pred0._ensure_device(device)

    with torch.no_grad():
        v_llla = pred0.llla_var(feat_id).squeeze(1)
        maha_norm = pred0.maha_norm_score(feat_id, logits_id).squeeze(1)

        # Only align on samples where penalty is active (>0)
        mask = (maha_norm > 0)
        n = int(mask.sum().item())
        print(f"[AutoBeta] mode={maha_mode}, stat={stat}, samples with score>0: n={n}")

        if n < 100:
            print("[AutoBeta] Too few samples with score>0; fallback beta=0.1")
            return 0.1

        v = v_llla[mask].detach().cpu().numpy()
        m = maha_norm[mask].detach().cpu().numpy()

        if stat == "mean":
            a = float(np.mean(v))
            b = float(np.mean(m))
        else:
            a = float(np.median(v))
            b = float(np.median(m))

        if b <= 1e-12:
            print("[AutoBeta] Maha statistic too small; fallback beta=0.1")
            return 0.1

        beta = a / b
        print(f"[AutoBeta] {stat}(llla_var)={a:.6f}, {stat}(maha_score)={b:.6f} => beta={beta:.6f}")
        return float(beta)

# =========================================================
# 7) PGD attack
# =========================================================
def pgd_attack(model, x, y, eps=8/255, alpha=2/255, steps=10):
    model.eval()
    x_adv = x.detach().clone()
    x_adv.requires_grad_(True)
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
    x_adv = x_adv.detach().clone()
    x_adv.requires_grad_(True)

    for _ in range(steps):
        _, logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad_(True)

    return x_adv.detach()

@torch.no_grad()
def extract_attack(model, loader, device, l2_normalize=True, eps=8/255, alpha=2/255, steps=10, max_batches=None):
    model.eval()
    feats, logits_list, labels_list = [], [], []
    for bi, (x, y) in enumerate(tqdm(loader, desc="AttackExtract")):
        if max_batches is not None and bi >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.enable_grad():
            x_adv = pgd_attack(model, x, y, eps=eps, alpha=alpha, steps=steps)
        feat, logits = model(x_adv)
        if l2_normalize:
            feat = F.normalize(feat, p=2, dim=1)
        feats.append(feat.detach())
        logits_list.append(logits.detach())
        labels_list.append(y.detach())
    return torch.cat(feats, 0), torch.cat(logits_list, 0), torch.cat(labels_list, 0)

# =========================================================
# 8) Train or load
# =========================================================
def train_or_load(model, train_loader, val_loader, device, epochs, save_path):
    if os.path.exists(save_path):
        print(f"\n[Info] Found checkpoint: '{save_path}'")
        print("[Info] Loading weights, skip training...")
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.to(device)
        val_acc = evaluate_accuracy(model, val_loader, device)
        print(f"[Info] Val accuracy after load: {val_acc:.2f}%")
        return

    print(f"\n[Training] Train from scratch ({epochs} epochs)...")
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    best = 0.0
    for ep in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            _, logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            pbar.set_postfix({"loss": float(loss.item()), "lr": sch.get_last_lr()[0]})
        sch.step()

        val_acc = evaluate_accuracy(model, val_loader, device)
        print(f"[Info] epoch={ep+1}, val_acc={val_acc:.2f}%")
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), save_path)
            print("[Info] saved best checkpoint")

# =========================================================
# 9) OOD datasets helpers
# =========================================================
class GaussianNoiseDataset(torch.utils.data.Dataset):
    def __init__(self, n, shape=(3,32,32), mean=0.0, std=1.0, transform=None):
        self.n = int(n)
        self.shape = shape
        self.mean = mean
        self.std = std
        self.transform = transform

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = torch.randn(self.shape) * self.std + self.mean
        x = torch.clamp(x, -3, 3)
        x = (x + 3) / 6.0
        y = 0
        if self.transform is not None:
            x = self.transform(x)
        return x, y

# =========================================================
# 10) Main
# =========================================================
def main():
    set_seed(0)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------
    BATCH_SIZE = 128
    TRAIN_EPOCHS = 50
    WEIGHT_PATH = "resnet18_cifar10_weights.pth"
    TAU = 1e-4
    JITTER = 1e-6
    L2_NORM_FEATURE =   False
    NORM_MAHA = True

    PURE_ALPHA = 1.0
    HYB_ALPHA = 1.0

    MAX_OOD_N = 10000
    ATTACK_EPS = 8/255
    ATTACK_ALPHA = 2/255
    ATTACK_STEPS = 10
    ATTACK_MAX_BATCHES = 40
    # -------------------------

    print("\n" + "="*70)
    print("     FIXED RMD Implementation - Multiple Modes Comparison")
    print("="*70)
    print(f"DEVICE={DEVICE}")

    # Transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Datasets
    ds_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    ds_id_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    ds_cifar100 = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)
    ds_svhn = datasets.SVHN(root="./data", split="test", download=True, transform=transform_test)
    ds_stl = datasets.STL10(root="./data", split="test", download=True,
                            transform=transforms.Compose([
                                transforms.Resize((32,32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ]))
    ds_noise = GaussianNoiseDataset(n=MAX_OOD_N, transform=lambda x: transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    )(x))

    # Loaders
    train_idx = range(45000)
    val_idx   = range(45000, 50000)

    train_loader = DataLoader(Subset(ds_train, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(Subset(ds_train, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    fit_loader   = DataLoader(Subset(ds_train, range(50000)), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    id_loader    = DataLoader(ds_id_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    def cap_dataset(ds, cap):
        if cap is None:
            return ds
        cap = min(len(ds), cap)
        return Subset(ds, range(cap))

    ood_sets = {
        "CIFAR100": cap_dataset(ds_cifar100, MAX_OOD_N),
        "SVHN": cap_dataset(ds_svhn, MAX_OOD_N),
        "STL10": cap_dataset(ds_stl, MAX_OOD_N),
        "GaussNoise": ds_noise,
    }
    ood_loaders = {k: DataLoader(v, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) for k, v in ood_sets.items()}

    # Model
    model = ResNet18(num_classes=10).to(DEVICE)
    train_or_load(model, train_loader, val_loader, DEVICE, TRAIN_EPOCHS, WEIGHT_PATH)

    # Fit
    print("\n--- Extract training feats/logits for fit ---")
    feat_train, logits_train, y_train = extract_all(model, fit_loader, DEVICE, l2_normalize=L2_NORM_FEATURE)
    fitter = LLLA_Fitter(num_classes=10, tau=TAU, jitter=JITTER)
    fitter.fit(feat_train, logits_train, y_train)
    del feat_train, logits_train, y_train
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Extract ID
    print("\n--- Extract ID test feats/logits ---")
    feat_id, logits_id, y_id = extract_all(model, id_loader, DEVICE, l2_normalize=L2_NORM_FEATURE)

    id_acc = evaluate_accuracy(model, id_loader, DEVICE)
    print(f"\n[ID] test accuracy: {id_acc:.2f}%")

    # =========================================================
    # Test multiple modes
    # =========================================================
    modes_to_test = ["min_class", "rmd_min", "neg_rmd_min"]
    
    results = {}
    
    for mode in modes_to_test:
        print(f"\n{'='*60}")
        print(f"  Testing mode: {mode}")
        print(f"{'='*60}")
        
        # Auto beta
        beta = auto_beta_from_id(fitter, feat_id, logits_id, alpha=HYB_ALPHA, 
                                  maha_mode=mode, normalize_maha=NORM_MAHA, stat="median")
        
        # Predictors
        pure = Predictor(fitter, alpha=PURE_ALPHA, beta=0.0, maha_mode=mode, normalize_maha=NORM_MAHA)
        hyb  = Predictor(fitter, alpha=HYB_ALPHA, beta=beta, maha_mode=mode, normalize_maha=NORM_MAHA)

        # ID scores
        id_msp  = msp_confidence_from_logits(logits_id)
        id_pure = pure.conf(feat_id, logits_id)
        id_hyb  = hyb.conf(feat_id, logits_id)

        print(f"[Hybrid] auto beta = {beta:.6f}")
        
        results[mode] = {"beta": beta}

        # OOD evaluation
        print(f"\n{'OOD set':<15} | {'MSP':>8} | {'Pure':>8} | {'Hybrid':>8}")
        print("-"*50)
        
        for name, loader in ood_loaders.items():
            feat_ood, logits_ood, _ = extract_all(model, loader, DEVICE, l2_normalize=L2_NORM_FEATURE)

            ood_msp  = msp_confidence_from_logits(logits_ood)
            ood_pure = pure.conf(feat_ood, logits_ood)
            ood_hyb  = hyb.conf(feat_ood, logits_ood)

            au_msp  = auroc_id_vs_ood(id_msp,  ood_msp)
            au_pure = auroc_id_vs_ood(id_pure, ood_pure)
            au_hyb  = auroc_id_vs_ood(id_hyb,  ood_hyb)
            
            results[mode][name] = {"msp": au_msp, "pure": au_pure, "hybrid": au_hyb}
            print(f"{name:<15} | {au_msp:8.4f} | {au_pure:8.4f} | {au_hyb:8.4f}")

    # =========================================================
    # Summary comparison
    # =========================================================
    print("\n" + "="*70)
    print("                    SUMMARY COMPARISON")
    print("="*70)
    print(f"{'OOD':<12} | {'MSP':>7} | ", end="")
    for mode in modes_to_test:
        print(f"{mode[:8]:>10} | ", end="")
    print()
    print("-"*70)
    
    for ood_name in ood_loaders.keys():
        msp_val = results[modes_to_test[0]][ood_name]["msp"]
        print(f"{ood_name:<12} | {msp_val:7.4f} | ", end="")
        for mode in modes_to_test:
            hyb_val = results[mode][ood_name]["hybrid"]
            # Mark best with *
            print(f"{hyb_val:10.4f} | ", end="")
        print()

    # Attack evaluation (only for best mode)
    print("\n" + "="*60)
    print("  Attack Evaluation (using min_class mode)")
    print("="*60)
    
    best_mode = "min_class"
    beta = results[best_mode]["beta"]
    pure = Predictor(fitter, alpha=PURE_ALPHA, beta=0.0, maha_mode=best_mode, normalize_maha=NORM_MAHA)
    hyb  = Predictor(fitter, alpha=HYB_ALPHA, beta=beta, maha_mode=best_mode, normalize_maha=NORM_MAHA)
    
    id_msp  = msp_confidence_from_logits(logits_id)
    id_pure = pure.conf(feat_id, logits_id)
    id_hyb  = hyb.conf(feat_id, logits_id)
    
    print(f"[Attack] PGD eps={ATTACK_EPS:.5f}, steps={ATTACK_STEPS}")
    feat_adv, logits_adv, _ = extract_attack(
        model, id_loader, DEVICE,
        l2_normalize=L2_NORM_FEATURE,
        eps=ATTACK_EPS, alpha=ATTACK_ALPHA, steps=ATTACK_STEPS,
        max_batches=ATTACK_MAX_BATCHES
    )

    adv_msp  = msp_confidence_from_logits(logits_adv)
    adv_pure = pure.conf(feat_adv, logits_adv)
    adv_hyb  = hyb.conf(feat_adv, logits_adv)

    print(f"\n{'Attack':<15} | {'MSP':>8} | {'Pure':>8} | {'Hybrid':>8}")
    print("-"*50)
    print(f"{'PGD':<15} | {auroc_id_vs_ood(id_msp, adv_msp):8.4f} | "
          f"{auroc_id_vs_ood(id_pure, adv_pure):8.4f} | "
          f"{auroc_id_vs_ood(id_hyb, adv_hyb):8.4f}")

if __name__ == "__main__":
    main()