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
# 2) Numerically stable SPD inverse (robust)
# =========================================================
def spd_inverse(A, jitter_eps=1e-6):
    """
    Robust inverse for symmetric positive (semi-)definite A.
    Tries Cholesky, else SVD pseudo-inverse with threshold, else add larger jitter and invert.
    Returns a finite tensor (nan/inf cleaned).
    """
    A_sym = 0.5 * (A + A.t())
    dtype = A_sym.dtype
    device = A_sym.device
    try:
        L = torch.linalg.cholesky(A_sym)
        inv = torch.cholesky_inverse(L)
    except Exception:
        try:
            U, S, Vh = torch.linalg.svd(A_sym, full_matrices=False)
            tol = max(A_sym.shape) * S.max() * torch.finfo(S.dtype).eps
            S_inv = torch.where(S > tol, 1.0 / S, torch.zeros_like(S))
            inv = (Vh.t() * S_inv.unsqueeze(0)).matmul(U.t())
        except Exception:
            I = torch.eye(A_sym.shape[0], device=device, dtype=dtype)
            inv = torch.linalg.inv(A_sym + max(jitter_eps, 1e-3) * I)

    inv = torch.nan_to_num(inv, nan=0.0, posinf=1e6, neginf=-1e6)
    return inv

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

        # ID statistics for normalization (per-mode)
        self.maha_id_mean = None
        self.maha_id_std = None
        self.maha_stats = None

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

        # Clean up NaN/Inf
        self.posterior_cov = torch.nan_to_num(self.posterior_cov, nan=0.0, posinf=1e6, neginf=-1e6)
        self.shared_cov_inv = torch.nan_to_num(self.shared_cov_inv, nan=0.0, posinf=1e6, neginf=-1e6)
        self.global_cov_inv = torch.nan_to_num(self.global_cov_inv, nan=0.0, posinf=1e6, neginf=-1e6)

        # --- 4. Compute ID Mahalanobis Statistics (using min_k, not pred) ---
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

        # --- store per-mode ID statistics ---
        min_mean = float(min_d_cls.mean().item())
        min_std  = float(min_d_cls.std(unbiased=False).item())
        if min_std < 1e-6:
            min_std = 1.0

        rmd = (min_d_cls - d_glob)
        rmd_mean = float(rmd.mean().item())
        rmd_std  = float(rmd.std(unbiased=False).item())
        if rmd_std < 1e-6:
            rmd_std = 1.0

        self.maha_stats = {
            'min_class': (min_mean, min_std),
            'rmd_min':   (rmd_mean, rmd_std),
        }

        self.maha_id_mean = min_mean
        self.maha_id_std  = min_std

        print(f"[Info] Min-class Maha stats on ID: mean={min_mean:.4f}, std={min_std:.4f}")
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
        
        all_d_cls = []
        for c in range(K):
            diff = features - self.fitter.class_means[c]
            d_c = (diff.matmul(self.fitter.shared_cov_inv) * diff).sum(dim=1)
            all_d_cls.append(d_c)
        all_d_cls = torch.stack(all_d_cls, dim=1)  # (N, K)
        
        diff_glob = features - self.fitter.global_mean
        d_glob = (diff_glob.matmul(self.fitter.global_cov_inv) * diff_glob).sum(dim=1)
        
        if self.maha_mode == "min_class":
            score = all_d_cls.min(dim=1)[0]
            
        elif self.maha_mode == "pred_class":
            pred = logits.argmax(dim=1)
            score = all_d_cls[torch.arange(N, device=features.device), pred]
            
        elif self.maha_mode == "rmd_min":
            min_d_cls = all_d_cls.min(dim=1)[0]
            score = min_d_cls - d_glob
            
        elif self.maha_mode == "rmd_pred":
            pred = logits.argmax(dim=1)
            d_pred = all_d_cls[torch.arange(N, device=features.device), pred]
            score = d_pred - d_glob
            
        else:
            raise ValueError(f"Unknown maha_mode: {self.maha_mode}")
        
        score = torch.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
        return score.unsqueeze(1)  # (N, 1)

    def maha_norm_score(self, features, logits):
        """
        Returns normalized Mahalanobis score.
        Higher = more OOD-like
        """
        raw_score = self.compute_maha_score(features, logits).squeeze(1)  # tensor on device

        if not self.normalize_maha:
            return raw_score.unsqueeze(1)

        if hasattr(self.fitter, "maha_stats") and (self.maha_mode in self.fitter.maha_stats):
            m, s = self.fitter.maha_stats[self.maha_mode]
        else:
            m, s = getattr(self.fitter, "maha_id_mean", 0.0), getattr(self.fitter, "maha_id_std", 1.0)

        m_t = torch.tensor(m, device=raw_score.device, dtype=raw_score.dtype)
        s_t = torch.tensor(s, device=raw_score.device, dtype=raw_score.dtype)

        if s_t == 0:
            s_t = torch.tensor(1.0, device=raw_score.device, dtype=raw_score.dtype)

        z = (raw_score - m_t) / s_t

        # KEEP sign information (do not clamp here). We use absolute-value based beta later for stability.
        z = torch.nan_to_num(z, nan=0.0, posinf=1e6, neginf=-1e6)
        return z.unsqueeze(1)

    @torch.no_grad()
    def conf(self, features, logits):
        self._ensure_device(features.device)
        
        v_llla = self.llla_var(features)  # (N,1)
        
        geo = self.beta * self.maha_norm_score(features, logits) if self.beta != 0.0 else 0.0
        
        if isinstance(geo, float) and geo == 0.0:
            sigma_total = v_llla
        else:
            sigma_total = v_llla + geo

        sigma_total = torch.nan_to_num(sigma_total, nan=1e6, posinf=1e6, neginf=0.0)
        sigma_total = torch.clamp(sigma_total, min=0.0)

        kappa = 1.0 / torch.sqrt(1.0 + (self.pi / 8.0) * sigma_total)
        kappa = torch.nan_to_num(kappa, nan=0.0, posinf=1.0, neginf=0.0)

        mod_logits = logits * kappa
        probs = torch.softmax(mod_logits, dim=1)
        conf = probs.max(dim=1)[0]
        return conf.detach().cpu().numpy().reshape(-1)
    
    @torch.no_grad()
    def get_maha_scores(self, features, logits):
        """Get raw Mahalanobis scores for analysis"""
        return self.compute_maha_score(features, logits).squeeze(1).cpu().numpy()

# =========================================================
# 6) AutoBeta: robust (abs-based) + cap
# =========================================================
def auto_beta_from_id(fitter, feat_id, logits_id, alpha=1.0, maha_mode="min_class", 
                      normalize_maha=True, stat="median"):
    """Choose beta so that stat(llla_var) ≈ beta * stat(maha_norm) on ID.
       Use absolute-value based stat for robustness and cap beta.
    """
    pred0 = Predictor(fitter, alpha=alpha, beta=0.0, maha_mode=maha_mode, 
                      normalize_maha=normalize_maha)

    device = feat_id.device
    pred0._ensure_device(device)

    with torch.no_grad():
        v_llla = pred0.llla_var(feat_id).squeeze(1)
        maha_norm = pred0.maha_norm_score(feat_id, logits_id).squeeze(1)

        # mask finite
        mask = torch.isfinite(maha_norm) & torch.isfinite(v_llla)
        if mask.sum().item() == 0:
            print("[AutoBeta] No finite samples; fallback beta=0.5")
            return 0.5

        maha_vals = maha_norm[mask].detach().cpu().numpy()
        v_vals = v_llla[mask].detach().cpu().numpy()

        # use absolute magnitude for matching to be robust to sign
        if stat == "mean":
            a = float(np.mean(v_vals))
            b = float(np.mean(np.abs(maha_vals)))
        else:
            a = float(np.median(v_vals))
            b = float(np.median(np.abs(maha_vals)))

        if b <= 1e-12:
            beta = 0.5
        else:
            beta = a / b

        # cap beta to avoid runaway scaling
        beta = float(np.clip(beta, 0.05, 3.0))
        print(f"[AutoBeta] mode={maha_mode}, stat={stat}, a={a:.6f}, b={b:.6f}, beta={beta:.6f}")
        return beta

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
    L2_NORM_FEATURE =   False  # Keep FALSE for high-magnitude detection
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
    print("      RMD-Focused Implementation - All Modes on Attack")
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

    # ----------------- Quick check for experiment -----------------
    print("\n[Debug] Inspect per-mode raw and normalized scores on ID")
    feat_id_dbg, logits_id_dbg, _ = extract_all(model, id_loader, DEVICE, l2_normalize=L2_NORM_FEATURE)

    pred_dbg = Predictor(fitter, alpha=1.0, beta=0.0, maha_mode="min_class", normalize_maha=False)
    raw_min = pred_dbg.get_maha_scores(feat_id_dbg, logits_id_dbg)
    pred_dbg.maha_mode = "rmd_min"
    raw_rmd = pred_dbg.get_maha_scores(feat_id_dbg, logits_id_dbg)

    import numpy as _np
    print("Raw stats (min_class): mean={:.4f}, std={:.4f}, >0_frac={:.4f}".format(
        float(_np.mean(raw_min)), float(_np.std(raw_min)), float((_np.array(raw_min)>0).mean())))
    print("Raw stats (rmd_min): mean={:.4f}, std={:.4f}, >0_frac={:.4f}".format(
        float(_np.mean(raw_rmd)), float(_np.std(raw_rmd)), float((_np.array(raw_rmd)>0).mean())))

    pred2 = Predictor(fitter, alpha=1.0, beta=0.0, maha_mode="rmd_min", normalize_maha=True)
    z_rmd = pred2.maha_norm_score(feat_id_dbg, logits_id_dbg).squeeze(1).cpu().numpy()

    print("Normed stats (rmd_min): mean={:.4f}, std={:.4f}, >0_frac={:.4f}".format(
        float(_np.mean(z_rmd)), float(_np.std(z_rmd)), float((_np.array(z_rmd)>0).mean())))
    # ----------------- End debug -----------------

    # Extract ID
    print("\n--- Extract ID test feats/logits ---")
    feat_id, logits_id, y_id = feat_id_dbg, logits_id_dbg, None  # reuse extracted ones

    id_acc = evaluate_accuracy(model, id_loader, DEVICE)
    print(f"\n[ID] test accuracy: {id_acc:.2f}%")

    # =========================================================
    # Test multiple modes on OOD sets
    # =========================================================
    modes_to_test = ["min_class", "rmd_min"]
    results_summary = {}

    print("\n" + "="*70)
    print("                  OOD Evaluation")
    print("="*70)

    for mode in modes_to_test:
        print(f"\n=== Mode: {mode} ===")
        beta = auto_beta_from_id(fitter, feat_id, logits_id, alpha=HYB_ALPHA, 
                                 maha_mode=mode, normalize_maha=NORM_MAHA, stat="median")
        results_summary[mode] = {"beta": beta}

        pure = Predictor(fitter, alpha=PURE_ALPHA, beta=0.0, maha_mode=mode, normalize_maha=NORM_MAHA)
        hyb  = Predictor(fitter, alpha=HYB_ALPHA, beta=beta, maha_mode=mode, normalize_maha=NORM_MAHA)

        id_msp  = msp_confidence_from_logits(logits_id)
        id_pure = pure.conf(feat_id, logits_id)
        id_hyb  = hyb.conf(feat_id, logits_id)

        print(f"[Hybrid] auto beta = {beta:.6f}")
        print(f"\n{'OOD set':<15} | {'MSP':>8} | {'Pure':>8} | {'Hybrid':>8}")
        print("-"*50)

        for name, loader in ood_loaders.items():
            feat_ood, logits_ood, _ = extract_all(model, loader, DEVICE, l2_normalize=L2_NORM_FEATURE)

            ood_msp  = msp_confidence_from_logits(logits_ood)
            ood_pure = pure.conf(feat_ood, logits_ood)
            ood_hyb  = hyb.conf(feat_ood, logits_ood)

            au_msp  = auroc_id_vs_ood(id_msp,  ood_msp)
            au_pure = auroc_id_vs_ood(id_pure, ood_pure)
            au_hyb  = auroc_id_vs_ood(id_hyb, ood_hyb)

            print(f"{name:<15} | {au_msp:8.4f} | {au_pure:8.4f} | {au_hyb:8.4f}")

    # =========================================================
    # Attack Evaluation (All Modes)
    # =========================================================
    print("\n" + "="*70)
    print("                  Attack Evaluation (PGD)")
    print("="*70)

    print(f"[Attack] Generating PGD samples (eps={ATTACK_EPS:.5f}, steps={ATTACK_STEPS})...")
    feat_adv, logits_adv, _ = extract_attack(
        model, id_loader, DEVICE,
        l2_normalize=L2_NORM_FEATURE,
        eps=ATTACK_EPS, alpha=ATTACK_ALPHA, steps=ATTACK_STEPS,
        max_batches=ATTACK_MAX_BATCHES
    )

    adv_msp = msp_confidence_from_logits(logits_adv)
    id_msp_final = msp_confidence_from_logits(logits_id)

    print(f"\n{'Mode':<15} | {'MSP':>8} | {'Pure':>8} | {'Hybrid':>8}")
    print("-"*50)

    for mode in modes_to_test:
        beta = results_summary[mode]["beta"]

        pure = Predictor(fitter, alpha=PURE_ALPHA, beta=0.0, maha_mode=mode, normalize_maha=NORM_MAHA)
        hyb  = Predictor(fitter, alpha=HYB_ALPHA, beta=beta, maha_mode=mode, normalize_maha=NORM_MAHA)

        id_pure = pure.conf(feat_id, logits_id)
        id_hyb  = hyb.conf(feat_id, logits_id)

        adv_pure = pure.conf(feat_adv, logits_adv)
        adv_hyb  = hyb.conf(feat_adv, logits_adv)

        au_msp  = auroc_id_vs_ood(id_msp_final, adv_msp)
        au_pure = auroc_id_vs_ood(id_pure, adv_pure)
        au_hyb  = auroc_id_vs_ood(id_hyb, adv_hyb)

        print(f"{mode:<15} | {au_msp:8.4f} | {au_pure:8.4f} | {au_hyb:8.4f}")

if __name__ == "__main__":
    main()

