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
    L = torch.linalg.cholesky(A)
    return torch.cholesky_inverse(L)

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
            feat = F.normalize(feat, p=2, dim=1)  # stabilize scale
        feats.append(feat)
        logits_list.append(logits)
        labels_list.append(y)
    return torch.cat(feats, 0), torch.cat(logits_list, 0), torch.cat(labels_list, 0)

# =========================================================
# 4) LLLA fitter + Mahalanobis + ID stats for normalization
# =========================================================
class LLLA_Fitter:
    def __init__(self, num_classes, tau=1e-4, jitter=1e-6):
        self.num_classes = int(num_classes)
        self.tau = float(tau)
        self.jitter = float(jitter)

        self.posterior_cov = None   # (d,d)
        self.class_means = None     # (K,d)
        self.shared_cov_inv = None  # (d,d)

        self.maha_id_mean = None
        self.maha_id_std = None

    def fit(self, features, logits, labels):
        device = features.device
        N, d = features.shape
        K = logits.shape[1]
        assert K == self.num_classes

        print(f"[-] Fit LLLA + Mahalanobis: N={N}, d={d}, device={device}")

        # weighted Hessian proxy (Kristiadi-style)
        probs = torch.softmax(logits, dim=1)
        r = 1.0 - (probs * probs).sum(dim=1)
        r = r.clamp_min(1e-4)
        Phi_w = features * torch.sqrt(r).unsqueeze(1)
        H = Phi_w.t().matmul(Phi_w)

        precision = H + self.tau * torch.eye(d, device=device)
        precision = precision + self.jitter * torch.eye(d, device=device)
        self.posterior_cov = spd_inverse(precision)

        # class means
        self.class_means = torch.zeros(self.num_classes, d, device=device)
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.any():
                self.class_means[c] = features[mask].mean(dim=0)

        # shared covariance inverse
        mu_per = self.class_means[labels]
        Xc = features - mu_per
        cov = Xc.t().matmul(Xc) / max(N - 1, 1)
        cov = cov + self.jitter * torch.eye(d, device=device)
        self.shared_cov_inv = spd_inverse(cov)

        # ID stats for normalized Mahalanobis (pred-class)
        pred = logits.argmax(dim=1)
        mu_pred = self.class_means[pred]
        diff = features - mu_pred
        maha = (diff.matmul(self.shared_cov_inv) * diff).sum(dim=1)

        self.maha_id_mean = float(maha.mean().item())
        self.maha_id_std  = float(maha.std(unbiased=False).item())
        if self.maha_id_std < 1e-6:
            self.maha_id_std = 1.0

        print(f"[Info] maha_id_mean={self.maha_id_mean:.4f}, maha_id_std={self.maha_id_std:.4f}")
        print("[-] Fit complete.")

# =========================================================
# 5) Predictor: Kristiadi confidence
# =========================================================
class Predictor:
    def __init__(self, fitter: LLLA_Fitter, alpha=1.0, beta=0.0, normalize_maha=True):
        self.fitter = fitter
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.normalize_maha = bool(normalize_maha)
        self.pi = 3.1415926535

    def _ensure_device(self, device):
        if self.fitter.posterior_cov.device != device:
            self.fitter.posterior_cov = self.fitter.posterior_cov.to(device)
        if self.fitter.class_means.device != device:
            self.fitter.class_means = self.fitter.class_means.to(device)
        if self.fitter.shared_cov_inv.device != device:
            self.fitter.shared_cov_inv = self.fitter.shared_cov_inv.to(device)

    def llla_var(self, features):
        self._ensure_device(features.device)
        tmp = features.matmul(self.fitter.posterior_cov)
        v = (tmp * features).sum(dim=1, keepdim=True)
        return self.alpha * v

    def maha_norm_predclass(self, features, logits):
        self._ensure_device(features.device)
        pred = logits.argmax(dim=1)
        mu = self.fitter.class_means[pred]
        diff = features - mu
        d2 = (diff.matmul(self.fitter.shared_cov_inv) * diff).sum(dim=1, keepdim=True)

        if not self.normalize_maha:
            return d2

        m = self.fitter.maha_id_mean
        s = self.fitter.maha_id_std
        z = (d2 - m) / s
        z = torch.clamp(z, min=0.0)  # max(0, z-score)
        return z

    @torch.no_grad()
    def conf(self, features, logits):
        self._ensure_device(features.device)
        v_llla = self.llla_var(features)
        geo = self.beta * self.maha_norm_predclass(features, logits) if self.beta != 0.0 else 0.0
        sigma_total = v_llla + geo
        kappa = 1.0 / torch.sqrt(1.0 + (self.pi / 8.0) * sigma_total)
        mod_logits = logits * kappa
        probs = torch.softmax(mod_logits, dim=1)
        conf = probs.max(dim=1)[0]
        return conf.detach().cpu().numpy().reshape(-1)

# =========================================================
# 6) AutoBeta: match magnitudes on ID
# =========================================================
def auto_beta_from_id(fitter, feat_id, logits_id, alpha=1.0, normalize_maha=True, stat="median"):
    """Choose beta so that stat(llla_var) ≈ beta * stat(maha_norm) on ID (where maha_norm>0)."""
    pred0 = Predictor(fitter, alpha=alpha, beta=0.0, normalize_maha=normalize_maha)

    # compute components on device
    device = feat_id.device
    pred0._ensure_device(device)

    with torch.no_grad():
        v_llla = pred0.llla_var(feat_id).squeeze(1)  # (N,)
        # reuse maha_norm_predclass
        maha_norm = pred0.maha_norm_predclass(feat_id, logits_id).squeeze(1)

        mask = (maha_norm > 0)
        n = int(mask.sum().item())
        print(f"[AutoBeta] stat={stat}, using ID samples with maha_norm>0: n={n}")

        if n < 100:
            print("[AutoBeta] Too few maha_norm>0 samples; fallback beta=0.1")
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
            print("[AutoBeta] maha statistic too small; fallback beta=0.1")
            return 0.1

        beta = a / b
        print(f"[AutoBeta] target: {stat}(llla_var)≈beta*{stat}(maha_norm)")
        print(f"[AutoBeta] {stat}(llla_var)={a:.6f}, {stat}(maha_norm)={b:.6f} => beta={beta:.6f}")
        return float(beta)

# =========================================================
# 7) PGD attack (attack-set as extra evaluation)
# =========================================================
def pgd_attack(model, x, y, eps=8/255, alpha=2/255, steps=10):
    """
    Untargeted PGD on input space (Linf).
    Returns adversarial x_adv clipped to [0,1] in normalized space? -> here we attack on normalized inputs directly.
    """
    model.eval()
    x_adv = x.detach().clone()
    x_adv.requires_grad_(True)

    # random init within eps
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
        # PGD needs grad -> temporarily enable
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
# 9) OOD datasets (far) helpers
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
        # map to [0,1] roughly (optional); here keep as tensor then apply transform if needed
        # We'll clamp after un-normalize? For simplicity, just clamp to [-3,3] then rescale to [0,1]
        x = torch.clamp(x, -3, 3)
        x = (x + 3) / 6.0
        y = 0
        if self.transform is not None:
            # transform expects PIL usually; but our transform_test is tensor normalize,
            # so apply normalize manually:
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
    L2_NORM_FEATURE = True
    NORM_MAHA = True

    PURE_ALPHA = 1.0
    HYB_ALPHA = 1.0

    # OOD sample cap (speed)
    MAX_OOD_N = 10000
    # Attack params
    ATTACK_EPS = 8/255
    ATTACK_ALPHA = 2/255
    ATTACK_STEPS = 10
    ATTACK_MAX_BATCHES = 40   # speed; set None for full test set
    # -------------------------

    print("\n--- Far-OOD + Attack eval (Kristiadi confidence AUROC only) ---")
    print(f"DEVICE={DEVICE}")

    # Transforms (CIFAR)
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

    # ID: CIFAR-10
    ds_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    ds_id_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    # Near-OOD: CIFAR-100 (optional)
    ds_cifar100 = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)

    # Far-OOD: SVHN / STL10 (torchvision built-in)
    ds_svhn = datasets.SVHN(root="./data", split="test", download=True, transform=transform_test)
    ds_stl = datasets.STL10(root="./data", split="test", download=True,
                            transform=transforms.Compose([
                                transforms.Resize((32,32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ]))

    # Far-OOD: Gaussian noise
    ds_noise = GaussianNoiseDataset(n=MAX_OOD_N, transform=lambda x: transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    )(x))

    # loaders
    train_idx = range(45000)
    val_idx   = range(45000, 50000)

    train_loader = DataLoader(Subset(ds_train, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(Subset(ds_train, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    fit_loader   = DataLoader(Subset(ds_train, range(50000)), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    id_loader    = DataLoader(ds_id_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # OOD loaders (cap size)
    def cap_dataset(ds, cap):
        if cap is None:
            return ds
        cap = min(len(ds), cap)
        return Subset(ds, range(cap))

    ood_sets = {
        "Near-OOD:CIFAR100": cap_dataset(ds_cifar100, MAX_OOD_N),
        "Far-OOD:SVHN": cap_dataset(ds_svhn, MAX_OOD_N),
        "Far-OOD:STL10": cap_dataset(ds_stl, MAX_OOD_N),
        "Far-OOD:GaussianNoise": ds_noise,
    }
    ood_loaders = {k: DataLoader(v, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) for k, v in ood_sets.items()}

    # model train/load
    model = ResNet18(num_classes=10).to(DEVICE)
    train_or_load(model, train_loader, val_loader, DEVICE, TRAIN_EPOCHS, WEIGHT_PATH)

    # fit
    print("\n--- Extract training feats/logits for fit ---")
    feat_train, logits_train, y_train = extract_all(model, fit_loader, DEVICE, l2_normalize=L2_NORM_FEATURE)
    fitter = LLLA_Fitter(num_classes=10, tau=TAU, jitter=JITTER)
    fitter.fit(feat_train, logits_train, y_train)
    del feat_train, logits_train, y_train
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # extract ID once
    print("\n--- Extract ID test feats/logits ---")
    feat_id, logits_id, y_id = extract_all(model, id_loader, DEVICE, l2_normalize=L2_NORM_FEATURE)

    # auto beta
    beta = auto_beta_from_id(fitter, feat_id, logits_id, alpha=HYB_ALPHA, normalize_maha=NORM_MAHA, stat="median")
    pure = Predictor(fitter, alpha=PURE_ALPHA, beta=0.0, normalize_maha=NORM_MAHA)
    hyb  = Predictor(fitter, alpha=HYB_ALPHA,  beta=beta, normalize_maha=NORM_MAHA)

    # compute ID scores
    id_msp  = msp_confidence_from_logits(logits_id)
    id_pure = pure.conf(feat_id, logits_id)
    id_hyb  = hyb.conf(feat_id, logits_id)

    id_acc = evaluate_accuracy(model, id_loader, DEVICE)
    print(f"\n[ID] test accuracy: {id_acc:.2f}%")
    print(f"[Hybrid] auto beta = {beta:.6f}")

    # eval OOD sets
    print("\n==================== OOD AUROC(conf) ====================")
    print(f"{'OOD set':<22} | {'MSP':>8} | {'Pure':>8} | {'Hybrid':>8}")
    print("-"*60)
    for name, loader in ood_loaders.items():
        feat_ood, logits_ood, _ = extract_all(model, loader, DEVICE, l2_normalize=L2_NORM_FEATURE)

        ood_msp  = msp_confidence_from_logits(logits_ood)
        ood_pure = pure.conf(feat_ood, logits_ood)
        ood_hyb  = hyb.conf(feat_ood, logits_ood)

        au_msp  = auroc_id_vs_ood(id_msp,  ood_msp)
        au_pure = auroc_id_vs_ood(id_pure, ood_pure)
        au_hyb  = auroc_id_vs_ood(id_hyb,  ood_hyb)
        print(f"{name:<22} | {au_msp:8.4f} | {au_pure:8.4f} | {au_hyb:8.4f}")

    # Attack-set evaluation (PGD on CIFAR-10 test)
    print("\n==================== Attack AUROC(conf) ====================")
    print(f"[Attack] PGD eps={ATTACK_EPS:.5f}, alpha={ATTACK_ALPHA:.5f}, steps={ATTACK_STEPS}, max_batches={ATTACK_MAX_BATCHES}")
    feat_adv, logits_adv, _ = extract_attack(
        model, id_loader, DEVICE,
        l2_normalize=L2_NORM_FEATURE,
        eps=ATTACK_EPS, alpha=ATTACK_ALPHA, steps=ATTACK_STEPS,
        max_batches=ATTACK_MAX_BATCHES
    )

    adv_msp  = msp_confidence_from_logits(logits_adv)
    adv_pure = pure.conf(feat_adv, logits_adv)
    adv_hyb  = hyb.conf(feat_adv, logits_adv)

    # compare clean ID vs adv (treat adv as "OOD-like")
    print(f"{'Attack set':<22} | {'MSP':>8} | {'Pure':>8} | {'Hybrid':>8}")
    print("-"*60)
    print(f"{'PGD(CIFAR10)':<22} | "
          f"{auroc_id_vs_ood(id_msp, adv_msp):8.4f} | "
          f"{auroc_id_vs_ood(id_pure, adv_pure):8.4f} | "
          f"{auroc_id_vs_ood(id_hyb, adv_hyb):8.4f}")

if __name__ == "__main__":
    main()
