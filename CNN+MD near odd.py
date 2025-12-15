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

def roc_auc_id_vs_ood(id_scores, ood_scores):
    """Higher score => more ID-like."""
    y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    y_score = np.concatenate([id_scores, ood_scores])
    return roc_auc_score(y_true, y_score)

def summarize_dist(name, arr):
    arr = np.asarray(arr).reshape(-1)
    q = np.quantile(arr, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    print(f"{name}: mean={arr.mean():.4f}, std={arr.std():.4f}")
    print(f"  q0={q[0]:.4f}, q10={q[1]:.4f}, q25={q[2]:.4f}, q50={q[3]:.4f}, q75={q[4]:.4f}, q90={q[5]:.4f}, q100={q[6]:.4f}")

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
# 3) Extract features/logits (device-safe) + optional L2 norm
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

def msp_confidence_from_logits(logits):
    probs = torch.softmax(logits, dim=1)
    return probs.max(dim=1)[0].detach().cpu().numpy()

# =========================================================
# 4) LLLA fitter + Mahalanobis + ID stats for normalization
# =========================================================
class LLLA_Fitter:
    def __init__(self, num_classes, tau=1e-4, jitter=1e-6):
        self.num_classes = int(num_classes)
        self.tau = float(tau)
        self.jitter = float(jitter)

        self.posterior_cov = None
        self.class_means = None
        self.shared_cov_inv = None

        self.maha_id_mean = None
        self.maha_id_std = None

    def fit(self, features, logits, labels):
        device = features.device
        N, d = features.shape
        K = logits.shape[1]
        assert K == self.num_classes

        print(f"[-] Fit LLLA + Mahalanobis: N={N}, d={d}, device={device}")

        # LLLA precision: H + tau I
        probs = torch.softmax(logits, dim=1)
        r = 1.0 - (probs * probs).sum(dim=1)    # scalar proxy
        r = r.clamp_min(1e-4)
        Phi_w = features * torch.sqrt(r).unsqueeze(1)
        H = Phi_w.t().matmul(Phi_w)

        precision = H + self.tau * torch.eye(d, device=device)
        precision = precision + self.jitter * torch.eye(d, device=device)
        self.posterior_cov = spd_inverse(precision)

        # Class means
        self.class_means = torch.zeros(self.num_classes, d, device=device)
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.any():
                self.class_means[c] = features[mask].mean(dim=0)

        # Shared covariance inverse (class-conditional centered)
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
# 5) Predictor: Kristiadi confidence + UQ-only scores
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

    def maha_raw_predclass(self, features, logits):
        self._ensure_device(features.device)
        pred = logits.argmax(dim=1)
        mu = self.fitter.class_means[pred]
        diff = features - mu
        d2 = (diff.matmul(self.fitter.shared_cov_inv) * diff).sum(dim=1, keepdim=True)
        return d2

    def maha_term(self, features, logits):
        maha_raw = self.maha_raw_predclass(features, logits)
        maha_norm = maha_raw

        if self.normalize_maha:
            m = self.fitter.maha_id_mean
            s = self.fitter.maha_id_std
            maha_norm = (maha_raw - m) / s
            maha_norm = torch.clamp(maha_norm, min=0.0)  # max(0, z-score)

        geo = self.beta * maha_norm
        return geo, maha_raw, maha_norm

    @torch.no_grad()
    def forward(self, features, logits, return_aux=False, debug=False, tag=""):
        self._ensure_device(features.device)

        v_llla = self.llla_var(features)
        geo, maha_raw, maha_norm = self.maha_term(features, logits)

        sigma_total = v_llla + geo
        kappa = 1.0 / torch.sqrt(1.0 + (self.pi / 8.0) * sigma_total)

        mod_logits = logits * kappa
        probs = torch.softmax(mod_logits, dim=1)
        conf = probs.max(dim=1)[0]  # Kristiadi-style maximum predictive probability

        if debug:
            print(f"\n[DEBUG: {tag}] alpha={self.alpha:.3f}, beta={self.beta:.6f}, norm_maha={self.normalize_maha}")
            print(f"  > logits_max_mean : {logits.max(dim=1)[0].mean().item():.4f}")
            print(f"  > llla_var_mean   : {v_llla.mean().item():.4f}")
            print(f"  > geo_term_mean   : {geo.mean().item():.4f}")
            print(f"  > sigma_tot_mean  : {sigma_total.mean().item():.4f}")
            print(f"  > kappa_mean      : {kappa.mean().item():.4f}")
            print(f"  > conf_mean       : {conf.mean().item():.4f}")

        conf_np = conf.detach().cpu().numpy().reshape(-1)
        if not return_aux:
            return conf_np

        aux = {
            "conf": conf_np,
            "kappa": kappa.detach().cpu().numpy().reshape(-1),
            "sigma_total": sigma_total.detach().cpu().numpy().reshape(-1),
            "llla_var": v_llla.detach().cpu().numpy().reshape(-1),
            "geo_term": geo.detach().cpu().numpy().reshape(-1),
            "maha_raw": maha_raw.detach().cpu().numpy().reshape(-1),
            "maha_norm": maha_norm.detach().cpu().numpy().reshape(-1),
        }
        return conf_np, aux

# =========================================================
# 6) Train or load
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
# 7) Auto beta: match magnitudes of llla_var and maha_norm on ID
# =========================================================
def auto_beta_from_id(fitter, feat_id, logits_id, alpha=1.0, normalize_maha=True,
                      stat="median", clip=(0.0, 5.0), eps=1e-12):
    """
    Choose beta so that typical geo_term (=beta*maha_norm) matches typical llla_var on ID.
    We align on samples where maha_norm > 0 to avoid being dominated by zeros.
    """
    tmp = Predictor(fitter, alpha=alpha, beta=1.0, normalize_maha=normalize_maha)
    _, aux = tmp.forward(feat_id, logits_id, return_aux=True)

    llla = np.asarray(aux["llla_var"]).reshape(-1)
    maha = np.asarray(aux["maha_norm"]).reshape(-1)

    mask = maha > 0
    if mask.sum() < 50:  # too few positives -> geo signal rare, set beta=0
        print("[AutoBeta] Too few maha_norm>0 on ID; set beta=0.")
        return 0.0

    llla_m = llla[mask]
    maha_m = maha[mask]

    if stat == "mean":
        a = float(llla_m.mean())
        b = float(maha_m.mean())
    else:  # median (robust)
        a = float(np.median(llla_m))
        b = float(np.median(maha_m))

    beta = a / (b + eps)

    if clip is not None:
        lo, hi = clip
        beta = float(np.clip(beta, lo, hi))

    print(f"[AutoBeta] stat={stat}, using ID samples with maha_norm>0: n={mask.sum()}")
    print(f"[AutoBeta] target: median(llla_var)≈beta*median(maha_norm)")
    print(f"[AutoBeta] {stat}(llla_var)={a:.6f}, {stat}(maha_norm)={b:.6f} => beta={beta:.6f}")
    return beta

# =========================================================
# 8) Main
# =========================================================
def main():
    set_seed(0)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    BATCH_SIZE = 128
    TRAIN_EPOCHS = 50
    WEIGHT_PATH = "resnet18_cifar10_weights.pth"

    TAU = 1e-4
    JITTER = 1e-6

    PURE_ALPHA = 1.0

    HYB_ALPHA = 1.0
    HYB_BETA = 0.05
    NORM_MAHA = True
    L2_NORM_FEATURE = True

    # ✅ 你要的：自动设定 beta，让两项贡献同一个数量级
    AUTO_BETA = True
    AUTO_BETA_STAT = "median"   # "median" 更稳健；也可改 "mean"
    AUTO_BETA_CLIP = (0.0, 5.0) # 避免极端 beta；你也可放宽如 (0, 20)

    print("\n--- LLLA / Hybrid UQ (Kristiadi-style eval + UQ-only AUROC) ---")
    print(f"DEVICE={DEVICE}")
    print(f"Pure  : alpha={PURE_ALPHA}, beta=0")
    print(f"Hybrid(init): alpha={HYB_ALPHA}, beta={HYB_BETA}, norm_maha={NORM_MAHA}, feat_L2norm={L2_NORM_FEATURE}")

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

    ds_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    ds_id_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    ds_ood_test = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)

    train_idx = range(45000)
    val_idx   = range(45000, 50000)

    train_loader = DataLoader(Subset(ds_train, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(Subset(ds_train, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    fit_loader   = DataLoader(Subset(ds_train, range(50000)), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    id_loader    = DataLoader(ds_id_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    ood_loader   = DataLoader(ds_ood_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = ResNet18(num_classes=10).to(DEVICE)
    train_or_load(model, train_loader, val_loader, DEVICE, TRAIN_EPOCHS, WEIGHT_PATH)

    print("\n--- Extract training feats/logits for fit ---")
    feat_train, logits_train, y_train = extract_all(model, fit_loader, DEVICE, l2_normalize=L2_NORM_FEATURE)

    fitter = LLLA_Fitter(num_classes=10, tau=TAU, jitter=JITTER)
    fitter.fit(feat_train, logits_train, y_train)

    del feat_train, logits_train, y_train
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n--- Extract test feats/logits (ID/OOD) ---")
    feat_id, logits_id, _ = extract_all(model, id_loader, DEVICE, l2_normalize=L2_NORM_FEATURE)
    feat_ood, logits_ood, _ = extract_all(model, ood_loader, DEVICE, l2_normalize=L2_NORM_FEATURE)

    # ✅ 自动 beta：让 geo_term 和 llla_var 同数量级
    if AUTO_BETA:
        HYB_BETA = auto_beta_from_id(
            fitter, feat_id, logits_id,
            alpha=HYB_ALPHA,
            normalize_maha=NORM_MAHA,
            stat=AUTO_BETA_STAT,
            clip=AUTO_BETA_CLIP
        )
        print(f"[Hybrid(auto)] alpha={HYB_ALPHA}, beta={HYB_BETA:.6f}")

    pure = Predictor(fitter, alpha=PURE_ALPHA, beta=0.0, normalize_maha=NORM_MAHA)
    hyb  = Predictor(fitter, alpha=HYB_ALPHA,  beta=HYB_BETA, normalize_maha=NORM_MAHA)

    msp_id  = msp_confidence_from_logits(logits_id)
    msp_ood = msp_confidence_from_logits(logits_ood)

    pure_conf_id = pure.forward(feat_id, logits_id, return_aux=False, debug=True, tag="Pure")
    pure_conf_ood = pure.forward(feat_ood, logits_ood, return_aux=False)

    hyb_conf_id, aux_id = hyb.forward(feat_id, logits_id, return_aux=True, debug=True, tag="Hybrid")
    hyb_conf_ood, aux_ood = hyb.forward(feat_ood, logits_ood, return_aux=True)

    auroc_msp  = roc_auc_id_vs_ood(msp_id, msp_ood)
    auroc_pure = roc_auc_id_vs_ood(pure_conf_id, pure_conf_ood)
    auroc_hyb  = roc_auc_id_vs_ood(hyb_conf_id, hyb_conf_ood)

    auroc_kappa = roc_auc_id_vs_ood(aux_id["kappa"], aux_ood["kappa"])
    auroc_neg_sigma = roc_auc_id_vs_ood(-aux_id["sigma_total"], -aux_ood["sigma_total"])
    auroc_maha_idscore = roc_auc_id_vs_ood(-aux_id["maha_norm"], -aux_ood["maha_norm"])

    mmc_id_msp, mmc_ood_msp = float(np.mean(msp_id)), float(np.mean(msp_ood))
    mmc_id_pure, mmc_ood_pure = float(np.mean(pure_conf_id)), float(np.mean(pure_conf_ood))
    mmc_id_hyb, mmc_ood_hyb = float(np.mean(hyb_conf_id)), float(np.mean(hyb_conf_ood))

    id_acc = evaluate_accuracy(model, id_loader, DEVICE)

    print("\n" + "="*78)
    print(f"{'RESNET-18  (ID=CIFAR-10)  vs  (Near-OOD=CIFAR-100)':^78}")
    print("="*78)
    print(f"ID Test Accuracy: {id_acc:.2f}%")
    print("-"*78)
    print(f"{'Method':<22} | {'MMC-ID':<10} | {'MMC-OOD':<10} | {'AUROC(conf)↑':<12}")
    print("-"*78)
    print(f"{'MSP (baseline)':<22} | {mmc_id_msp:<10.4f} | {mmc_ood_msp:<10.4f} | {auroc_msp:<12.4f}")
    print(f"{'Pure LLLA':<22}      | {mmc_id_pure:<10.4f} | {mmc_ood_pure:<10.4f} | {auroc_pure:<12.4f}")
    print(f"{'Hybrid (LLLA+Maha)':<22} | {mmc_id_hyb:<10.4f} | {mmc_ood_hyb:<10.4f} | {auroc_hyb:<12.4f}")
    print("="*78)

    print("\n[UQ-only scoring]")
    print(f"AUROC(kappa as ID-score)        : {auroc_kappa:.4f}")
    print(f"AUROC(-sigma_total as ID-score) : {auroc_neg_sigma:.4f}")

    print("\n[Diagnostics]")
    print(f"AUROC(-maha_norm as ID-score)   : {auroc_maha_idscore:.4f}")

    print("\n[Diagnostics: distributions]")
    summarize_dist("ID  llla_var", aux_id["llla_var"])
    summarize_dist("ID  geo_term", aux_id["geo_term"])
    summarize_dist("ID  maha_norm", aux_id["maha_norm"])
    summarize_dist("OOD llla_var", aux_ood["llla_var"])
    summarize_dist("OOD geo_term", aux_ood["geo_term"])
    summarize_dist("OOD maha_norm", aux_ood["maha_norm"])

if __name__ == "__main__":
    main()
