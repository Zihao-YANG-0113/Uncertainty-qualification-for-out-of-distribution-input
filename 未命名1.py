import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from tqdm import tqdm
import os
import sys

# ==========================================
# 1. Model Definition: Simplified ResNet-18
# ==========================================
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        feature = out.view(out.size(0), -1)
        logit = self.linear(feature)
        return feature, logit

# ==========================================
# 2. LLLA Fitter (Unified Covariance and Means)
# ==========================================
class LLLA_Fitter:
    """Calculates the covariance (Sigma) for LLLA and means/inverses for Mahalanobis distance."""
    def __init__(self, num_classes, tau=1.0):
        self.num_classes = num_classes
        self.tau = tau
        self.posterior_cov = None 
        self.class_means = None
        self.shared_cov_inv = None
        self.pi = 3.1415926535

    def fit(self, features, labels):
        N = len(labels)
        print(f"[-] Fitting LLLA Components (N={N})...")
        device = features.device
        N, dim = features.shape
        
        # 1. LLLA Covariance (Sigma)
        H = torch.matmul(features.T, features)
        precision_matrix = H + self.tau * torch.eye(dim, device=device)
        self.posterior_cov = torch.inverse(precision_matrix + 1e-6 * torch.eye(dim, device=device))
        
        # 2. Mahalanobis Means
        self.class_means = torch.zeros(self.num_classes, dim, device=device)
        for c in range(self.num_classes):
            class_mask = (labels == c)
            if class_mask.sum() > 0:
                self.class_means[c] = features[class_mask].mean(dim=0)
            
        # 3. Mahalanobis Shared Inverse Covariance
        batch_means = self.class_means[labels]
        X_centered = features - batch_means
        cov = torch.matmul(X_centered.T, X_centered) / (N - 1)
        self.shared_cov_inv = torch.inverse(cov + 1e-6 * torch.eye(dim, device=device))
        print("[-] Fit Complete.")

# ==========================================
# 3. Predictor Classes (Pure LLLA and Hybrid LLLA)
# ==========================================
class Predictor:
    """Base class for prediction logic."""
    def __init__(self, fitter, llla_scale, beta=0.0):
        self.fitter = fitter
        self.llla_scale = llla_scale
        self.beta = beta
        self.pi = 3.1415926535

    def get_llla_variance(self, features):
        """Calculates LLLA variance (phi^T * Sigma * phi)."""
        temp_llla = torch.matmul(features, self.fitter.posterior_cov)
        sigma_llla_sq_base = (temp_llla * features).sum(dim=1, keepdim=True)
        return sigma_llla_sq_base * self.llla_scale
    
    def get_mahalanobis_term(self, features, original_logits):
        """Calculates scaled Mahalanobis distance (Beta * Maha)."""
        if self.beta == 0.0:
            return torch.zeros_like(original_logits.max(dim=1, keepdim=True)[0])
            
        pred_labels = original_logits.argmax(dim=1)
        closest_means = self.fitter.class_means[pred_labels]
        diff = features - closest_means
        temp_dist = torch.matmul(diff, self.fitter.shared_cov_inv)
        mahalanobis_dist_base = (temp_dist * diff).sum(dim=1, keepdim=True)
        return mahalanobis_dist_base * self.beta

    def predict(self, features, original_logits, debug=False, tag=""):
        device = features.device
        
        # 1. Calculate Variance Components
        sigma_llla_sq_scaled = self.get_llla_variance(features)
        geo_term = self.get_mahalanobis_term(features, original_logits)

        # 2. Total Variance and Probit Approximation
        sigma_total_sq = sigma_llla_sq_scaled + geo_term
        
        kappa = 1.0 / torch.sqrt(1.0 + (self.pi / 8.0) * sigma_total_sq)
        modulated_logits = original_logits * kappa
        
        probs = torch.softmax(modulated_logits, dim=1)
        confs = probs.max(dim=1)[0]
        
        if debug:
            mean_conf = confs.mean().item()
            mean_logit = original_logits.max(dim=1)[0].mean().item()
            
            print(f"\n[DEBUG: {tag}] Scale={self.llla_scale:.1f}, Beta={self.beta:.4f}")
            print(f"  > Original Logits Mean: {mean_logit:.4f}")
            print(f"  > LLLA Var (Scaled)   : {sigma_llla_sq_scaled.mean().item():.4f}")
            print(f"  > Geo Term (Scaled)   : {geo_term.mean().item():.4f}")
            print(f"  > Total Var (Sigma^2) : {sigma_total_sq.mean().item():.4f}")
            print(f"  > Kappa Mean          : {kappa.mean().item():.4f}")
            print(f"  > Mean Confidence     : {mean_conf:.4f}")

        return confs.cpu().numpy()

# ==========================================
# 4. Training and Utility Functions
# ==========================================
def extract_features(model, loader, device):
    """Extracts ResNet features and Logits"""
    model.eval()
    features_list, logits_list, labels_list = [], [], []
    for images, labels in tqdm(loader, desc="Extracting Features"):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            feat, logit = model(images)
        features_list.append(feat.cpu())
        logits_list.append(logit.cpu())
        labels_list.append(labels.cpu())
    return torch.cat(features_list), torch.cat(logits_list), torch.cat(labels_list)

def train_cnn(model, train_loader, val_loader, device, epochs, save_path):
    """Trains CNN model from scratch or loads checkpoint."""
    if os.path.exists(save_path):
        print(f"\n[Info] Found ResNet checkpoint: '{save_path}'")
        print("[Info] Loading weights, skipping training...")
        try:
            model.load_state_dict(torch.load(save_path, map_location=device))
            model.to(device)
            print("[Info] Load successful! Accuracy check...")
            val_acc = evaluate_accuracy(model, val_loader, device)
            print(f"[Info] Loaded Model Validation Accuracy: {val_acc:.2f}%")
            return
        except Exception as e:
            print(f"[Warning] Load failed ({e}), starting retraining...")

    print(f"\n[Training] Starting training from scratch ({epochs} epochs)...")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    best_acc = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            _, outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix({'Loss': total_loss/(total/images.size(0)), 'Acc': 100.*correct/total, 'LR': scheduler.get_last_lr()[0]})
            
        scheduler.step()
        
        val_acc = evaluate_accuracy(model, val_loader, device)
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"[Info] Epoch {epoch+1}: Validation Acc {val_acc:.2f}%. Saving model...")
            torch.save(model.state_dict(), save_path)
            
    print(f"[Info] Training complete. Best validation Acc: {best_acc:.2f}%.")

def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            _, outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

def msp_confidence(logits):
    """Calculates Max Softmax Probability (MSP) confidence."""
    probs = torch.softmax(logits, dim=1)
    return probs.max(dim=1)[0].cpu().numpy()

def print_auroc(id_scores, ood_scores):
    """Calculates AUROC (Area Under the ROC Curve)."""
    # ID is positive class (score should be high), OOD is negative class (score should be low)
    y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    y_scores = np.concatenate([id_scores, ood_scores])
    # For OOD detection, we look at P(ID) vs P(OOD). High score means high P(ID).
    return roc_auc_score(y_true, y_scores)


# ==========================================
# 5. Main Execution
# ==========================================
def main():
    BATCH_SIZE = 128
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    WEIGHT_PATH = "resnet18_cifar10_weights.pth"
    TRAIN_EPOCHS = 50 # Set higher for production quality (e.g., 100-200)

    # --- Prediction Parameters ---
    PURE_LLLA_SCALE = 1.0       # Theoretical LLLA baseline (Alpha = 1.0)
    HYBRID_LLLA_SCALE = 5.0     # Recommended CNN Alpha correction
    HYBRID_BETA = 0.01          # Recommended CNN Beta (reduced Geo term)
    TAU = 1e-4                  # Prior precision (from LLLA paper, common default)
    # ---------------------------
    
    print("--- LLLA Baseline & Hybrid Model Comparison ---")
    print(f"Architecture: ResNet-18 (Train from Scratch)")
    print(f"Pure LLLA (Theory): Scale={PURE_LLLA_SCALE:.1f}, Beta=0.0")
    print(f"Hybrid LLLA (CNN Opt): Scale={HYBRID_LLLA_SCALE:.1f}, Beta={HYBRID_BETA:.4f}")
    
    # 1. Data Loading and Preprocessing
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

    print("Loading datasets...")
    ds_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    ds_test_id = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    ds_test_ood = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test) # OOD Test
    
    # Data Loaders Setup
    train_indices = range(45000)
    val_indices = range(45000, 50000)
    
    train_loader = DataLoader(Subset(ds_train, train_indices), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(ds_train, val_indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    fit_loader = DataLoader(Subset(ds_train, range(50000)), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    id_loader = DataLoader(ds_test_id, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    ood_loader = DataLoader(ds_test_ood, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 2. Model Instantiation and Training/Loading
    model = ResNet18(num_classes=10)
    model.to(DEVICE)
    train_cnn(model, train_loader, val_loader, DEVICE, epochs=TRAIN_EPOCHS, save_path=WEIGHT_PATH)
    
    # 3. LLLA Fitting (Post-hoc)
    print("\n--- Extracting Training Features for LLLA Fit ---")
    feat_train, _, label_train = extract_features(model, fit_loader, DEVICE)
    
    fitter = LLLA_Fitter(num_classes=10, tau=TAU)
    fitter.fit(feat_train.to(DEVICE), label_train.to(DEVICE))
    
    del feat_train, label_train
    torch.cuda.empty_cache()
    
    # 4. Predictor Instantiation
    pure_llla = Predictor(fitter, llla_scale=PURE_LLLA_SCALE, beta=0.0)
    hybrid_llla_optimal = Predictor(fitter, llla_scale=HYBRID_LLLA_SCALE, beta=HYBRID_BETA)
    
    # 5. Feature Extraction for Prediction
    print("\n--- Extracting Test Features (ID/OOD) ---")
    feat_id, logits_id, labels_id = extract_features(model, id_loader, DEVICE)
    feat_ood, logits_ood, _ = extract_features(model, ood_loader, DEVICE)

    # 6. Prediction and Comparison
    
    # MSP Baseline
    msp_confs_id = msp_confidence(logits_id)
    msp_confs_ood = msp_confidence(logits_ood)
    
    # Pure LLLA (Alpha=1.0)
    pure_llla_confs_id = pure_llla.predict(feat_id, logits_id, debug=True, tag="Pure LLLA (Theory)")
    pure_llla_confs_ood = pure_llla.predict(feat_ood, logits_ood)
    
    # Hybrid LLLA (Alpha=5.0, Beta=0.01)
    hybrid_confs_id = hybrid_llla_optimal.predict(feat_id, logits_id, debug=True, tag="Hybrid LLLA (CNN Opt)")
    hybrid_confs_ood = hybrid_llla_optimal.predict(feat_ood, logits_ood)
    
    # 7. Final Results Calculation
    id_acc = evaluate_accuracy(model, id_loader, DEVICE)

    auroc_msp = print_auroc(msp_confs_id, msp_confs_ood)
    auroc_pure_llla = print_auroc(pure_llla_confs_id, pure_llla_confs_ood)
    auroc_hybrid = print_auroc(hybrid_confs_id, hybrid_confs_ood)

    print("\n" + "="*70)
    print(f"{'RESNET-18 LLLA/HYBRID COMPARISON (CIFAR-10 vs CIFAR-100)':^70}")
    print("="*70)
    print(f"ID Test Accuracy: {id_acc:.2f}%")
    print("-" * 70)
    print(f"{'Method':<20} | {'ID Conf Mean':<15} | {'OOD Conf Mean':<15} | {'AUROC ↑':<15}")
    print("-" * 70)
    print(f"{'MSP (Baseline)':<20} | {np.mean(msp_confs_id):<15.4f} | {np.mean(msp_confs_ood):<15.4f} | {auroc_msp:<15.4f}")
    print(f"{'Pure LLLA (a=1.0)':<20} | {np.mean(pure_llla_confs_id):<15.4f} | {np.mean(pure_llla_confs_ood):<15.4f} | {auroc_pure_llla:<15.4f}")
    print(f"{'Hybrid LLLA (a=5.0)':<20} | {np.mean(hybrid_confs_id):<15.4f} | {np.mean(hybrid_confs_ood):<15.4f} | {auroc_hybrid:<15.4f}")
    print("="*70)
    print("结论预测：Hybrid LLLA (a=5.0) 应该能达到最佳 AUROC，同时保持 ID 置信度在 0.90 左右。")


if __name__ == "__main__":
    main()