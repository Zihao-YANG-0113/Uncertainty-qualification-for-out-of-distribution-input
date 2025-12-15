import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from tqdm import tqdm
import sys
import os  # <--- 新增: 用于检查文件是否存在

# ==========================================
# 1. 核心算法: Density-Modulated LLLA
# ==========================================
class DensityModulatedLLLA:
    def __init__(self, feature_dim, num_classes, tau=1.0, beta=1.0, llla_scale=1.0):
        self.num_classes = num_classes
        self.tau = tau
        self.beta = beta
        self.llla_scale = llla_scale
        self.posterior_cov = None 
        self.class_means = None
        self.shared_cov_inv = None

    def fit(self, features, labels):
        N = len(labels)
        print(f"[-] Fitting Density-Modulated LLLA (N={N})...")
        device = features.device
        N, dim = features.shape
        
        # A. LLLA (Hessian)
        H = torch.matmul(features.T, features)
        precision_matrix = H + self.tau * torch.eye(dim, device=device)
        self.posterior_cov = torch.inverse(precision_matrix + 1e-4 * torch.eye(dim, device=device))
        
        # B. Geometry (GDA)
        self.class_means = torch.zeros(self.num_classes, dim, device=device)
        for c in range(self.num_classes):
            class_mask = (labels == c)
            if class_mask.sum() > 0:
                self.class_means[c] = features[class_mask].mean(dim=0)
            
        batch_means = self.class_means[labels]
        X_centered = features - batch_means
        cov = torch.matmul(X_centered.T, X_centered) / (N - 1)
        self.shared_cov_inv = torch.inverse(cov + 1e-4 * torch.eye(dim, device=device))
        print("[-] Fit Complete.")

    def predict_uncertainty(self, features, original_logits, beta_override=None, debug=False, tag=""):
        # 1. LLLA Variance
        temp = torch.matmul(features, self.posterior_cov)
        sigma_llla_sq = (temp * features).sum(dim=1, keepdim=True)
        sigma_llla_sq = sigma_llla_sq * self.llla_scale # Scaling
        
        # 2. Mahalanobis Distance
        pred_labels = original_logits.argmax(dim=1)
        closest_means = self.class_means[pred_labels]
        diff = features - closest_means
        temp_dist = torch.matmul(diff, self.shared_cov_inv)
        mahalanobis_dist = (temp_dist * diff).sum(dim=1, keepdim=True)
        
        # 3. Fusion
        current_beta = self.beta if beta_override is None else beta_override
        geo_term = current_beta * mahalanobis_dist
        sigma_total_sq = sigma_llla_sq + geo_term
        
        # 4. Probit Approximation
        pi = 3.1415926535
        kappa = 1.0 / torch.sqrt(1.0 + (pi / 8.0) * sigma_total_sq)
        modulated_logits = original_logits * kappa
        
        # 5. Confidence
        probs = torch.softmax(modulated_logits, dim=1)
        confs = probs.max(dim=1)[0]
        
        if debug:
            ratio = geo_term / (sigma_llla_sq + 1e-9)
            mean_conf = confs.mean().item()
            mean_logit = original_logits.max(dim=1)[0].mean().item()
            
            print(f"\n[DEBUG: {tag}] Beta={current_beta}, Scale={self.llla_scale}")
            print(f"  > 原始 Logits Mean  : {mean_logit:.4f} (目标 > 8.0)")
            print(f"  > LLLA方差 (Scaled) : {sigma_llla_sq.mean().item():.4f}")
            print(f"  > 几何修正 (Term)   : {geo_term.mean().item():.4f}")
            print(f"  > 缩放系数 (Kappa)  : {kappa.mean().item():.4f}")
            print(f"  > 平均置信度 (Conf) : {mean_conf:.4f}")
            
            if tag == "ID" and mean_conf < 0.85:
                 print("  [提示] ID置信度略低，若需提高请减小 Beta 或 Scale。")
        
        return confs.cpu().numpy()

# ==========================================
# 2. 辅助函数
# ==========================================
def pgd_attack(model, images, labels, eps=0.01, alpha=0.005, steps=5, device='cuda'):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    adv_images = images + torch.empty_like(images).uniform_(-eps, eps)
    adv_images = torch.clamp(adv_images, 0, 1).detach()
    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images).logits
        loss = F.cross_entropy(outputs, labels)
        grad = torch.autograd.grad(loss, adv_images)[0]
        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()
    return adv_images

def extract_features(model, loader, device, attack=False):
    model.eval()
    features_list, logits_list, labels_list = [], [], []
    desc = "Extracting (Attack)" if attack else "Extracting"
    for images, labels in tqdm(loader, desc=desc):
        images, labels = images.to(device), labels.to(device)
        if attack:
            with torch.enable_grad():
                images = pgd_attack(model, images, labels, device=device)
        with torch.no_grad():
            outputs = model(images, output_hidden_states=True)
            feat = outputs.hidden_states[-2][:, 0, :] 
            logit = outputs.logits
        features_list.append(feat.cpu())
        logits_list.append(logit.cpu())
        labels_list.append(labels.cpu())
    return torch.cat(features_list), torch.cat(logits_list), torch.cat(labels_list)

# ==========================================
# 3. [升级版] 训练或加载 Head
# ==========================================
def train_or_load_head(model, train_loader, device, epochs=5, save_path="cifar10_head_ft.pth"):
    # 1. 检查是否存在已保存的模型
    if os.path.exists(save_path):
        print(f"\n[Info] 发现已训练的 Head 权重文件: '{save_path}'")
        print("[Info] 正在加载，跳过训练...")
        try:
            # 只加载 classifier 部分的权重
            model.classifier.load_state_dict(torch.load(save_path, map_location=device))
            print("[Info] 加载成功！")
            return
        except Exception as e:
            print(f"[Warning] 加载失败 ({e})，准备重新训练...")

    # 2. 如果不存在，则开始训练
    print(f"\n[Training] 未发现保存的权重，开始 Linear Probing 训练 ({epochs} epochs)...")
    print("[Training] Freezing backbone, unfreezing classifier...")
    
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=3e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'Loss': total_loss/(total/images.size(0)), 'Acc': 100.*correct/total})
            
    # 3. 训练完成后保存
    print(f"[Info] 训练完成，正在保存 Head 权重到 '{save_path}'...")
    torch.save(model.classifier.state_dict(), save_path)
    print("[Info] 保存成功！下次运行将直接加载。\n")

# ==========================================
# 4. 主程序
# ==========================================
def main():
    BATCH_SIZE = 128
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 最佳参数配置
    MY_LLLA_SCALE = 1000.0    
    MY_BETA = 0.02         
    
    FT_EPOCHS = 5
    N_FT_DATA = 50000 
    N_FIT = 50000     
    N_TEST = 2000
    N_ADV = 500
    
    # 定义权重保存路径
    HEAD_CHECKPOINT_PATH = "vit_tiny_cifar10_head.pth"
    
    print(f"Config: Scale={MY_LLLA_SCALE}, Beta={MY_BETA}")
    
    # 数据加载
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), 
    ])
    
    print("Loading datasets...")
    ds_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    ds_test_id = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    ds_test_ood = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    
    def get_random_subset(ds, n):
        n = min(n, len(ds))
        indices = torch.randperm(len(ds))[:n]
        return Subset(ds, indices)

    train_loader_ft = DataLoader(Subset(ds_train, range(N_FT_DATA)), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    fit_loader = DataLoader(Subset(ds_train, range(N_FIT)), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    id_loader = DataLoader(get_random_subset(ds_test_id, N_TEST), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    ood_loader = DataLoader(get_random_subset(ds_test_ood, N_TEST), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    adv_loader = DataLoader(get_random_subset(ds_test_id, N_ADV), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 加载模型
    model_name = "WinKawaks/vit-tiny-patch16-224"
    print(f"Loading model: {model_name}...")
    model = ViTForImageClassification.from_pretrained(model_name, num_labels=10, ignore_mismatched_sizes=True)
    model.to(DEVICE)
    
    # Step 0: 智能微调 (加载或训练)
    # 只要这个文件存在，就会跳过训练
    train_or_load_head(model, train_loader_ft, DEVICE, epochs=FT_EPOCHS, save_path=HEAD_CHECKPOINT_PATH)
    
    # Step 1: 拟合 LLLA
    print("Extracting training features for LLLA fit...")
    feat_train, _, label_train = extract_features(model, fit_loader, DEVICE)
    
    dm_llla = DensityModulatedLLLA(
        feature_dim=feat_train.shape[1], 
        num_classes=10, 
        beta=MY_BETA, 
        llla_scale=MY_LLLA_SCALE
    )
    dm_llla.fit(feat_train.to(DEVICE), label_train.to(DEVICE))
    
    del feat_train, label_train
    torch.cuda.empty_cache()
    
    # Step 2: 评估
    def evaluate_dataset(loader, tag, attack=False):
        feat, logits, _ = extract_features(model, loader, DEVICE, attack=attack)
        results = {}
        probs = torch.softmax(logits, dim=1)
        results['MSP'] = probs.max(dim=1)[0].numpy()
        
        if tag == "ID":
            preds = probs.argmax(dim=1)
            all_labels = []
            for _, y in loader: all_labels.append(y)
            all_labels = torch.cat(all_labels).numpy()
            acc = accuracy_score(all_labels, preds.numpy())
            print(f"  > ID Accuracy: {acc*100:.2f}%")

        res_pure_llla = []
        res_ours = []
        first_batch = True
        batch_size = 100
        
        for i in range(0, len(feat), batch_size):
            f_b = feat[i:i+batch_size].to(DEVICE)
            l_b = logits[i:i+batch_size].to(DEVICE)
            
            res_pure_llla.append(dm_llla.predict_uncertainty(f_b, l_b, beta_override=0.0))
            res_ours.append(dm_llla.predict_uncertainty(f_b, l_b, debug=first_batch, tag=tag))
            first_batch = False
            
        results['Pure_LLLA'] = np.concatenate(res_pure_llla)
        results['Ours'] = np.concatenate(res_ours)
        torch.cuda.empty_cache()
        return results

    print("\n>>> Evaluating ID...")
    res_id = evaluate_dataset(id_loader, "ID")
    
    print("\n>>> Evaluating OOD...")
    res_ood = evaluate_dataset(ood_loader, "Near-OOD")
    
    print("\n>>> Evaluating Attack...")
    res_adv = evaluate_dataset(adv_loader, "Attack", attack=True)
    
    # Step 3: 结果
    def print_auroc(id_scores, ood_scores):
        y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
        y_scores = np.concatenate([id_scores, ood_scores])
        return roc_auc_score(y_true, y_scores)

    print("\n" + "="*80)
    print(f"{'FINAL RESULTS (Finetuned + Scale='+str(MY_LLLA_SCALE)+', Beta='+str(MY_BETA)+')':^80}")
    print("="*80)
    print(f"{'Task':<25} | {'MSP':<10} | {'LLLA Only':<15} | {'Ours':<15}")
    print("-" * 80)
    
    s_msp = print_auroc(res_id['MSP'], res_ood['MSP'])
    s_llla = print_auroc(res_id['Pure_LLLA'], res_ood['Pure_LLLA'])
    s_ours = print_auroc(res_id['Ours'], res_ood['Ours'])
    print(f"{'Near-OOD (CIFAR-100)':<25} | {s_msp:.4f}     | {s_llla:.4f}          | {s_ours:.4f}")
    
    s_msp = print_auroc(res_id['MSP'], res_adv['MSP'])
    s_llla = print_auroc(res_id['Pure_LLLA'], res_adv['Pure_LLLA'])
    s_ours = print_auroc(res_id['Ours'], res_adv['Ours'])
    print(f"{'Attack (PGD)':<25} | {s_msp:.4f}     | {s_llla:.4f}          | {s_ours:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()