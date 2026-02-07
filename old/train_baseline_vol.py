import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os

from config import cfg
from data_loader import load_and_process_data
from model import MPO_Network

os.makedirs('models', exist_ok=True)
MODEL_SAVE_PATH = 'models/baseline_vol_model.pth'

def train_volatility():
    train_loader, _, _ = load_and_process_data()
    
    model = MPO_Network().to(cfg.DEVICE).double()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.MSELoss()
    
    print(f"🌊 [Benchmark] 开始训练 Volatility 模型 (for Risk Parity)...")
    print(f"   目标: 让预测的 L (协方差因子) 逼近真实的波动")

    for epoch in range(cfg.EPOCHS):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Vol Epoch {epoch+1}/{cfg.EPOCHS}")
        
        for x, y in pbar:
            x, y = x.to(cfg.DEVICE).double(), y.to(cfg.DEVICE).double()
            
            # --- 构造波动率标签 (Proxy Label) ---
            # 真实波动率很难观测，我们用 y^2 近似 (Ret^2 approx Variance)
            # y: (Batch, Horizon, Assets)
            # target_vol: (Batch, Horizon, Assets) -> 这是方差
            target_variance = y ** 2
            
            # --- 手动 Forward ---
            _, (h_n, _) = model.lstm(x)
            context = h_n[-1]
            batch_size = x.size(0)
            
            # 只用 L Head (协方差预测头)
            L_flat = model.L_head(context)
            L_pred = L_flat.view(batch_size, cfg.PREDICT_HORIZON, cfg.NUM_ASSETS, cfg.NUM_ASSETS)
            
            # 处理 L 保证合法性
            mask = torch.tril(torch.ones_like(L_pred))
            L_pred = L_pred * mask
            diag_mask = torch.eye(cfg.NUM_ASSETS, device=cfg.DEVICE).view(1, 1, cfg.NUM_ASSETS, cfg.NUM_ASSETS)
            L_pred = L_pred + diag_mask * (F.softplus(L_pred) + 1e-5 - L_pred)
            
            # 计算预测的方差 (Predicted Variance)
            # Sigma = L * L.T
            # 对角线元素 Sigma_ii = sum(L_ik^2)
            # 我们只需要对角线部分来做 MSE 监督（简化版 Risk Parity 只需要方差）
            Sigma_diag = torch.diagonal(L_pred @ L_pred.transpose(-1, -2), dim1=-2, dim2=-1)
            
            # Loss: 预测方差 vs 真实方差proxy
            loss = criterion(Sigma_diag, target_variance)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'VolMSE': f"{loss.item():.8f}"})
            
        print(f"Epoch {epoch+1} Avg Vol MSE: {epoch_loss / len(train_loader):.8f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Volatility 基准模型已保存至: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_volatility()