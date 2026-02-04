import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from config import cfg
from data_loader import load_and_process_data
from model import MPO_Network

# 确保模型保存目录存在
os.makedirs('models', exist_ok=True)
MODEL_SAVE_PATH = 'models/baseline_mse_model.pth'

def train_mse():
    # 1. 准备数据
    train_loader, _, _ = load_and_process_data()
    
    # 2. 初始化模型
    model = MPO_Network().to(cfg.DEVICE).double()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.MSELoss()
    
    print(f"📉 [Benchmark] 开始训练 MSE 模型 (Two-Stage)...")
    print(f"   目标: 让预测的 mu 尽可能接近真实收益率")
    print(f"   注意: 训练阶段跳过 Solver，速度会很快。")

    loss_history = []
    
    for epoch in range(cfg.EPOCHS):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"MSE Epoch {epoch+1}/{cfg.EPOCHS}")
        
        for x, y in pbar:
            x, y = x.to(cfg.DEVICE).double(), y.to(cfg.DEVICE).double()
            
            # --- 关键修改：手动 Forward，跳过 Solver ---
            # 我们不需要生成 w_plan，只需要 mu_pred
            # 这样既快，又符合 Two-Stage 的定义（预测与优化解耦）
            
            # 1. LSTM 编码
            _, (h_n, _) = model.lstm(x)
            context = h_n[-1]
            
            # 2. 只调用 Mu Head
            batch_size = x.size(0)
            mu_pred = model.mu_head(context)
            mu_pred = mu_pred.view(batch_size, cfg.PREDICT_HORIZON, cfg.NUM_ASSETS)
            
            # 3. 计算 MSE Loss
            # y 是 (Batch, Horizon, Assets)
            loss = criterion(mu_pred, y)
            
            # 4. 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'MSE': f"{loss.item():.6f}"})
            
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1} Avg MSE: {avg_loss:.6f}")

    # 保存
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ MSE 基准模型已保存至: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_mse()