import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import cfg
from data_loader import load_and_process_data
from model import MPO_Network

# 修改 train.py

def sharpe_loss(w_plan, y_future, w_prev, cost_coeff=0.01): # <--- 传入 w_prev 和 cost_coeff
    """
    w_plan: (Batch, Horizon, Assets)
    y_future: (Batch, Horizon, Assets)
    w_prev: (Batch, Assets)
    """
    # 1. 计算毛收益
    gross_ret = (w_plan * y_future).sum(dim=2) # (Batch, Horizon)
    
    # 2. 计算交易成本 (与 Solver 保持一致的 L1 Norm)
    # 注意：这里需要计算 w_plan[t] - w_plan[t-1] 的完整序列
    # 构造完整的权重路径: [w_prev, w_0, w_1, ..., w_{H-1}]
    # 这一步稍微有点繁琐，但必须做
    
    # 将 w_prev 扩展为 (Batch, 1, Assets) 以便拼接
    w_prev_expanded = w_prev.unsqueeze(1)
    
    # 拼接: (Batch, H+1, Assets)
    w_all = torch.cat([w_prev_expanded, w_plan], dim=1)
    
    # 计算差分: |w_t - w_{t-1}|
    turnover = torch.norm(w_all[:, 1:] - w_all[:, :-1], p=1, dim=2) # (Batch, Horizon)
    
    # 3. 计算净收益 (Net Return)
    net_ret = gross_ret - cost_coeff * turnover
    
    # 4. 计算 Sharpe (基于净收益)
    mean_ret = net_ret.mean(dim=1)
    std_ret = net_ret.std(dim=1) + 1e-6
    sharpe = mean_ret / std_ret
    
    return -sharpe.mean()

# ==========================
# 2. 训练主循环
# ==========================
def train():
    # 准备数据
    train_loader, test_loader, _ = load_and_process_data()
    
    # 初始化模型
    model = MPO_Network().to(cfg.DEVICE).double() # CVXPY 需要 Double 精度
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    
    print(f"🚀 模型已加载至 {cfg.DEVICE}. 开始训练 {cfg.EPOCHS} Epochs...")
    
    loss_history = []
    
    for epoch in range(cfg.EPOCHS):
        model.train()
        epoch_loss = 0
        
        # 进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}")
        
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(cfg.DEVICE).double(), y.to(cfg.DEVICE).double()
            
            # 初始持仓：假设每个 Batch 开始时是空仓或者均匀持仓
            # 在真实的 LSTM 序列训练中，应该把上一个 Batch 的 w 传进来
            # 这里为了简化，假设每天早上都从 1/N 开始调仓 (或者全现金)
            # 更好的做法是: w_prev = torch.ones(...) / N
            w_prev = torch.ones(x.size(0), cfg.NUM_ASSETS, device=cfg.DEVICE, dtype=torch.double) / cfg.NUM_ASSETS
            
            # --- Forward ---
            # w_plan 是 Solver 解出来的最优路径
            w_plan, mu_pred, L_pred = model(x, w_prev)
            
            # --- Loss ---
            # 使用新的带成本的 Loss，传入 cfg.COST_COEFF
            loss = sharpe_loss(w_plan, y, w_prev, cost_coeff=cfg.COST_COEFF)
            
            # --- Backward ---
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪 (防止 LSTM 梯度爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'SharpeLoss': f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f} (Implied Sharpe: {-avg_loss:.4f})")
        
    # ==========================
    # 3. 简单的结果可视化
    # ==========================
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Negative Sharpe Loss')
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('diff_mpo_training_loss.png')
    print("📈 训练完成，Loss 曲线已保存至 training_loss.png")
    
    # 保存模型

    SAVE_PATH = 'models/diff_mpo_sharpe.pth'  # 科学命名
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"🏆 Diff-MPO (Ours) 模型已保存至: {SAVE_PATH}")


if __name__ == "__main__":
    train()