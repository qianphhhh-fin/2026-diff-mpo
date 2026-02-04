import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import cfg
from data_loader import load_and_process_data
from model import MPO_Network

# ==========================
# 1. 定义 Sharpe Loss
# ==========================
def sharpe_loss(w_plan, y_future, transaction_cost_coeff=0.0005):
    """
    w_plan: (Batch, Horizon, Assets) 优化器产出的权重
    y_future: (Batch, Horizon, Assets) 未来的真实收益率
    """
    # 1. 计算组合收益 R_p = w * r
    # (Batch, H, N) * (Batch, H, N) -> sum -> (Batch, H)
    portfolio_ret = (w_plan * y_future).sum(dim=2)
    
    # 2. 计算交易成本 (简化版: 既然是 Loss，我们希望惩罚高换手)
    # 这一步在 Solver 里已经惩罚过了，但在 Loss 里再加一次双保险
    # 这里为了简便，主要看纯收益的夏普，把成本隐含在 w 的选择中
    # 如果 w 乱变，Solver 里的 cost 项会很大，导致 w 被约束，
    # 间接导致 portfolio_ret 变差 (因为没钱赚了)
    
    # 3. 计算 Sharpe
    # 按 Batch 计算平均收益和标准差
    # 假设无风险利率为 0 (或者已经是超额收益)
    mean_ret = portfolio_ret.mean(dim=1) # (Batch,)
    std_ret = portfolio_ret.std(dim=1) + 1e-6 # (Batch,)
    
    sharpe = mean_ret / std_ret
    
    # 目标是最大化 Sharpe => 最小化 -Sharpe
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
            # 我们用真实的未来收益 y 来评价 w_plan 好不好
            loss = sharpe_loss(w_plan, y)
            
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