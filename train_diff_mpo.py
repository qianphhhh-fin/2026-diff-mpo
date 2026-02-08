"""
脚本名称: train_diff_mpo.py
功能描述: 
    Diff-MPO 模型的独立训练脚本 (Pre-training)。
    用于在回测开始前，在整个训练集上对模型进行预训练，或者进行超参数调试。

主要功能:
    1. calc_composite_loss: 定义复合损失函数 (Sortino + MaxDD + Turnover)。
    2. train: 主训练循环。
       - 加载数据。
       - 前向传播 (Model -> Solver)。
       - 计算 Loss (包含 MSE, MPO Loss, CVaR Penalty)。
       - 反向传播与参数更新。
       - 保存训练好的模型权重。

输入:
    - data_loader.py 提供的数据。
    - config.py 的配置。

输出:
    - 训练好的模型文件 'models/diff_mpo_sharpe.pth'。
    - 训练 Loss 曲线图 'diff_mpo_training_loss.png'。

与其他脚本的关系:
    - 独立运行的入口脚本。
    - 其 calc_composite_loss 函数被 strategy.py 复用。
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from config import cfg
from data_loader import load_and_process_data
from model import MPO_Network_Factor

# ==========================
# 1. 定义复合损失函数 (Composite Loss)
# ==========================
def calc_composite_loss(w_plan, y_future, w_prev, cost_coeff=0.001):
    """
    计算包含 Sortino、MaxDD 和 Turnover 惩罚的复合 Loss
    
    参数:
    w_plan: (Batch, Horizon, Assets) -> Solver 输出的未来 H 步权重
    y_future: (Batch, Horizon, Assets) -> 真实未来收益率
    w_prev: (Batch, Assets) -> 初始持仓
    """
    batch_size = w_plan.size(0)
    horizon = w_plan.size(1)
    
    # --- A. 构建完整的资金流 ---
    # 1. 计算换手率 (Turnover)
    # 拼接 w_prev 和 w_plan，形成完整路径 [w_0, w_1, ..., w_H]
    w_prev_expanded = w_prev.unsqueeze(1) # (B, 1, N)
    w_all = torch.cat([w_prev_expanded, w_plan], dim=1) # (B, H+1, N)
    
    # 计算每一步的换手: |w_t - w_{t-1}|
    # dim=2 (Assets) 求和
    turnover_seq = torch.norm(w_all[:, 1:] - w_all[:, :-1], p=1, dim=2) # (B, H)
    
    # 2. 计算净收益率 (Net Returns)
    # Gross Ret = sum(w * y)
    gross_ret_seq = (w_plan * y_future).sum(dim=2) # (B, H)
    # Net Ret = Gross - Cost
    net_ret_seq = gross_ret_seq - cost_coeff * turnover_seq # (B, H)
    
    # --- B. 计算各个 Loss 组件 ---
    
    # Component 1: Sortino Ratio (代替 Sharpe)
    # 只惩罚下行波动
    mean_ret = net_ret_seq.mean(dim=1)
    # 筛选出小于 0 的收益，计算其平方均值作为下行风险
    downside_returns = torch.clamp(net_ret_seq, max=0.0)
    downside_std = torch.sqrt(torch.mean(downside_returns**2, dim=1) + 1e-8)
    
    # Sortino = Mean / Downside_Dev
    sortino = (mean_ret - 0.0) / (downside_std + 1e-6)
    loss_sortino = -sortino.mean()
    
    # Component 2: Max Drawdown Penalty (最大回撤惩罚)
    # 计算累计净值曲线 Wealth Curve
    # log(1+r) 近似 r，累加得到 log wealth
    # cum_log_ret = torch.cumsum(torch.log1p(net_ret_seq), dim=1)
    # 找到截止当前的最高点 (Running Max)
    # PyTorch 的 cummax 返回 (values, indices)
    # running_max, _ = torch.cummax(cum_log_ret, dim=1)
    # 计算回撤: Current - Max
    # drawdowns = cum_log_ret - running_max
    # 找到最大回撤 (最小值)
    # max_dd, _ = torch.min(drawdowns, dim=1) # (B,) 注意这是负数，比如 -0.1
    
    # 惩罚项：回撤越深(负得越多)，Loss越大
    # 使用平方惩罚，让模型极度厌恶深回撤
    # loss_max_dd = torch.mean(max_dd**2) 
    
    # Component 3: Turnover Smoothing (换手率平滑)
    # 惩罚权重的剧烈跳变 (L2 Norm of diff)
    # 即使 Solver 允许换手，神经网络也不应该输出震荡的信号
    # w_diff_sq = torch.sum((w_all[:, 1:] - w_all[:, :-1])**2, dim=2) # (B, H)
    # loss_smoothing = torch.mean(w_diff_sq)
    
    # --- C. 总 Loss ---
    # 从 Config 读取系数
    # lambda_dd = getattr(cfg, 'LOSS_GAMMA_DD', 5.0)
    # lambda_turnover = getattr(cfg, 'LOSS_GAMMA_TURNOVER', 1.0)
    
    # [MODIFIED] 只保留 Sortino，其他注释掉
    total_loss = loss_sortino # + lambda_dd * loss_max_dd + lambda_turnover * loss_smoothing
    
    return total_loss, {
        "Sortino": -loss_sortino.item(), # 记录正的 Sortino 方便看
        # "MaxDD_Penalty": loss_max_dd.item(),
        # "Smooth_Penalty": loss_smoothing.item()
    }

# ==========================
# 2. 训练主循环
# ==========================
# def train():
#     # 准备数据
#     train_loader, test_loader, _ = load_and_process_data()
    
#     # 初始化模型
#     model = MPO_Network_Factor().to(cfg.DEVICE).double() 
#     optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    
#     print(f"🚀 模型已加载至 {cfg.DEVICE}. 开始训练 {cfg.EPOCHS} Epochs...")
#     print(f"   Loss Config: Gamma_DD={getattr(cfg, 'LOSS_GAMMA_DD', 5.0)}, Gamma_Turnover={getattr(cfg, 'LOSS_GAMMA_TURNOVER', 1.0)}")
    
#     loss_history = []
    
#     for epoch in range(cfg.EPOCHS):
#         model.train()
#         epoch_loss = 0
        
#         # 记录细分指标用于监控
#         metrics_sum = {"Sortino": 0, "MaxDD_Penalty": 0, "Smooth_Penalty": 0}
        
#         pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}")
        
#         for batch_idx, (x, y) in enumerate(pbar):
#             x, y = x.to(cfg.DEVICE).double(), y.to(cfg.DEVICE).double()
            
#             # 初始持仓：假设每天早上都从 1/N 开始 (简化假设)
#             # 在更严谨的实现中，可以用 LSTM state 传递真实的 w_prev，但在 Batch 训练中很难做到
#             w_prev = torch.ones(x.size(0), cfg.NUM_ASSETS, device=cfg.DEVICE, dtype=torch.double) / cfg.NUM_ASSETS
            
#             # --- Forward ---
#             w_plan, mu_pred, L_pred = model(x, w_prev)
            
#             # --- Composite Loss ---
#             loss_mpo, metrics = calc_composite_loss(w_plan, y, w_prev, cost_coeff=cfg.COST_COEFF)
            
#             # --- Auxiliary Losses ---
#             # 1. MSE Loss for mu prediction
#             # [REMOVED] 模型不再预测有意义的 mu，MSE Loss 已无意义
#             # loss_mse = torch.nn.functional.mse_loss(mu_pred, y)
            
#             # 2. Realized Risk Penalty (CVaR Violation)
#             # [REMOVED] 暂时只优化纯 Sortino，移除辅助约束
#             # port_ret = (w_plan * y).sum(dim=2)
#             # violation = torch.relu(-port_ret - cfg.CVAR_LIMIT)
#             # loss_realized_risk = torch.mean(violation**2)
            
#             # Total Loss
#             # [MODIFIED] 只包含 loss_mpo (即 Sortino)
#             loss = loss_mpo # + 1000.0 * loss_mse + 20.0 * loss_realized_risk
            
#             # --- Backward ---
#             optimizer.zero_grad()
#             loss.backward()
            
#             # 梯度裁剪 (关键！防止 MaxDD 导致的梯度爆炸)
#             grad_clip = getattr(cfg, 'GRAD_CLIP', 0.5)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            
#             optimizer.step()
            
#             epoch_loss += loss.item()
            
#             # 累加监控指标
#             for k, v in metrics.items():
#                 metrics_sum[k] += v
                
#             pbar.set_postfix({'Loss': f"{loss.item():.2f}", 'Sortino': f"{metrics['Sortino']:.2f}"})
        
#         avg_loss = epoch_loss / len(train_loader)
#         loss_history.append(avg_loss)
        
#         # 打印本 Epoch 的平均指标
#         avg_sortino = metrics_sum["Sortino"] / len(train_loader)
#         avg_dd_pen = metrics_sum["MaxDD_Penalty"] / len(train_loader)
#         print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Avg Sortino: {avg_sortino:.4f} | DD Pen: {avg_dd_pen:.4f}")
        
#     # ==========================
#     # 3. 结果保存
#     # ==========================
#     plt.figure(figsize=(10, 5))
#     plt.plot(loss_history, label='Composite Loss')
#     plt.title('Training Progress (Composite Loss)')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('diff_mpo_training_loss.png')
#     print("📈 训练完成，Loss 曲线已保存。")
    
#     SAVE_PATH = 'models/diff_mpo_sharpe.pth' 
#     torch.save(model.state_dict(), SAVE_PATH)
#     print(f"🏆 Diff-MPO (Ours) 模型已保存至: {SAVE_PATH}")


# if __name__ == "__main__":
#     train()