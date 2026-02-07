import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
from config import cfg
from data_loader import load_and_process_data
from model import MPO_Network
from mpo_solver import DifferentiableMPO # 确保能导入

# 设置设备
DEVICE = cfg.DEVICE
plt.style.use('seaborn-v0_8')

def run_neuron_analysis():
    print("🧠 正在进行 Diff-MPO 神经元阻断分析 (Neuron Ablation)...")
    
    # 1. 准备数据
    _, test_loader, _ = load_and_process_data()
    df = pd.read_csv(cfg.DATA_PATH, index_col=0, parse_dates=True)
    split_date = pd.Timestamp(cfg.TRAIN_SPLIT_DATE)
    
    # 确保有足够的数据对应 test_loader
    # test_loader 的长度可能因为 batch drop last 而略短，我们需要对齐
    test_dates = df.index[df.index >= split_date]
    
    # 获取测试集期间的市场基准 (用于对比画图)
    market_prices = df.loc[test_dates, cfg.ASSETS[0]] 
    market_cum_ret = (1 + market_prices.pct_change().fillna(0)).cumprod()

    # 2. 加载模型
    model = MPO_Network().to(DEVICE).double()
    try:
        model.load_state_dict(torch.load('models/diff_mpo_sharpe.pth', map_location=DEVICE))
    except FileNotFoundError:
        print("❌ 错误: 未找到模型文件 models/diff_mpo_sharpe.pth，请先运行 eval.py 或 train.py")
        return

    model.eval()
    
    # 获取隐藏层维度
    hidden_dim = model.lstm.hidden_size
    print(f"   检测到 LSTM 隐藏层维度: {hidden_dim}")

    # 初始化 Solver (在 CPU 上运行以避免 cvxpylayers 的 CUDA bug)
    # 注意：不要把这个 solver 放到 GPU 上，因为我们要喂给它 CPU 数据
    solver = DifferentiableMPO()

    # ==========================================
    # 3. 定义评估核心 (支持神经元阻断)
    # ==========================================
    def get_performance(mask_neuron_idx=None):
        """
        mask_neuron_idx: int, 要阻断的神经元索引 (0 ~ 63)
        返回: Sharpe Ratio
        """
        all_net_ret = []
        # current_w 放在 CPU 上，方便后续直接处理
        current_w = torch.ones(cfg.NUM_ASSETS, dtype=torch.double) / cfg.NUM_ASSETS
        
        # 遍历测试集
        for x_batch, y_batch in test_loader:
            # 输入数据依然去 GPU (为了 LSTM 推理速度)
            x_batch = x_batch.to(DEVICE).double()
            # y_batch 也去 GPU，但稍后计算收益时我们会拉回 CPU
            y_batch = y_batch.to(DEVICE).double()
            
            batch_size = x_batch.size(0)
            
            with torch.no_grad():
                # 1. LSTM Forward (GPU)
                _, (h_n, _) = model.lstm(x_batch)
                context = h_n[-1] # (Batch, Hidden)
                
                # --- 🧠 关键手术：神经元阻断 ---
                if mask_neuron_idx is not None:
                    context[:, mask_neuron_idx] = 0.0
                
                # 2. Heads (GPU)
                mu_pred = model.mu_head(context).view(batch_size, cfg.PREDICT_HORIZON, cfg.NUM_ASSETS)
                L_flat = model.L_head(context)
                L_pred = L_flat.view(batch_size, cfg.PREDICT_HORIZON, cfg.NUM_ASSETS, cfg.NUM_ASSETS)
                
                # 构造 L (GPU)
                mask = torch.tril(torch.ones_like(L_pred))
                L_pred = L_pred * mask
                diag_mask = torch.eye(cfg.NUM_ASSETS, device=DEVICE).view(1, 1, cfg.NUM_ASSETS, cfg.NUM_ASSETS)
                L_pred = L_pred + diag_mask * (torch.nn.functional.softplus(L_pred) + 1e-5 - L_pred)
                
                # 3. Solver (CPU)
                # ⚠️【修复关键】⚠️：将所有 Tensor 移回 CPU 再传给 Solver
                # cvxpylayers 在处理 CUDA tensor 转 numpy 时经常报错
                mu_cpu = mu_pred.cpu()
                L_cpu = L_pred.cpu()
                w_prev_cpu = current_w.repeat(batch_size, 1) # current_w 已经在 CPU 了
                
                # 调用 Solver (在 CPU 上解)
                w_plan = solver(mu_cpu, L_cpu, w_prev_cpu)
                
                # 4. 计算收益 (CPU)
                w_t = w_plan[:, 0, :] # (Batch, Assets)
                y_t = y_batch[:, 0, :].cpu() # 真实收益移回 CPU
                
                # 计算 Gross Return
                ret = (w_t * y_t).sum(dim=1)
                all_net_ret.append(ret.numpy())
                
        all_net_ret = np.concatenate(all_net_ret)
        mean = np.mean(all_net_ret) * 252
        std = np.std(all_net_ret) * np.sqrt(252)
        return mean / (std + 1e-6)

    # ==========================================
    # 4. 执行分析
    # ==========================================
    
    # A. 计算 Baseline
    print("   计算 Baseline 性能...")
    base_sharpe = get_performance(mask_neuron_idx=None)
    print(f"   ✅ Baseline Sharpe: {base_sharpe:.4f}")
    
    # B. 遍历所有神经元
    importance = []
    print(f"   正在扫描 {hidden_dim} 个神经元...")
    
    for i in tqdm(range(hidden_dim)):
        s = get_performance(mask_neuron_idx=i)
        # Drop > 0 表示该神经元对正向收益有贡献（阻断它导致 Sharpe 下降）
        drop = (base_sharpe - s) / (abs(base_sharpe) + 1e-6)
        importance.append(drop)
    
    importance = np.array(importance)
    
    # C. 找到 Top-K 神经元
    top_k_indices = np.argsort(importance)[::-1][:5] # 取下降幅度最大的前5个
    print("\n🏆 最重要的功能性神经元 (Top 5):")
    for idx in top_k_indices:
        print(f"   Neuron #{idx}: Sharpe 下降 {importance[idx]:.2%}")
        
    top_neuron_idx = top_k_indices[0]
    
    # ==========================================
    # 5. 可视化 Top 神经元的激活行为
    # ==========================================
    print(f"\n📸 正在绘制 Neuron #{top_neuron_idx} 的时序激活图...")
    
    activations = []
    # 再次遍历提取激活值
    for x_batch, _ in test_loader:
        x_batch = x_batch.to(DEVICE).double()
        with torch.no_grad():
            _, (h_n, _) = model.lstm(x_batch)
            act = h_n[-1][:, top_neuron_idx].cpu().numpy()
            activations.append(act)
            
    activations = np.concatenate(activations)
    
    # 对齐长度
    plot_len = min(len(activations), len(test_dates))
    dates = test_dates[:plot_len]
    mkt_curve = market_cum_ret.iloc[:plot_len]
    act_data = activations[:plot_len]
    
    # 绘图
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 绘制市场曲线
    color = 'tab:gray'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Market Cumulative Return', color=color)
    ax1.plot(dates, mkt_curve, color=color, alpha=0.5, label='Market (Benchmark)', linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # 绘制神经元激活值
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel(f'Neuron #{top_neuron_idx} Activation', color=color, fontsize=12, fontweight='bold')
    # 使用散点图或细线，因为激活值波动可能很快
    ax2.plot(dates, act_data, color=color, alpha=0.8, linewidth=1.0, label=f'Neuron #{top_neuron_idx}')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f'Mechanistic Analysis: What does Neuron #{top_neuron_idx} do?', fontsize=14)
    fig.autofmt_xdate()
    plt.tight_layout()
    
    save_path = 'neuron_analysis.png'
    plt.savefig(save_path, dpi=100)
    print(f"✅ 结果已保存至 {save_path}")
    
    # 保存 CSV
    df_imp = pd.DataFrame({'Neuron_ID': range(hidden_dim), 'Importance_Drop': importance})
    df_imp.to_csv('neuron_importance.csv', index=False)

if __name__ == "__main__":
    run_neuron_analysis()