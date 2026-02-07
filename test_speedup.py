
"""
脚本名称: test_speedup.py
功能描述: 
    独立测试脚本，用于比较 Diff-MPO 在推理阶段 (Inference) 的不同实现方式的速度和结果一致性。
    重点比较以下两种模式：
    1. Baseline (Old): 模拟 eval_rolling_all.py 中的逐日推理，每次 Batch=1，频繁 IO。
    2. Optimized (New): 批量推理模式，一次性处理多个时间步的数据，利用矩阵并行加速。

    注意：Optimized 模式在回测中意味着我们一次性预测未来 N 天，但在真实回测中，
    我们只能利用 t 时刻的信息预测 t+1，不能偷看未来。
    因此，"合法"的加速方式是：
    - 预先准备好所有 t 时刻的 feature tensor (Batch=Total_Days)。
    - 一次性喂给模型 (Batch Inference)。
    - 得到所有天数的预测结果 (mu, L)。
    - 然后只在 Solver 层面进行逐日迭代 (因为 w_prev 依赖于前一天的 w)。
    - 甚至 Solver 也可以并行化？不，Solver 是串行的 (Stateful)，除非我们忽略 Transaction Cost 的动态影响。
    - 但大部分耗时可能在 LSTM 和 Heads 上，所以 Batch Inference 能加速 Model 部分。

    本脚本测试：
    "逐日 Model + 逐日 Solver" vs "批量 Model + 逐日 Solver"
"""

import torch
import pandas as pd
import numpy as np
import time
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import cfg
from data_loader import MPODataset
from model import MPO_Network_Factor
from mpo_solver import DifferentiableMPO_cvx

def run_speedup_benchmark():
    print("🏎️ 开始 Diff-MPO 推理速度基准测试 (Speedup Benchmark)...")
    print(f"   Device: {cfg.DEVICE}")
    
    # ==========================
    # 1. 准备数据 (Data Chunk)
    # ==========================
    # 加载 1000 天的数据用于测试
    # 确保有足够 Lookback
    N_DAYS = 1000
    print(f"   -> Loading {N_DAYS} days of data...")
    
    df_raw = pd.read_csv(cfg.DATA_PATH, index_col=0, parse_dates=True)
    all_features = df_raw.values
    all_returns = df_raw[cfg.ASSETS].values
    
    # 构造 N_DAYS 个样本
    # 每个样本 Input: (Lookback, Feat)
    X_list = []
    Y_list = []
    
    start_idx = cfg.LOOKBACK_WINDOW
    end_idx = start_idx + N_DAYS
    
    # 为了模拟 eval_rolling_all，我们需要逐个切片
    for t in range(start_idx, end_idx):
        x_window = all_features[t - cfg.LOOKBACK_WINDOW : t]
        # 简单归一化 (模拟 scaler)
        x_window = (x_window - x_window.mean(axis=0)) / (x_window.std(axis=0) + 1e-6)
        X_list.append(x_window)
        Y_list.append(all_returns[t]) # Dummy Y
        
    X_tensor = torch.tensor(np.array(X_list), dtype=torch.double) # (N, Lookback, Feat)
    
    print(f"   -> Input Tensor Shape: {X_tensor.shape}")
    
    # ==========================
    # 2. 初始化模型
    # ==========================
    model = MPO_Network_Factor().to(cfg.DEVICE).double()
    model.eval()
    
    # 初始持仓
    w_init = torch.ones(1, cfg.NUM_ASSETS, device=cfg.DEVICE, dtype=torch.double) / cfg.NUM_ASSETS
    
    # ==========================
    # 3. Baseline: 逐日循环 (Day-by-Day Loop)
    # ==========================
    print("\n🐢 Running Baseline (Day-by-Day)...")
    
    # 必须把 Tensor 拆回 CPU 列表来模拟真实场景的 IO
    # 在 eval_rolling_all 中，每次是从 numpy -> tensor -> gpu
    X_cpu_list = [t.unsqueeze(0) for t in X_tensor] 
    
    w_prev = w_init.clone()
    results_baseline = []
    
    start_time = time.time()
    
    for i in tqdm(range(N_DAYS), desc="Baseline"):
        # 1. IO Overhead
        x_day = X_cpu_list[i].to(cfg.DEVICE)
        
        # 2. Model Inference
        with torch.no_grad():
            w_plan, _, _ = model(x_day, w_prev)
        
        # 3. State Update
        w_action = w_plan[:, 0, :] # (1, N)
        w_prev = w_action
        
        # 4. Result Retrieval
        results_baseline.append(w_action.cpu().numpy())
        
    time_baseline = time.time() - start_time
    print(f"   -> Baseline Time: {time_baseline:.4f}s ({N_DAYS/time_baseline:.1f} iter/s)")
    
    # ==========================
    # 4. Optimized: 批量模型 + 串行求解 (Batch Model + Serial Solver)
    # ==========================
    print("\n🐇 Running Optimized (Batch Model + Serial Solver)...")
    
    start_time_opt = time.time()
    
    # 1. Batch Model Inference
    # 一次性将所有 X 推入 GPU 计算 mu 和 L
    # Batch Size 可以很大，比如 1000
    BATCH_SIZE_LARGE = 256
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE_LARGE, shuffle=False)
    
    mu_list = []
    L_list = []
    
    with torch.no_grad():
        for (x_batch,) in loader:
            x_batch = x_batch.to(cfg.DEVICE)
            
            # 我们只用 model 的 Encoder 和 Head 部分
            # 这一步需要拆解 model.forward，或者给 model 加一个只输出参数的接口
            # 这里我们手动调用 model 的子模块 (White-box Optimization)
            
            # --- Model Internal Logic ---
            _, (h_n, _) = model.lstm(x_batch)
            context = h_n[-1]
            
            mu = model.mu_head(context).view(-1, cfg.PREDICT_HORIZON, cfg.NUM_ASSETS)
            
            B_flat = model.B_head(context)
            B = B_flat.view(-1, cfg.PREDICT_HORIZON, cfg.NUM_ASSETS, cfg.NUM_FACTORS)
            D_flat = model.D_head(context)
            D = torch.nn.functional.softplus(D_flat) + 1e-3
            D = D.view(-1, cfg.PREDICT_HORIZON, cfg.NUM_ASSETS)
            
            factor_cov = torch.matmul(B, B.transpose(-1, -2))
            idiosyncratic_cov = torch.diag_embed(D**2)
            Sigma = factor_cov + idiosyncratic_cov
            
            epsilon_eye = 1e-5 * torch.eye(cfg.NUM_ASSETS, device=cfg.DEVICE).view(1, 1, cfg.NUM_ASSETS, cfg.NUM_ASSETS)
            Sigma_stabilized = Sigma + epsilon_eye
            try:
                L = torch.linalg.cholesky(Sigma_stabilized)
            except RuntimeError:
                L = torch.diag_embed(D + 1e-3)
            # ----------------------------
            
            mu_list.append(mu)
            L_list.append(L)
            
    # 拼接所有预测参数
    mu_all = torch.cat(mu_list, dim=0)
    L_all = torch.cat(L_list, dim=0)
    
    # 2. Serial Solver Loop
    # 这一步无法并行，因为 w_t 依赖 w_{t-1}
    # 但我们省去了 LSTM 的重复计算和 IO
    
    w_prev = w_init.clone()
    results_opt = []
    
    # 提取 Solver
    solver = model.mpo_layer
    cvar_limit = torch.tensor(cfg.CVAR_LIMIT, device=cfg.DEVICE, dtype=torch.double).expand(1)
    
    for i in tqdm(range(N_DAYS), desc="Optimized"):
        # 取出第 i 天的参数 (1, H, N)
        mu_day = mu_all[i:i+1]
        L_day = L_all[i:i+1]
        
        # 纯 Solver 计算
        # DifferentiableMPO_cvx 调用 CvxpyLayer
        with torch.no_grad():
            w_plan = solver(mu_day, L_day, w_prev, cvar_limit)
        
        w_action = w_plan[:, 0, :]
        w_prev = w_action
        results_opt.append(w_action.cpu().numpy())
        
    time_opt = time.time() - start_time_opt
    print(f"   -> Optimized Time: {time_opt:.4f}s ({N_DAYS/time_opt:.1f} iter/s)")
    
    # ==========================
    # 5. 结果对比
    # ==========================
    print("\n⚖️ 结果一致性检查...")
    
    res_base = np.concatenate(results_baseline, axis=0)
    res_opt = np.concatenate(results_opt, axis=0)
    
    diff = np.abs(res_base - res_opt).max()
    print(f"   -> Max Difference: {diff:.6e}")
    
    if diff < 1e-6:
        print("   ✅ 结果一致！优化方案有效。")
        speedup = time_baseline / time_opt
        print(f"   🚀 加速比: {speedup:.2f}x")
    else:
        print("   ❌ 结果不一致，请检查逻辑。")

if __name__ == "__main__":
    run_speedup_benchmark()
