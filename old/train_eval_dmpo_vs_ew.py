import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
import os
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

# 引入基础配置
from config import cfg
from data_loader import MPODataset
# 注意：mpo_solver 必须引用，因为我们需要 DifferentiableMPO 层
from mpo_solver import DifferentiableMPO
# 引入 Loss 计算
from train_diff_mpo import calc_composite_loss 

# 设置风格
plt.style.use('seaborn-v0_8')
device = cfg.DEVICE

# ==========================================
# 0. 临时定义新模型：结构化协方差 (Factor Model)
# ==========================================
class MPO_Network_Factor(nn.Module):
    def __init__(self, input_dim, num_assets, hidden_dim=64):
        super(MPO_Network_Factor, self).__init__()
        
        # 1. 特征提取器
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim,
            num_layers=cfg.NUM_LAYERS,
            batch_first=True,
            dropout=cfg.DROPOUT
        )
        
        # 2. 预测头
        
        # Head A: 收益率 mu (不变)
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(32, cfg.PREDICT_HORIZON * num_assets)
        )
        
        # Head B: 结构化协方差 (Structured Covariance)
        # 假设存在 K 个隐因子 (Latent Factors)
        # 经验法则: K < N. 这里我们设定 K=3 (对应 Market, Size, Value 等宏观力量)
        self.num_factors = 3 
        self.num_assets = num_assets
        
        # 预测因子载荷 B (Batch, H, N, K)
        # 这代表每个资产对 3 个隐因子的敏感度
        self.B_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, cfg.PREDICT_HORIZON * num_assets * self.num_factors)
        )
        
        # 预测特异性波动 D (Batch, H, N)
        # 这代表每个资产特有的、不能被因子解释的波动
        self.D_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, cfg.PREDICT_HORIZON * num_assets)
        )
        
        # 3. 优化层
        self.mpo_layer = DifferentiableMPO()
        
    def forward(self, x, w_prev):
        batch_size = x.size(0)
        
        # --- Encoding ---
        _, (h_n, _) = self.lstm(x)
        context = h_n[-1]
        
        # --- Parameter Prediction ---
        
        # 1. Mu
        mu = self.mu_head(context)
        mu = mu.view(batch_size, cfg.PREDICT_HORIZON, self.num_assets)
        
        # 2. Sigma (Factor Model Construction)
        # B: (Batch, H, N, K)
        B_flat = self.B_head(context)
        B = B_flat.view(batch_size, cfg.PREDICT_HORIZON, self.num_assets, self.num_factors)
        
        # D: (Batch, H, N) -> 必须为正
        D_flat = self.D_head(context)
        D = F.softplus(D_flat) + 1e-4 # 保证大于0
        D = D.view(batch_size, cfg.PREDICT_HORIZON, self.num_assets)
        
        # 构造协方差矩阵 Sigma = B @ B.T + diag(D^2)
        # 这种构造方式天然保证 Sigma 是对称正定的 (SPSD)
        
        # B @ B.T -> (Batch, H, N, N)
        factor_cov = torch.matmul(B, B.transpose(-1, -2)) 
        
        # Idiosyncratic variance matrix
        # torch.diag_embed 会把 D^2 放到对角线上
        idiosyncratic_cov = torch.diag_embed(D**2)
        
        Sigma = factor_cov + idiosyncratic_cov
        
        # --- Cholesky Decomposition ---
        # Solver 需要 L (where Sigma = L @ L.T)
        # 为了数值稳定性，加一个小扰动
        Sigma_stabilized = Sigma + 1e-6 * torch.eye(self.num_assets, device=x.device).view(1, 1, self.num_assets, self.num_assets)
        
        try:
            L = torch.linalg.cholesky(Sigma_stabilized)
        except RuntimeError:
            # 如果万一炸了（极少情况），回退到只用特异性波动 (对角阵)
            L = torch.diag_embed(D + 1e-3)
        
        # --- Optimization ---
        # 转移到 CPU 给 cvxpylayers 计算
        mu_cpu = mu.cpu()
        L_cpu = L.cpu()
        w_prev_cpu = w_prev.cpu()
        
        w_plan_cpu = self.mpo_layer(mu_cpu, L_cpu, w_prev_cpu)
        w_plan = w_plan_cpu.to(x.device)
        
        return w_plan, mu, L


class MPO_Transformer_Factor(nn.Module):
    def __init__(self, input_dim, num_assets, lookback_window, hidden_dim=64, nhead=4, num_layers=2):
        super(MPO_Transformer_Factor, self).__init__()
        
        # 1. Input Projection & Positional Encoding
        # 将输入特征映射到 d_model 维度
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # 可学习的位置编码 (Learnable Positional Encoding)
        # Shape: (1, Lookback, Hidden) -> 广播到 Batch
        self.pos_encoder = nn.Parameter(torch.randn(1, lookback_window, hidden_dim) * 0.02)
        
        # 2. Transformer Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim * 4,
            dropout=cfg.DROPOUT,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. 预测头 (与 Factor Model 保持一致)
        self.num_assets = num_assets
        self.num_factors = 3 
        
        # Head A: 收益率 mu
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(32, cfg.PREDICT_HORIZON * num_assets)
        )
        
        # Head B: 结构化协方差 (B & D)
        # 因子载荷 B
        self.B_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, cfg.PREDICT_HORIZON * num_assets * self.num_factors)
        )
        # 特异性波动 D
        self.D_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, cfg.PREDICT_HORIZON * num_assets)
        )
        
        # 4. 优化层
        self.mpo_layer = DifferentiableMPO()
        
    def forward(self, x, w_prev):
        # x: (Batch, Lookback, Features)
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # --- Transformer Encoding ---
        # 1. Embedding + Positional Encoding
        # 注意：如果实际输入长度小于 lookback (极少情况)，切片 pos_encoder
        x_embed = self.embedding(x) + self.pos_encoder[:, :seq_len, :]
        
        # 2. Attention
        # Transformer 输出: (Batch, Lookback, Hidden)
        x_trans = self.transformer_encoder(x_embed)
        
        # 3. Aggregation
        # 取最后一个时间步的特征作为 Context (类似 LSTM 的 h_n[-1])
        context = x_trans[:, -1, :] 
        
        # --- Parameter Prediction (逻辑与 LSTM 版完全一致) ---
        
        # 1. Mu
        mu = self.mu_head(context).view(batch_size, cfg.PREDICT_HORIZON, self.num_assets)
        
        # 2. Sigma (Factor Model)
        B_flat = self.B_head(context)
        B = B_flat.view(batch_size, cfg.PREDICT_HORIZON, self.num_assets, self.num_factors)
        
        D_flat = self.D_head(context)
        D = F.softplus(D_flat) + 1e-4
        D = D.view(batch_size, cfg.PREDICT_HORIZON, self.num_assets)
        
        # Sigma = B @ B.T + D^2
        factor_cov = torch.matmul(B, B.transpose(-1, -2)) 
        idiosyncratic_cov = torch.diag_embed(D**2)
        Sigma = factor_cov + idiosyncratic_cov
        
        # Cholesky
        Sigma_stabilized = Sigma + 1e-6 * torch.eye(self.num_assets, device=x.device).view(1, 1, self.num_assets, self.num_assets)
        try:
            L = torch.linalg.cholesky(Sigma_stabilized)
        except RuntimeError:
            L = torch.diag_embed(D + 1e-3)
        
        # --- Optimization ---
        w_plan = self.mpo_layer(mu.cpu(), L.cpu(), w_prev.cpu()).to(x.device)
        
        return w_plan, mu, L
    

# ==========================================
# 1. 滚动回测主程序
# ==========================================
def run_walk_forward_experiment():
    print("⚔️ [Walk-Forward Experiment] Diff-MPO (Factor Model) vs 1/N ...")
    
    # 1. 准备全量数据
    df_raw = pd.read_csv(cfg.DATA_PATH, index_col=0, parse_dates=True)
    
    # 自动获取特征维度 (不再依赖 model.py 里的硬编码)
    all_features = df_raw.values 
    input_feature_dim = all_features.shape[1]
    
    # 目标资产
    all_returns = df_raw[cfg.ASSETS].values
    dates = df_raw.index
    
    # 设置回测时间轴 (建议从 2018 开始)
    TEST_START_YEAR = 2018 
    TEST_END_YEAR = dates[-1].year
    
    # 初始化记录器
    results_dmpo = [] 
    results_ew = []   
    
    # 初始化新模型 (Factor Model)
    # 注意：这里使用我们在脚本里定义的 MPO_Network_Factor
    # model = MPO_Network_Factor(
    #     input_dim=input_feature_dim,
    #     num_assets=cfg.NUM_ASSETS,
    #     hidden_dim=cfg.HIDDEN_DIM
    # ).to(device).double()
    # print(f"   模型架构: Factor Model (3 Latent Factors)")



    # 初始化新模型 (Transformer + Factor Model)
    # 注意：这里改用 Transformer 版本
    model = MPO_Transformer_Factor(
        input_dim=input_feature_dim,
        num_assets=cfg.NUM_ASSETS,
        lookback_window=cfg.LOOKBACK_WINDOW, # <--- 必须传入此参数
        hidden_dim=cfg.HIDDEN_DIM,
        nhead=4,      # 4 头注意力 (64/4=16 dim per head)
        num_layers=2  # 2 层 Transformer Block
    ).to(device).double()

    print(f"   模型架构: Transformer Factor Model")
    
    print(f"   输入维度: {input_feature_dim}, 资产数: {cfg.NUM_ASSETS}")
    print(f"   回测区间: {TEST_START_YEAR} -> {TEST_END_YEAR}")    

    
    # 2. 滚动循环
    for year in range(TEST_START_YEAR, TEST_END_YEAR + 1):
        print(f"\n📅 正在处理年份: {year} ...")
        
        # --- A. 时间切分 (Expanding Window) ---
        train_end_dt = pd.Timestamp(f"{year}-01-01")
        test_end_dt = pd.Timestamp(f"{year+1}-01-01")
        
        train_mask = dates < train_end_dt
        test_mask = (dates >= train_end_dt) & (dates < test_end_dt)
        
        if sum(test_mask) < cfg.LOOKBACK_WINDOW:
            print(f"   ⚠️ 数据不足，跳过 {year}")
            continue
            
        # --- B. 数据防泄漏处理 ---
        scaler = StandardScaler()
        X_train = scaler.fit_transform(all_features[train_mask])
        Y_train = all_returns[train_mask]
        
        X_test = scaler.transform(all_features[test_mask])
        Y_test = all_returns[test_mask] 
        test_dates_curr = dates[test_mask]
        
        # 构建 DataLoader
        train_ds = MPODataset(X_train, Y_train, cfg.LOOKBACK_WINDOW, cfg.PREDICT_HORIZON)
        # Drop last 保证 Batch 完整，shuffle 增加泛化
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)
        
        # --- C. 模型微调 (Fine-tune) ---
        # 每年 10 Epochs，学习率 5e-4
        optimizer = optim.Adam(model.parameters(), lr=5e-4) 
        model.train()
        
        train_pbar = tqdm(range(10), desc=f"   Training {year}", leave=False)
        for ep in train_pbar:
            ep_loss = 0
            for x_b, y_b in train_loader:
                x_b, y_b = x_b.to(device).double(), y_b.to(device).double()
                
                # 假设 w_prev 每天重置为 1/N 
                w_prev_b = torch.ones(x_b.size(0), cfg.NUM_ASSETS, device=device, dtype=torch.double) / cfg.NUM_ASSETS
                
                w_plan, _, _ = model(x_b, w_prev_b)
                
                # Loss 计算 (包含 MaxDD 和 Turnover 惩罚)
                loss, _ = calc_composite_loss(w_plan, y_b, w_prev_b, cost_coeff=cfg.COST_COEFF)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ep_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{ep_loss/len(train_loader):.4f}"})
            
# --- D. 样本外预测 (Rolling Inference) ---
        model.eval()
        
        # 1. 准备 Inference 数据
        # 修正：我们需要回溯更多的数据，以便做 Lag
        # infer_start_idx 保持不变，还是回溯 Lookback-1 (这是为了对齐 Y[0] 的时间点)
        test_start_idx = np.where(test_mask)[0][0]
        # 我们多取 1 天数据，防止数组越界，但核心是在 loop 里控制
        infer_start_idx = max(0, test_start_idx - cfg.LOOKBACK_WINDOW) 
        
        # 注意：这里取出的 X_infer_raw 包含了 "昨天" 和 "今天" 的数据
        X_infer_raw = all_features[infer_start_idx : test_start_idx + len(test_dates_curr)]
        X_infer_scaled = scaler.transform(X_infer_raw)
        
        Y_realized = Y_test 
        
        curr_w = torch.ones(1, cfg.NUM_ASSETS, device=device, dtype=torch.double) / cfg.NUM_ASSETS
        
        with torch.no_grad():
            for t in range(len(Y_realized)):
                # ================= 核心修复 =================
                # 目标：交易 Y_realized[t] (Day T 的收益)
                # 约束：只能看 Day T-1 及以前的数据
                
                # 你的 infer_start_idx 使得 X_infer_scaled 的对齐如下：
                # 假设 Lookback=60
                # 如果我们从 infer_start_idx = test_start - 60 开始取
                # 那么 X_infer_scaled[59] 是 Day T-1
                # 那么 X_infer_scaled[60] 是 Day T
                
                # 正确的窗口：[t : t + 60] 
                # 这里 t=0 时，取的是 [0:60]，最后一个点是 index 59 (Day T-1)
                # 这样就是：用过去 60 天 (截至昨天) 的数据，预测今天的仓位
                
                x_window = X_infer_scaled[t : t + cfg.LOOKBACK_WINDOW]
                
                # 之前的错误代码是取了 [t+1 : t+1+60] 或者类似的偏移导致看到了 Day T
                # 务必确保你的 X_infer_scaled 构造方式支持这种切片
                
                # 让我们用更直观的方式重新切片，防止索引混乱：
                # 我们需要 "截止到 t-1 的 Lookback 个数据"
                # 在 all_features 中，Day T 的索引是 test_start_idx + t
                # 所以我们需要 range: [test_start_idx + t - Lookback : test_start_idx + t]
                
                # 这种绝对索引法最安全，不会错：
                curr_abs_idx = test_start_idx + t
                x_raw_window = all_features[curr_abs_idx - cfg.LOOKBACK_WINDOW : curr_abs_idx]
                
                # 安全检查
                if len(x_raw_window) != cfg.LOOKBACK_WINDOW: 
                    # 只有年初第一天可能遇到这个问题（如果数据不够），通常不会
                    # 如果不够，就跳过或用 1/N
                    results_dmpo.append(results_ew[-1] if results_ew else 0.0) 
                    results_ew.append(0.0)
                    continue

                # 实时归一化 (用当年的 scaler)
                x_window_scaled = scaler.transform(x_raw_window)
                # ===========================================
                
                x_tensor = torch.tensor(x_window_scaled).unsqueeze(0).to(device).double()
                
                # 预测
                w_pred, _, _ = model(x_tensor, curr_w)
                w_action = w_pred[0, 0, :] 
                
                # 记录结果
                w_np = w_action.cpu().numpy()
                y_today = Y_realized[t]
                
                # ... (后续计算收益逻辑不变)
                w_prev_np = curr_w[0].cpu().numpy()
                turnover = np.sum(np.abs(w_np - w_prev_np))
                cost = turnover * cfg.COST_COEFF
                
                gross_ret = np.sum(w_np * y_today)
                net_ret = gross_ret - cost
                results_dmpo.append(net_ret)
                
                w_ew = np.ones(cfg.NUM_ASSETS) / cfg.NUM_ASSETS
                ret_ew = np.sum(w_ew * y_today)
                results_ew.append(ret_ew)
                
                curr_w = w_action.unsqueeze(0)
                
    # 3. 结果汇总
    print("\n📊 计算最终指标...")
    
    total_days = len(results_dmpo)
    idx = dates[-total_days:]
    
    s_dmpo = pd.Series(results_dmpo, index=idx)
    s_ew = pd.Series(results_ew, index=idx)
    
    # 净值曲线
    wealth_dmpo = (1 + s_dmpo).cumprod()
    wealth_ew = (1 + s_ew).cumprod()
    
    # 指标计算函数
    def calc_metrics(series, name):
        ann_ret = series.mean() * 252
        ann_vol = series.std() * np.sqrt(252)
        sharpe = (ann_ret - 0.02) / (ann_vol + 1e-6)
        
        downside = series[series<0]
        sortino = (ann_ret - 0.02) / (downside.std() * np.sqrt(252) + 1e-6)
        
        cum = (1+series).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        max_dd = dd.min()
        
        calmar = ann_ret / (abs(max_dd) + 1e-6)
        
        return {
            "Strategy": name,
            "Return": f"{ann_ret:.2%}",
            "Sharpe": f"{sharpe:.2f}",
            "Sortino": f"{sortino:.2f}",
            "Calmar": f"{calmar:.2f}",
            "MaxDD": f"{max_dd:.2%}"
        }
    
    m1 = calc_metrics(s_dmpo, "Diff-MPO (Factor Model)")
    m2 = calc_metrics(s_ew, "1/N Benchmark")
    
    res_df = pd.DataFrame([m1, m2])
    print("\n🏆 滚动回测最终结果:")
    print(res_df)
    
    # 画图
    plt.figure(figsize=(12, 6))
    plt.plot(wealth_dmpo, label='Diff-MPO (Factor Model)', linewidth=2)
    plt.plot(wealth_ew, label='1/N Benchmark', linestyle='--', alpha=0.7)
    plt.title(f'Walk-Forward: Factor Model vs 1/N ({TEST_START_YEAR}-{TEST_END_YEAR})', fontsize=14)
    plt.ylabel('Cumulative Wealth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = 'walk_forward_factor_model.png'
    plt.savefig(save_path, dpi=300)
    print(f"\n📈 图片已保存至: {save_path}")

if __name__ == "__main__":
    run_walk_forward_experiment()