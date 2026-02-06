import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from mpo_solver import DifferentiableMPO

class MPO_Network(nn.Module):
    def __init__(self):
        super(MPO_Network, self).__init__()
        
        # 1. 特征提取器 (Feature Extractor)
        # 输入: (Batch, Lookback, Features)
        # 注意：这里的 input_size=13 必须和你 data_loader 里生成的特征列数一致
        # 如果你只用了5个因子+VIX+利率等，要确保列数对齐。
        # 这里建议写死或者从 data.shape 获取，暂时按 fetch_data 的默认列数 13
        self.lstm = nn.LSTM(
            input_size=cfg.INPUT_FEATURE_DIM, 
            hidden_size=cfg.HIDDEN_DIM,
            num_layers=cfg.NUM_LAYERS,
            batch_first=True,
            dropout=cfg.DROPOUT
        )
        
        # 2. 预测头 (Prediction Heads)
        
        # Head A: 预测收益率 mu (Batch, Horizon, Assets)
        self.mu_head = nn.Sequential(
            nn.Linear(cfg.HIDDEN_DIM, 32),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(32, cfg.PREDICT_HORIZON * cfg.NUM_ASSETS)
        )
        
        # Head B: 预测协方差因子 L (Batch, Horizon, Assets, Assets)
        # ⚠️ 修复点：直接在这里计算 N的平方
        num_l_params = cfg.PREDICT_HORIZON * (cfg.NUM_ASSETS * cfg.NUM_ASSETS)
        
        self.L_head = nn.Sequential(
            nn.Linear(cfg.HIDDEN_DIM, 64),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(64, num_l_params) 
        )
        
        # 3. 嵌入可微优化层
        self.mpo_layer = DifferentiableMPO()
        
    def forward(self, x, w_prev):
        """
        x: (Batch, T, F) 历史特征 [GPU]
        w_prev: (Batch, N) 上一期的持仓权重 [GPU]
        """
        batch_size = x.size(0)
        
        # --- A. 编码 (Encoding) ---
        # LSTM 依然在 GPU 上跑，享受加速
        _, (h_n, _) = self.lstm(x)
        context = h_n[-1] # (Batch, Hidden)
        
        # --- B. 预测参数 (Parameter Prediction) ---
        
        # 1. Mu (收益率)
        mu = self.mu_head(context)
        mu = mu.view(batch_size, cfg.PREDICT_HORIZON, cfg.NUM_ASSETS)
        
        # 2. L (协方差 Cholesky 因子)
        L_flat = self.L_head(context)
        L = L_flat.view(batch_size, cfg.PREDICT_HORIZON, cfg.NUM_ASSETS, cfg.NUM_ASSETS)
        
        # --- C. 数学变换 (保证合法性) ---
        mask = torch.tril(torch.ones_like(L))
        L = L * mask 
        
        diag_mask = torch.eye(cfg.NUM_ASSETS, device=x.device).view(1, 1, cfg.NUM_ASSETS, cfg.NUM_ASSETS)
        L = L + diag_mask * (F.softplus(L) + 1e-5 - L) 
        
        # --- D. 优化 (Optimization) ---
        # ⚠️ 关键优化：Solver 现在支持 GPU (FastDiffMPO)
        # 传递 CVaR 限制 (Config 中定义)
        w_plan = self.mpo_layer(mu, L, w_prev, cvar_limit=None)
        
        return w_plan, mu, L
    

class MPO_Network_Factor(nn.Module):
    """
    基于 LSTM 的因子模型 MPO 网络。
    使用结构化协方差矩阵 Sigma = B*B.T + D^2 代替直接预测。
    """
    def __init__(self):
        super(MPO_Network_Factor, self).__init__()
        
        # 1. 特征提取器 (LSTM Backbone)
        self.lstm = nn.LSTM(
            input_size=cfg.INPUT_FEATURE_DIM, 
            hidden_size=cfg.HIDDEN_DIM,
            num_layers=cfg.NUM_LAYERS,
            batch_first=True,
            dropout=cfg.DROPOUT
        )
        
        # 2. 预测头
        
        # Head A: 预期收益率 mu (Batch, H, N)
        self.mu_head = nn.Sequential(
            nn.Linear(cfg.HIDDEN_DIM, 32),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(32, cfg.PREDICT_HORIZON * cfg.NUM_ASSETS)
        )
        
        # Head B: 结构化协方差 (Structured Covariance)
        # 我们不再预测 N*N，而是预测因子载荷 B (N*K) 和特异性波动 D (N)
        
        # 预测因子载荷 B (Batch, H, N, K)
        # 代表每个资产对 K 个隐因子的敏感度
        self.B_head = nn.Sequential(
            nn.Linear(cfg.HIDDEN_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, cfg.PREDICT_HORIZON * cfg.NUM_ASSETS * cfg.NUM_FACTORS)
        )
        
        # 预测特异性波动 D (Batch, H, N)
        # 代表每个资产特有的、不能被因子解释的波动
        self.D_head = nn.Sequential(
            nn.Linear(cfg.HIDDEN_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, cfg.PREDICT_HORIZON * cfg.NUM_ASSETS)
        )
        
        # 3. 可微优化层 (cvxpylayers)
        self.mpo_layer = DifferentiableMPO()
        
    def forward(self, x, w_prev):
        """
        x: (Batch, Lookback, Features)
        w_prev: (Batch, Assets)
        """
        batch_size = x.size(0)
        
        # --- 1. Encoding ---
        # LSTM 输出: output, (h_n, c_n)
        # 我们取最后一个时间步的 hidden state 作为 context
        _, (h_n, _) = self.lstm(x)
        context = h_n[-1] # (Batch, Hidden)
        
        # --- 2. Parameter Prediction ---
        
        # A. 预测 Mu
        mu = self.mu_head(context)
        mu = mu.view(batch_size, cfg.PREDICT_HORIZON, cfg.NUM_ASSETS)
        
        # B. 预测 Sigma (通过因子模型)
        # B: (Batch, H, N, K)
        B_flat = self.B_head(context)
        B = B_flat.view(batch_size, cfg.PREDICT_HORIZON, cfg.NUM_ASSETS, cfg.NUM_FACTORS)
        
        # D: (Batch, H, N) -> 必须大于 0
        D_flat = self.D_head(context)
        # 使用 Softplus 保证正数，并加上最小阈值防止数值不稳定
        # 增大底噪到 1e-3，防止矩阵过于接近奇异
        D = F.softplus(D_flat) + 1e-3
        D = D.view(batch_size, cfg.PREDICT_HORIZON, cfg.NUM_ASSETS)
        
        # C. 构造协方差矩阵 Sigma = B @ B.T + diag(D^2)
        # factor_cov: (Batch, H, N, N)
        factor_cov = torch.matmul(B, B.transpose(-1, -2)) 
        
        # idiosyncratic_cov: (Batch, H, N, N)
        idiosyncratic_cov = torch.diag_embed(D**2)
        
        Sigma = factor_cov + idiosyncratic_cov
        
        # --- 3. Cholesky Decomposition ---
        # Solver 需要 L (where Sigma = L @ L.T)
        # 为了 Cholesky 的数值稳定性，加上微小的对角线扰动
        # 增大 Epsilon 到 1e-5
        epsilon_eye = 1e-5 * torch.eye(cfg.NUM_ASSETS, device=x.device)
        epsilon_eye = epsilon_eye.view(1, 1, cfg.NUM_ASSETS, cfg.NUM_ASSETS)
        
        Sigma_stabilized = Sigma + epsilon_eye
        
        try:
            L = torch.linalg.cholesky(Sigma_stabilized)
        except RuntimeError:
            # Fallback: 如果 Cholesky 失败（极罕见），回退到只使用对角阵 D
            # 这通常发生在梯度爆炸导致参数出现 NaN 时
            L = torch.diag_embed(D + 1e-3)
        
        # --- 4. Optimization Layer ---
        # FastDiffMPO 支持 GPU
        w_plan = self.mpo_layer(mu, L, w_prev, cvar_limit=None)
        
        # 返回 w_plan 以及中间预测值 (mu, L) 用于调试或监控
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
        w_plan = self.mpo_layer(mu, L, w_prev)
        
        return w_plan, mu, L

# 添加到 model.py

class E2E_Network(nn.Module):
    """
    End-to-End Policy Network
    直接从特征映射到投资组合权重 (Softmax)，跳过优化求解器。
    这是 Deep Reinforcement Learning (DDPG/PPO) 在量化中的简化版。
    """
    def __init__(self):
        super(E2E_Network, self).__init__()
        
        # 复用 Config 里的参数，保证与 Diff-MPO 仅仅是头部的区别
        self.lstm = nn.LSTM(
            input_size=cfg.INPUT_FEATURE_DIM, 
            hidden_size=cfg.HIDDEN_DIM,
            num_layers=cfg.NUM_LAYERS,
            batch_first=True,
            dropout=cfg.DROPOUT
        )
        
        # 直接输出权重
        self.policy_head = nn.Sequential(
            nn.Linear(cfg.HIDDEN_DIM, 32),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(32, cfg.NUM_ASSETS) 
        )
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, w_prev=None):
        # x: (Batch, Lookback, Features)
        
        _, (h_n, _) = self.lstm(x)
        context = h_n[-1]
        
        logits = self.policy_head(context)
        
        # 使用 Softmax 保证 sum(w) = 1 且 w >= 0
        w = self.softmax(logits)
        
        # 为了公平对比，我们也可以在这里强行截断单标的权重
        # 但标准的 Softmax 很难做到 w <= 0.3 的硬约束
        # 这里保留 Softmax 的原汁原味，体现 "Blackbox" 的特征
        
        # 输出格式对齐: (Batch, Horizon, Assets)
        # 这里 Horizon=1 (单步决策)
        w = w.unsqueeze(1) 
        
        # E2E 策略没有预测 mu 和 L，返回 None
        return w, None, None