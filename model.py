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
        # ⚠️ 关键修复：Solver 必须在 CPU 上运行
        # 1. 把所有输入移到 CPU
        mu_cpu = mu.cpu()
        L_cpu = L.cpu()
        w_prev_cpu = w_prev.cpu()
        
        # 2. 调用 Solver (CPU)
        # cvxpylayers 在 CPU 上工作得最稳定
        w_plan_cpu = self.mpo_layer(mu_cpu, L_cpu, w_prev_cpu)
        
        # 3. 结果移回 GPU (以便后续计算 Loss 和反向传播)
        w_plan = w_plan_cpu.to(x.device)
        
        return w_plan, mu, L