import torch

class Config:
    # ==========================
    # 1. 路径与数据设置
    # ==========================
    DATA_PATH = 'mpo_experiment_data.csv' 
    TRAIN_SPLIT_DATE = '2015-01-01'
    
    # 资产列表 (10个基础资产)
    ASSETS = [
        'Val_Growth', 'Val_Value', 
        'Size_SmallCap', 'Size_LargeCap',
        'Mom_Loser', 'Mom_Winner',
        'Prof_LowProf', 'Prof_HighProf',
        'Inv_LowInv', 'Inv_HighInv'
    ]
    NUM_ASSETS = len(ASSETS)
    
    # ==========================
    # 2. MPO 物理参数
    # ==========================
    PREDICT_HORIZON = 5
    LOOKBACK_WINDOW = 60
    INIT_WEALTH = 1.0
    RISK_AVERSION = 0.5
    COST_COEFF = 0.002
    
    # ==========================
    # 3. 模型基础架构参数
    # ==========================
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    INPUT_FEATURE_DIM = 15  # 10 Assets + 3 Macro + 2 Market
    
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    DROPOUT = 0.2
    
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    SEED = 42

    # ==========================
    # 4. 因子模型参数 (Factor Model) [NEW]
    # ==========================
    # 隐因子数量 (Latent Factors): 建议 < NUM_ASSETS. 
    # 3 通常代表 Market, Size, Value 三大类风险
    NUM_FACTORS = 3 
    
    # 数值稳定性参数
    FACTOR_D_MIN = 1e-4       # 特异性波动 D 的最小值 (防止除零)
    CHOLESKY_EPSILON = 1e-6   # Cholesky 分解时的对角线扰动

    # ==========================
    # 5. Loss 惩罚系数
    # ==========================
    LOSS_GAMMA_DD = 5.0  
    LOSS_GAMMA_TURNOVER = 1.0 

    # ==========================
    # 6. CVaR 约束参数 [NEW]
    # ==========================
    CVAR_ENABLE = True
    CVAR_CONFIDENCE = 0.95
    CVAR_LIMIT = 0.08  # 放宽至 8% 以避免在极端行情下(如2020)物理无解导致的高换手

    def __repr__(self):
        return str(vars(self))

cfg = Config()