"""
脚本名称: config.py
功能描述: 
    全局配置中心，存储项目运行所需的所有超参数、路径和常量。
    集中管理配置可以避免硬编码，方便实验管理和参数调整。

主要内容:
    1. 路径设置: 数据文件路径、训练集分割日期。
    2. 资产配置: 资产列表 (10个基础资产)。
    3. MPO 物理参数: 预测窗口、回看窗口、风险厌恶系数、交易成本。
    4. 模型架构参数: 神经网络层数、维度、学习率等。
    5. 因子模型参数: 隐因子数量、数值稳定性参数。
    6. CVaR 约束参数: 置信度、上限、惩罚系数。

输入:
    - 无 (这是一个静态配置脚本)。

输出:
    - cfg 对象: 被项目中的所有其他脚本 (data_loader, model, strategy, mpo_solver, train_diff_mpo, eval_rolling_all) 导入使用。

与其他脚本的关系:
    - 被所有功能脚本引用，作为参数的唯一来源。
"""

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
    LOOKBACK_WINDOW = 63
    INIT_WEALTH = 1.0
    RISK_AVERSION = 5   # 增大风险厌恶系数 (0.5 -> 5.0)，抑制过度冒险
    COST_COEFF = 0.002
    
    # ==========================
    # 3. 模型基础架构参数
    # ==========================
    DEVICE = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'
    INPUT_FEATURE_DIM = 15  # 10 Assets + 3 Macro + 2 Market
    
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    DROPOUT = 0.2
    
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    LR_FINE_TUNE_SCALE = 0.5 # 微调学习率缩放比例 (相对于 LEARNING_RATE)
    GRAD_CLIP = 0.5          # 梯度裁剪阈值
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
    LOSS_GAMMA_DD = 10.0  
    LOSS_GAMMA_TURNOVER = 3.0 

    # ==========================
    # 6. CVaR 约束参数 [NEW]
    # ==========================
    # CVAR_ENABLE = False
    CVAR_CONFIDENCE = 0.95
    CVAR_LIMIT = 0.05  # 尾部风险
    CVAR_PENALTY = 0 # 避免CVaR 惩罚过大（之前是50），在loss中淹没其他惩罚项

    def __repr__(self):
        return str(vars(self))

cfg = Config()