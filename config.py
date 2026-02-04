import torch

class Config:
    # ==========================
    # 1. 路径与数据设置
    # ==========================
    DATA_PATH = 'mpo_experiment_data.csv'  
    TRAIN_SPLIT_DATE = '2015-01-01'        # 建议往后推一点，因为数据从1926年开始，训练集可以很长
    
    # ⚠️ 核心修改：资产列表 (必须与 0_fetch_data.py 生成的 CSV 列名严格对应)
    ASSETS = [
        'Val_Growth', 'Val_Value', 
        'Size_SmallCap', 'Size_LargeCap',
        'Mom_Loser', 'Mom_Winner',
        'Prof_LowProf', 'Prof_HighProf',
        'Inv_LowInv', 'Inv_HighInv'
    ]
    
    NUM_ASSETS = len(ASSETS)  # 自动变为 10
    
    # ==========================
    # 2. 多期优化 (MPO) 物理参数
    # ==========================
    PREDICT_HORIZON = 5    # 预测未来 5 天
    LOOKBACK_WINDOW = 60   # 回看过去 60 天 (LSTM输入)
    
    # 交易物理约束
    INIT_WEALTH = 1.0      
    RISK_AVERSION = 0.5    # 风险厌恶系数
    
    # 交易成本系数 (Lambda_cost)
    # 因为现在是 Long-Only 的基础资产，换手可能会比纯因子低，但手续费依然不能忽略
    COST_COEFF = 0.001     
    
    # ==========================
    # 3. 模型与训练参数
    # ==========================
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ⚠️ 注意：输入特征维度
    # 现在的 CSV 包含：10个资产 + 3个宏观(VIX,US10Y,Spread) + 2个市场(SPY_Ret, SPY_Vol) = 15列
    # 你的 model.py 里 LSTM 的 input_size 必须匹配这个数字
    INPUT_FEATURE_DIM = 15 
    
    # LSTM 结构
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    DROPOUT = 0.2
    
    # 训练超参
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    
    SEED = 42

    def __repr__(self):
        return str(vars(self))

cfg = Config()