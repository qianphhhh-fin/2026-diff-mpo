import torch

class Config:
    # ==========================
    # 1. 路径与数据设置
    # ==========================
    DATA_PATH = 'mpo_experiment_data.csv'  # 上一步下载的数据
    TRAIN_SPLIT_DATE = '2020-01-01'        # 切分点：此前为训练集，此后为测试集(含COVID时期)
    
    # 资产列表 (必须与 CSV 列名对应)
    ASSETS = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'] 
    NUM_ASSETS = len(ASSETS)
    
    # ==========================
    # 2. 多期优化 (MPO) 物理参数
    # ==========================
    PREDICT_HORIZON = 5    # H: 预测未来多少天 (Solver规划的步长)
    LOOKBACK_WINDOW = 60   # T: LSTM 回看过去多少天作为输入
    
    # 交易物理约束 (非常关键)
    INIT_WEALTH = 1.0      # 初始资金 (归一化)
    RISK_AVERSION = 0.5    # Gamma: 风险厌恶系数 (越高越怕死，越低越贪婪)
    
    # 交易成本系数 (Lambda_cost)
    # 假设双边万分之五 (0.0005)，为了让模型敏感，训练时通常会放大一点
    COST_COEFF = 0.001     
    
    # ==========================
    # 3. 模型与训练参数
    # ==========================
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # LSTM 结构
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    DROPOUT = 0.2
    
    # 训练超参
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    
    # 随机种子 (复现用)
    SEED = 42

    def __repr__(self):
        return str(vars(self))

# 实例化一个全局配置对象
cfg = Config()