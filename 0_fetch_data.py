"""
脚本名称: 0_fetch_data.py
功能描述: 
    负责从本地目录和网络数据源获取原始金融数据，并进行预处理和合并，生成统一的实验数据集。
    主要任务包括：
    1. 读取清洗后的 Fama-French 因子数据 (本地 CSV)。
    2. 下载 FRED 宏观经济数据 (VIX, 10Y Yield, Credit Spread)。
    3. 下载 SPY 市场数据并计算技术指标 (滚动收益率、波动率)。
    4. 合并所有数据源，进行对齐和填充，生成最终的 CSV 文件。

输入:
    - 本地目录 'data/raw_data/' 下的 CSV 文件 (Portfolios_Formed_on_*.csv)。
    - 网络数据源 (FRED API, Yahoo Finance)。

输出:
    - 'mpo_experiment_data.csv': 合并后的完整数据集，供 config.py 和 data_loader.py 使用。
    - 'data/macro_features.csv', 'data/market_technicals.csv': 中间过程文件。

与其他脚本的关系:
    - 前置脚本: 无 (这是流水线的第一步)。
    - 后继脚本: 生成的数据被 config.py 引用，并由 data_loader.py 读取以构建 PyTorch Dataset。
"""

import pandas as pd
import yfinance as yf
import requests
import io
import numpy as np
import os
import time
from datetime import datetime

# =================配置区域=================
DATA_DIR = 'data'             
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw_data') # 请确保清洗好的csv放在这里
FINAL_FILE = 'mpo_experiment_data.csv'

START_DATE = '1990-01-01'
END_DATE = datetime.today().strftime('%Y-%m-%d')

# 核心资产池配置
# 格式: { '前缀': ('文件名', [原始列名], {列名映射}) }
FACTOR_CONFIG = {
    # 1. 价值因子 (Value vs Growth)
    'Val': ('Portfolios_Formed_on_BE-ME_Daily.csv', 
            ['Lo 30', 'Hi 30'], 
            {'Lo 30': 'Growth', 'Hi 30': 'Value'}),
            
    # 2. 规模因子 (Size)
    # 注意：根据您提供的信息，这个文件名里的 daily 是小写
    'Size': ('Portfolios_Formed_on_ME_daily.csv', 
             ['Lo 30', 'Hi 30'], 
             {'Lo 30': 'SmallCap', 'Hi 30': 'LargeCap'}),
             
    # 3. 动量因子 (Momentum)
    # 使用 12-2 动量 (标准学术定义)
    'Mom': ('10_Portfolios_Prior_12_2_Daily.csv', 
            ['Lo PRIOR', 'Hi PRIOR'], 
            {'Lo PRIOR': 'Loser', 'Hi PRIOR': 'Winner'}),
            
    # 4. 盈利因子 (Profitability)
    'Prof': ('Portfolios_Formed_on_OP_Daily.csv', 
             ['Lo 30', 'Hi 30'], 
             {'Lo 30': 'LowProf', 'Hi 30': 'HighProf'}),

    # 5. 投资因子 (Investment)
    'Inv': ('Portfolios_Formed_on_INV_Daily.csv', 
            ['Lo 30', 'Hi 30'], 
            {'Lo 30': 'LowInv', 'Hi 30': 'HighInv'}) 
}
# =========================================

os.makedirs(DATA_DIR, exist_ok=True)

def fetch_french_universe_clean():
    print(f"📂 [本地] 正在读取清洗后的 CSV 文件 ({RAW_DATA_DIR})...")
    
    if not os.path.exists(RAW_DATA_DIR):
        print(f"⛔ 错误：找不到目录 {RAW_DATA_DIR}")
        return None

    all_dfs = []
    
    for prefix, (filename, cols_to_keep, rename_map) in FACTOR_CONFIG.items():
        file_path = os.path.join(RAW_DATA_DIR, filename)
        print(f"   ...正在读取 {prefix} ({filename})")
        
        if not os.path.exists(file_path):
            print(f"      ❌ 文件不存在: {file_path}")
            continue
            
        try:
            # 1. 直接读取 CSV (因为您已经清洗过，表头在第一行)
            df = pd.read_csv(file_path)
            
            # 2. 清洗列名 (去除前后空格)
            df.columns = df.columns.str.strip()
            df.dropna(how='all',axis=0,inplace=True)  # 删除全空行
            # 3. 处理日期列
            # 假设第一列是 Date (19260701 这种格式)
            if 'Date' in df.columns:
                df['Date'] = df['Date'].astype(int).astype(str).str.strip()
                df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d', errors='coerce')
                df = df.set_index('Date')
            else:
                print(f"      ⚠️ 警告: {filename} 中没找到 'Date' 列，尝试使用第一列作为索引")
                df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.strip()
                df.index = pd.to_datetime(df.iloc[:, 0], format='%Y%m%d', errors='coerce')

            # 4. 筛选列
            missing_cols = [c for c in cols_to_keep if c not in df.columns]
            if missing_cols:
                print(f"      ❌ 缺少列: {missing_cols}。现有列: {list(df.columns)[:5]}...")
                continue
                
            df = df[cols_to_keep]
            
            # 5. 重命名
            new_names = {c: f"{prefix}_{rename_map.get(c, c)}" for c in cols_to_keep}
            df = df.rename(columns=new_names)
            
            # 6. 数值清洗
            # French 数据通常是百分比 (0.39 -> 0.39%)，需要除以 100
            # 缺失值标记通常是 -99.99 或 -999
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.replace([-99.99, -99.9, -999], np.nan)
            df = df / 100.0
            
            all_dfs.append(df)
            
        except Exception as e:
            print(f"      ❌ 读取失败: {e}")
            import traceback
            traceback.print_exc()

    if not all_dfs:
        print("⛔ 未能加载任何数据。")
        return None

    # 合并所有因子
    print("   正在合并资产...")
    try:
        universe = pd.concat(all_dfs, axis=1, join='outer')
    except Exception as e:
        print(f"⛔ 合并失败: {e}")
        return None

    # 截取时间
    universe = universe.loc[START_DATE:END_DATE].dropna()
    
    print(f"   ✅ 基础资产池构建完成。形状: {universe.shape}")
    return universe

def fetch_macro_fred():
    # ... (保持原有的宏观数据下载逻辑不变) ...
    file_path = os.path.join(DATA_DIR, 'macro_features.csv')
    if os.path.exists(file_path):
        print(f"✅ [本地] 已检测到宏观数据 (FRED)，跳过下载。")
        return pd.read_csv(file_path, index_col=0, parse_dates=True)

    print("⬇️ [下载] 正在下载 FRED 宏观数据...")
    fred_urls = {
        'VIX': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS',
        'US10Y': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10',
        'Credit_Spread': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2'
    }
    dfs = []
    try:
        headers = {'User-Agent': 'Mozilla/5.0'} 
        for name, url in fred_urls.items():
            print(f"   ...获取 {name}")
            r = requests.get(url, headers=headers, timeout=20)
            df = pd.read_csv(io.BytesIO(r.content), index_col=0, parse_dates=True)
            df = df.replace('.', np.nan).astype(float)
            df.columns = [name]
            dfs.append(df)
            time.sleep(1)
        macro_data = pd.concat(dfs, axis=1, sort=True).ffill().loc[START_DATE:END_DATE]
        macro_data.to_csv(file_path)
        return macro_data
    except Exception as e:
        print(f"   ❌ FRED 下载失败: {e}")
        return None

def fetch_yahoo_spy():
    # ... (保持原有的 SPY 下载逻辑不变) ...
    file_path = os.path.join(DATA_DIR, 'market_technicals.csv')
    if os.path.exists(file_path):
        print(f"✅ [本地] 已检测到 SPY 数据，跳过。")
        return pd.read_csv(file_path, index_col=0, parse_dates=True)

    print("⬇️ [下载] 正在下载 SPY (作为市场特征)...")
    try:
        spy = yf.download('SPY', start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
        if isinstance(spy.columns, pd.MultiIndex):
            if 'Close' in spy.columns.levels[0]:
                close = spy['Close']
                if spy.columns.nlevels > 1: close = close.iloc[:, 0]
            else: close = spy.iloc[:, 0]
        else: close = spy['Close']
            
        feats = pd.DataFrame(index=spy.index)
        feats['Mkt_Ret_60d'] = close.pct_change(60)
        feats['Mkt_Vol_20d'] = close.pct_change().rolling(20).std()
        feats.dropna(inplace=True)
        feats.to_csv(file_path)
        return feats
    except Exception as e:
        print(f"   ❌ SPY 下载失败: {e}")
        return None

def merge_and_save():
    print("\n🔗 开始合并数据...")
    df_assets = fetch_french_universe_clean()
    df_macro = fetch_macro_fred()
    df_spy = fetch_yahoo_spy()
    
    if df_assets is None: return

    # 合并
    full_df = df_assets.join(df_macro, how='left').join(df_spy, how='left')
    full_df.ffill(inplace=True)
    full_df.dropna(inplace=True)

    full_df.to_csv(FINAL_FILE)
    print(f"\n🎉 数据准备完成！")
    print(f"   文件路径: {FINAL_FILE}")
    print(f"   时间范围: {full_df.index.min().date()} 至 {full_df.index.max().date()}")
    print(f"   总行数: {len(full_df)}")
    
    cols = list(full_df.columns)
    asset_cols = [c for c in cols if '_' in c and any(k in c for k in FACTOR_CONFIG.keys())]
    print(f"   包含资产 ({len(asset_cols)}个): {asset_cols}")

if __name__ == "__main__":
    merge_and_save()