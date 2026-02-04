import pandas as pd
import yfinance as yf
import requests
import zipfile
import io
import numpy as np
import os
import time
from datetime import datetime

# =================配置区域=================
DATA_DIR = 'data'        # 中间数据保存目录
FINAL_FILE = 'mpo_experiment_data.csv'
START_DATE = '1990-01-01'
END_DATE = datetime.today().strftime('%Y-%m-%d')
# =========================================

# 确保数据目录存在
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_fama_french():
    file_path = os.path.join(DATA_DIR, 'fama_french.csv')
    if os.path.exists(file_path):
        print(f"✅ [本地] 已检测到 Fama-French 数据，跳过下载。")
        return pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    print("⬇️ [下载] 正在下载 Fama-French 5因子数据...")
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    try:
        response = requests.get(url, timeout=30)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                df = pd.read_csv(f, skiprows=3, index_col=0)
        
        # 清洗
        df.index.name = 'Date'
        df.index = pd.to_datetime(df.index, format='%Y%m%d', errors='coerce')
        df = df.dropna()
        df = df.loc[START_DATE:END_DATE]
        df = df / 100.0 # 单位转换
        df.columns = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
        
        # 保存中间文件
        df.to_csv(file_path)
        print(f"   💾 Fama-French 已保存至 {file_path}")
        return df
    except Exception as e:
        print(f"   ❌ Fama-French 下载失败: {e}")
        return None

def fetch_macro_fred():
    file_path = os.path.join(DATA_DIR, 'macro_features.csv')
    if os.path.exists(file_path):
        print(f"✅ [本地] 已检测到宏观数据 (FRED)，跳过下载。")
        return pd.read_csv(file_path, index_col=0, parse_dates=True)

    print("⬇️ [下载] 正在下载 FRED 宏观数据...")
    fred_urls = {
        'VIX': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS',
        'US10Y': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10',
        'US3M': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3',
        'Credit_Spread': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2'
    }
    
    dfs = []
    try:
        for name, url in fred_urls.items():
            print(f"   ...获取 {name}")
            # 添加 User-Agent 防止被简单的反爬拦截
            headers = {'User-Agent': 'Mozilla/5.0'} 
            r = requests.get(url, headers=headers, timeout=20)
            df = pd.read_csv(io.BytesIO(r.content), index_col=0, parse_dates=True)
            df = df.replace('.', np.nan).astype(float)
            df.columns = [name]
            dfs.append(df)
            time.sleep(1) # 礼貌性延迟，防止封IP
            
        # 修复 Pandas 警告: 显式指定 sort=True
        macro_data = pd.concat(dfs, axis=1, sort=True)
        macro_data = macro_data.loc[START_DATE:END_DATE].ffill()
        
        macro_data.to_csv(file_path)
        print(f"   💾 宏观数据已保存至 {file_path}")
        return macro_data
    except Exception as e:
        print(f"   ❌ FRED 下载失败: {e}")
        return None

def fetch_yahoo_spy():
    file_path = os.path.join(DATA_DIR, 'market_technicals.csv')
    if os.path.exists(file_path):
        print(f"✅ [本地] 已检测到 SPY 技术面数据，跳过下载。")
        return pd.read_csv(file_path, index_col=0, parse_dates=True)

    print("⬇️ [下载] 正在通过 Yahoo Finance 下载 SPY...")
    
    # 重试机制：最多尝试 3 次
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # yfinance 自动下载
            spy = yf.download('SPY', start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
            
            if spy.empty:
                raise ValueError("Yahoo 返回了空数据")

            # 清洗
            feats = pd.DataFrame(index=spy.index)
            # 处理多级索引问题 (yfinance 新版特性)
            if isinstance(spy.columns, pd.MultiIndex):
                close = spy['Close']['SPY'] if 'SPY' in spy.columns.levels[1] else spy.iloc[:, 0]
                vol = spy['Volume']['SPY'] if 'SPY' in spy.columns.levels[1] else spy.iloc[:, 1]
            else:
                close = spy['Close']
                vol = spy['Volume']

            feats['Mkt_Ret_20d'] = close.pct_change(20)
            feats['Mkt_Vol_20d'] = close.pct_change().rolling(20).std()
            feats['Mkt_Volume_Log'] = np.log(vol + 1)
            
            feats.to_csv(file_path)
            print(f"   💾 SPY 数据已保存至 {file_path}")
            return feats
            
        except Exception as e:
            print(f"   ⚠️ 尝试 {attempt+1}/{max_retries} 失败: {e}")
            if "Rate limited" in str(e) or "Too Many Requests" in str(e):
                wait_time = 10 * (attempt + 1)
                print(f"      ⏳ 触发限流，等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                time.sleep(2)
    
    print("   ❌ Yahoo 数据下载最终失败。请稍后再试或检查网络。")
    return None

def merge_and_save():
    print("\n🔗 开始合并数据...")
    
    df_ff = fetch_fama_french()
    df_macro = fetch_macro_fred()
    df_spy = fetch_yahoo_spy()
    
    # 检查完整性
    if df_ff is None or df_macro is None or df_spy is None:
        print("\n⛔ 错误：部分数据缺失，无法合并。")
        print("   请查看上方报错信息，重新运行脚本以补全缺失部分。")
        print("   (已下载的部分保存在 /data 文件夹中，无需重新下载)")
        return

    # 合并
    print("   正在对齐时间戳...")
    full_df = df_ff.join(df_macro, how='left').join(df_spy, how='left')
    
    # 去除空值 (保留三者都有数据的日期)
    original_len = len(full_df)
    full_df.dropna(inplace=True)
    final_len = len(full_df)
    
    if final_len == 0:
        print("   ⚠️ 警告：合并后数据为空！请检查各个源的时间范围是否有重叠。")
        return

    full_df.to_csv(FINAL_FILE)
    print(f"\n🎉 大功告成！")
    print(f"   原始行数: {original_len}")
    print(f"   清洗后行数: {final_len}")
    print(f"   最终文件: {FINAL_FILE}")
    print(f"   特征列表: {list(full_df.columns)}")

if __name__ == "__main__":
    merge_and_save()