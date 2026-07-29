import pandas as pd
import numpy as np
import joblib
import os
from tqdm import tqdm
from scipy.stats import rankdata, spearmanr

RELATIVE_MARKET_FEATURE_SET = '158+39_reduced25_relmarket12'
RELATIVE_MARKET_FEATURES = (
    'cs_return_5_pct',
    'cs_return_20_pct',
    'cs_return_60_pct',
    'cs_volatility_20_pct',
    'cs_volume_ratio_20_pct',
    'cs_ma20_distance_pct',
    'cs_ma60_distance_pct',
    'market_return_5',
    'market_return_20',
    'market_breadth_up',
    'market_breadth_above_ma20',
    'market_return_20_dispersion',
)
RISK_MARKET_FEATURE_SET = '158+39_reduced25_relmarket12_risk15'
STOCK_RISK_FEATURES = (
    'cs_return_1_pct',
    'cs_return_3_pct',
    'cs_momentum_gap_5_20_pct',
    'cs_momentum_gap_5_60_pct',
    'cs_downside_vol_5_pct',
    'cs_downside_vol_20_pct',
    'cs_drawdown_20_pct',
)
MARKET_PRESSURE_FEATURES = (
    'market_return_1',
    'market_return_3',
    'market_downside_vol_5',
    'market_downside_vol_20',
    'market_drawdown_20',
    'market_breadth_change_5',
    'market_ma20_breadth_change_5',
    'market_crowding_20',
)
RISK_MARKET_FEATURES = (*STOCK_RISK_FEATURES, *MARKET_PRESSURE_FEATURES)
SELECTION_MOMENTUM_FEATURES = (
    'cs_return_5_pct',
    'cs_return_20_pct',
    'cs_return_60_pct',
)
SELECTION_RETURN_FEATURE = 'return_1'


def add_relative_market_features(df):
    """增加严格因果的同日横截面相对特征与市场状态特征。"""
    required = {'股票代码', '日期', '收盘', '成交量'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'相对市场特征缺少基础列: {sorted(missing)}')

    panel = df.copy()
    panel['日期'] = pd.to_datetime(panel['日期'])
    panel['收盘'] = pd.to_numeric(panel['收盘'], errors='coerce')
    panel['成交量'] = pd.to_numeric(panel['成交量'], errors='coerce')
    panel = panel.sort_values(['股票代码', '日期'])

    stock_codes = panel['股票代码']
    dates = panel['日期']
    close_by_stock = panel.groupby(stock_codes, sort=False)['收盘']
    volume_by_stock = panel.groupby(stock_codes, sort=False)['成交量']

    return_1 = close_by_stock.pct_change(periods=1, fill_method=None)
    return_3 = close_by_stock.pct_change(periods=3, fill_method=None)
    return_5 = close_by_stock.pct_change(periods=5, fill_method=None)
    return_20 = close_by_stock.pct_change(periods=20, fill_method=None)
    return_60 = close_by_stock.pct_change(periods=60, fill_method=None)
    volatility_20 = return_1.groupby(
        stock_codes,
        sort=False,
    ).transform(lambda values: values.rolling(20, min_periods=20).std())
    downside_squared = return_1.clip(upper=0.0).pow(2)
    downside_volatility_5 = downside_squared.groupby(
        stock_codes,
        sort=False,
    ).transform(
        lambda values: values.rolling(5, min_periods=5).mean().pow(0.5)
    )
    downside_volatility_20 = downside_squared.groupby(
        stock_codes,
        sort=False,
    ).transform(
        lambda values: values.rolling(20, min_periods=20).mean().pow(0.5)
    )
    volume_ma20 = volume_by_stock.transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    close_ma20 = close_by_stock.transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    close_ma60 = close_by_stock.transform(
        lambda values: values.rolling(60, min_periods=60).mean()
    )
    close_high20 = close_by_stock.transform(
        lambda values: values.rolling(20, min_periods=20).max()
    )
    volume_ratio_20 = panel['成交量'] / volume_ma20.replace(0.0, np.nan) - 1.0
    ma20_distance = panel['收盘'] / close_ma20.replace(0.0, np.nan) - 1.0
    ma60_distance = panel['收盘'] / close_ma60.replace(0.0, np.nan) - 1.0
    drawdown_20 = panel['收盘'] / close_high20.replace(0.0, np.nan) - 1.0
    momentum_gap_5_20 = return_5 - return_20
    momentum_gap_5_60 = return_5 - return_60

    relative_sources = {
        'cs_return_5_pct': return_5,
        'cs_return_20_pct': return_20,
        'cs_return_60_pct': return_60,
        'cs_volatility_20_pct': volatility_20,
        'cs_volume_ratio_20_pct': volume_ratio_20,
        'cs_ma20_distance_pct': ma20_distance,
        'cs_ma60_distance_pct': ma60_distance,
    }
    for name, values in relative_sources.items():
        # 缺少足够历史的股票不参与当日排名，并用中性百分位填充。
        panel[name] = values.groupby(dates).rank(
            method='average',
            pct=True,
        ).fillna(0.5)

    risk_relative_sources = {
        'cs_return_1_pct': return_1,
        'cs_return_3_pct': return_3,
        'cs_momentum_gap_5_20_pct': momentum_gap_5_20,
        'cs_momentum_gap_5_60_pct': momentum_gap_5_60,
        'cs_downside_vol_5_pct': downside_volatility_5,
        'cs_downside_vol_20_pct': downside_volatility_20,
        'cs_drawdown_20_pct': drawdown_20,
    }
    for name, values in risk_relative_sources.items():
        panel[name] = values.groupby(dates).rank(
            method='average',
            pct=True,
        ).fillna(0.5)

    panel['market_return_1'] = return_1.groupby(dates).transform('mean')
    panel['market_return_3'] = return_3.groupby(dates).transform('mean')
    panel['market_return_5'] = return_5.groupby(dates).transform('mean')
    panel['market_return_20'] = return_20.groupby(dates).transform('mean')

    up_indicator = return_1.gt(0.0).astype(float).where(return_1.notna())
    above_ma20_indicator = (
        ma20_distance.gt(0.0).astype(float).where(ma20_distance.notna())
    )
    panel['market_breadth_up'] = up_indicator.groupby(dates).transform('mean')
    panel['market_breadth_above_ma20'] = (
        above_ma20_indicator.groupby(dates).transform('mean')
    )

    market_return_20_mean = return_20.groupby(dates).transform('mean')
    panel['market_return_20_dispersion'] = (
        (return_20 - market_return_20_mean)
        .pow(2)
        .groupby(dates)
        .transform('mean')
        .pow(0.5)
    )
    panel['market_downside_vol_5'] = (
        downside_volatility_5.groupby(dates).transform('mean')
    )
    panel['market_downside_vol_20'] = (
        downside_volatility_20.groupby(dates).transform('mean')
    )
    panel['market_drawdown_20'] = drawdown_20.groupby(dates).transform('mean')
    panel['market_breadth_change_5'] = (
        panel['market_breadth_up']
        - panel.groupby(stock_codes, sort=False)['market_breadth_up'].shift(5)
    )
    panel['market_ma20_breadth_change_5'] = (
        panel['market_breadth_above_ma20']
        - panel.groupby(
            stock_codes,
            sort=False,
        )['market_breadth_above_ma20'].shift(5)
    )

    # 用个股收益与等权市场收益的20日相关性均值近似市场拥挤度。
    # 相比全股票两两相关矩阵，该定义为 O(NT)，且只依赖当日及过去数据。
    market_return_1_by_row = panel['market_return_1']
    rolling_return_mean = return_1.groupby(
        stock_codes,
        sort=False,
    ).transform(lambda values: values.rolling(20, min_periods=20).mean())
    rolling_market_mean = market_return_1_by_row.groupby(
        stock_codes,
        sort=False,
    ).transform(lambda values: values.rolling(20, min_periods=20).mean())
    rolling_product_mean = (return_1 * market_return_1_by_row).groupby(
        stock_codes,
        sort=False,
    ).transform(lambda values: values.rolling(20, min_periods=20).mean())
    rolling_return_second_moment = return_1.pow(2).groupby(
        stock_codes,
        sort=False,
    ).transform(lambda values: values.rolling(20, min_periods=20).mean())
    rolling_market_second_moment = market_return_1_by_row.pow(2).groupby(
        stock_codes,
        sort=False,
    ).transform(lambda values: values.rolling(20, min_periods=20).mean())
    rolling_covariance = (
        rolling_product_mean - rolling_return_mean * rolling_market_mean
    )
    rolling_return_variance = (
        rolling_return_second_moment - rolling_return_mean.pow(2)
    ).clip(lower=0.0)
    rolling_market_variance = (
        rolling_market_second_moment - rolling_market_mean.pow(2)
    ).clip(lower=0.0)
    rolling_correlation = rolling_covariance / (
        rolling_return_variance.pow(0.5)
        * rolling_market_variance.pow(0.5)
    ).replace(0.0, np.nan)
    panel['market_crowding_20'] = (
        rolling_correlation.clip(lower=0.0, upper=1.0)
        .groupby(dates)
        .transform('mean')
    )

    generated_features = [*RELATIVE_MARKET_FEATURES, *RISK_MARKET_FEATURES]
    panel[generated_features] = (
        panel[generated_features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return panel.sort_index()


# 特征工程
def _rolling_linear_regression(x, y):
    x = np.vstack([np.ones(len(x)), x]).T
    beta, res, _, _ = np.linalg.lstsq(x, y, rcond=None)
    return beta[1], res[0] if len(res) > 0 else 0.0, np.sum((y - (x @ beta))**2)
def engineer_features_158plus39(df):
    """
    计算技术指标和 Alpha 特征并合并；函数名为兼容旧配置保留。
    """
    # 为了避免修改原始DataFrame，创建一个副本
    df_copy = df.copy()

    # 1. 计算158个Alpha特征
    df_158 = engineer_features(df_copy)
    
    # 2. 计算39个技术指标特征
    df_39 = engineer_features_39(df_copy)

    # 3. 合并两个DataFrame
    # 首先，从df_39中选取我们需要的列，避免与df_158中的原始列（如'开盘'）重复
    feature_cols_39 = [
        'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 
        'volume_change', 'obv', 'volume_ma_5', 'volume_ma_20', 'volume_ratio', 
        'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60', 
        'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  
        'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
    ]
    
    # 确保所有列都存在于df_39中
    feature_cols_39_exist = [col for col in feature_cols_39 if col in df_39.columns]
    
    # 合并，df_158 已经包含了原始列和158个特征
    df_final = pd.concat([df_158, df_39[feature_cols_39_exist]], axis=1)

    # 4. 处理可能因为合并产生的重复列（如果两个函数生成了同名特征）
    df_final = df_final.loc[:,~df_final.columns.duplicated()]

    # 5. 统一处理inf和NaN
    df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_final.fillna(0, inplace=True)
    
    return df_final

def engineer_features_39(df):
    """
    计算39个技术指标特征。
    'stock_idx','开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
    'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
    'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 
    'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  
    'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
    """
    try:
        import talib
        import numpy as np
    except ImportError:
        print("请安装TA-Lib库: pip install TA-Lib")
        raise

    df = df.copy()

    # 基础变量
    open_ = df['开盘'].astype(float)
    high = df['最高'].astype(float)
    low = df['最低'].astype(float)
    close = df['收盘'].astype(float)
    volume = df['成交量'].astype(float)

    # 移动平均线 (SMA, EMA)
    df['sma_5'] = talib.SMA(close, timeperiod=5)
    df['sma_20'] = talib.SMA(close, timeperiod=20)
    df['ema_12'] = talib.EMA(close, timeperiod=12)
    df['ema_26'] = talib.EMA(close, timeperiod=26)
    df['ema_60'] = talib.EMA(close, timeperiod=60)

    # MACD
    macd_line, macd_signal_line, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd_line
    df['macd_signal'] = macd_signal_line

    # RSI
    df['rsi'] = talib.RSI(close, timeperiod=14)

    # KDJ
    df['kdj_k'], df['kdj_d'] = talib.STOCH(high, low, close, fastk_period=9, slowk_period=3, slowd_period=3)
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']

    # Bollinger Bands
    df['boll_mid'], df['boll_upper'], df['boll_lower'] = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    # 标准差 = (上轨 - 中轨) / 2
    df['boll_std'] = (df['boll_upper'] - df['boll_mid']) / 2

    # 删除临时列
    df.drop(columns=['boll_upper', 'boll_lower'], inplace=True)

    # ATR
    df['atr_14'] = talib.ATR(high, low, close, timeperiod=14)

    # OBV (On-Balance Volume)
    df['obv'] = talib.OBV(close, volume)

    # Volume-related features
    df['volume_change'] = volume.pct_change()
    df['volume_ma_5'] = talib.SMA(volume, timeperiod=5)
    df['volume_ma_20'] = talib.SMA(volume, timeperiod=20)
    df['volume_ratio'] = df['volume_ma_5'] / df['volume_ma_20']

    # Returns and Volatility
    df['return_1'] = close.pct_change(1)
    df['return_5'] = close.pct_change(5)
    df['return_10'] = close.pct_change(10)
    df['volatility_10'] = df['return_1'].rolling(10).std()
    df['volatility_20'] = df['return_1'].rolling(20).std()

    # Spreads
    df['high_low_spread'] = high - low
    df['open_close_spread'] = open_ - close
    df['high_close_spread'] = high - close
    df['low_close_spread'] = low - close

    # 处理 inf 和 -inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 填充 NaN 值（注意：这可能引入偏差，根据下游任务决定是否保留）
    df.fillna(0, inplace=True)

    return df

def engineer_features(df):
    """
    使用talib加速特征计算
    """
    try:
        import talib
    except ImportError:
        print("请安装TA-Lib库: pip install TA-Lib")
        raise

    # 为了避免修改原始DataFrame，创建一个副本
    df = df.copy()

    # 基础变量
    open_ = df['开盘'].astype(float)
    high = df['最高'].astype(float)
    low = df['最低'].astype(float)
    close = df['收盘'].astype(float)
    volume = df['成交量'].astype(float)
    vwap = df['成交额'] / (volume + 1e-12)

    # 特征列表
    features = []
    feature_names = []

    # 1. K-line features (9 features) - 向量化操作，速度很快，无需更改
    features.extend([
        (close - open_) / (open_ + 1e-12),
        (high - low) / (open_ + 1e-12),
        (close - open_) / (high - low + 1e-12),
        (high - pd.concat([open_, close], axis=1).max(axis=1)) / (open_ + 1e-12),
        (high - pd.concat([open_, close], axis=1).max(axis=1)) / (high - low + 1e-12),
        (pd.concat([open_, close], axis=1).min(axis=1) - low) / (open_ + 1e-12),
        (pd.concat([open_, close], axis=1).min(axis=1) - low) / (high - low + 1e-12),
        (2 * close - high - low) / (open_ + 1e-12),
        (2 * close - high - low) / (high - low + 1e-12)
    ])
    feature_names.extend(['KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2'])

    # 2. Price-related features (4 features) - 向量化操作，无需更改
    features.extend([
        open_ / (close + 1e-12),
        high / (close + 1e-12),
        low / (close + 1e-12),
        vwap / (close + 1e-12)
    ])
    feature_names.extend(['OPEN0', 'HIGH0', 'LOW0', 'VWAP0'])

    windows = [5, 10, 20, 30, 60]

    # 3. Price change features (5 features) - 向量化操作，无需更改
    for w in windows:
        features.append(close.shift(w) / (close + 1e-12))
        feature_names.append(f'ROC{w}')

    # 4. Moving average features (5 features) - 使用 talib 加速
    for w in windows:
        features.append(talib.SMA(close, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'MA{w}')

    # 5. Standard deviation features (5 features) - 使用 talib 加速
    for w in windows:
        features.append(talib.STDDEV(close, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'STD{w}')

    # 6. Regression-based features (10 features) - 使用 talib 加速
    for w in windows:
        slope = talib.LINEARREG_SLOPE(close, timeperiod=w)
        features.append(slope / (close + 1e-12))
        feature_names.append(f'BETA{w}')

        # Residuals
        intercept = talib.LINEARREG_INTERCEPT(close, timeperiod=w)
        predicted = slope * (w - 1) + intercept
        resi = close - predicted
        features.append(resi / (close + 1e-12))
        feature_names.append(f'RESI{w}')

    # 7. Max/Min features (10 features) - 使用 talib 加速
    for w in windows:
        features.append(talib.MAX(high, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'MAX{w}')
    for w in windows:
        features.append(talib.MIN(low, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'MIN{w}')

    # 8. Quantile features (10 features) - talib 不支持，保留原实现
    for w in windows:
        features.append(close.rolling(w).quantile(0.8) / (close + 1e-12))
        feature_names.append(f'QTLU{w}')
    for w in windows:
        features.append(close.rolling(w).quantile(0.2) / (close + 1e-12))
        feature_names.append(f'QTLD{w}')

    # 9. Rank features (5 features) - talib 不支持，保留原实现
    for w in windows:
        features.append(close.rolling(w).rank(pct=True))
        feature_names.append(f'RANK{w}')

    # 10. Stochastic oscillator features (5 features) - talib.STOCH 计算的是另一指标，保留原实现
    for w in windows:
        min_low = low.rolling(w).min()
        max_high = high.rolling(w).max()
        features.append((close - min_low) / (max_high - min_low + 1e-12))
        feature_names.append(f'RSV{w}')

    # 11. Index of Max/Min features (10 features) - IMXD 可由 IMAX-IMIN 精确恢复，删除冗余列
    for w in windows:
        features.append(high.rolling(w).apply(np.argmax, raw=True) / w)
        feature_names.append(f'IMAX{w}')
    for w in windows:
        features.append(low.rolling(w).apply(np.argmin, raw=True) / w)
        feature_names.append(f'IMIN{w}')

    # 12. Correlation features (10 features) - 使用 talib 加速
    log_volume = np.log(volume + 1)
    for w in windows:
        features.append(talib.CORREL(close, log_volume, timeperiod=w))
        feature_names.append(f'CORR{w}')
    
    close_ret = close / close.shift(1)
    volume_ret = volume / (volume.shift(1) + 1e-12)
    log_volume_ret = np.log(volume_ret + 1)
    for w in windows:
        # talib.CORREL 需要 Series，且不能有 NaN
        corr_df = pd.concat([close_ret, log_volume_ret], axis=1).fillna(0)
        features.append(talib.CORREL(corr_df.iloc[:, 0], corr_df.iloc[:, 1], timeperiod=w))
        feature_names.append(f'CORD{w}')

    # 13. Count features (10 features) - CNTD=CNTP-CNTN，删除冗余列
    close_diff_pos = (close > close.shift(1))
    close_diff_neg = (close < close.shift(1))
    for w in windows:
        features.append(close_diff_pos.rolling(w).mean())
        feature_names.append(f'CNTP{w}')
    for w in windows:
        features.append(close_diff_neg.rolling(w).mean())
        feature_names.append(f'CNTN{w}')

    # 14. Sum of price change features (10 features) - SUMD=SUMP-SUMN，删除冗余列
    close_diff_abs = (close - close.shift(1)).abs()
    close_diff_up = (close - close.shift(1)).clip(lower=0)
    close_diff_down = -(close - close.shift(1)).clip(upper=0)
    for w in windows:
        sum_abs = close_diff_abs.rolling(w).sum()
        sum_up = close_diff_up.rolling(w).sum()
        features.append(sum_up / (sum_abs + 1e-12))
        feature_names.append(f'SUMP{w}')
    for w in windows:
        sum_abs = close_diff_abs.rolling(w).sum()
        sum_down = close_diff_down.rolling(w).sum()
        features.append(sum_down / (sum_abs + 1e-12))
        feature_names.append(f'SUMN{w}')

    # 15. Volume-related features (10 features) - 使用 talib 加速
    for w in windows:
        features.append(talib.SMA(volume, timeperiod=w) / (volume + 1e-12))
        feature_names.append(f'VMA{w}')
    for w in windows:
        features.append(talib.STDDEV(volume, timeperiod=w) / (volume + 1e-12))
        feature_names.append(f'VSTD{w}')

    # 16. Weighted volume features (5 features) - 向量化操作，无需更改
    vol_weighted_ret = (close / close.shift(1) - 1).abs() * volume
    for w in windows:
        mean_vol_w_ret = vol_weighted_ret.rolling(w).mean()
        std_vol_w_ret = vol_weighted_ret.rolling(w).std()
        features.append(std_vol_w_ret / (mean_vol_w_ret + 1e-12))
        feature_names.append(f'WVMA{w}')

    # 17. Volume change sum features (10 features) - VSUMD=VSUMP-VSUMN，删除冗余列
    volume_diff_abs = (volume - volume.shift(1)).abs()
    volume_diff_up = (volume - volume.shift(1)).clip(lower=0)
    volume_diff_down = -(volume - volume.shift(1)).clip(upper=0)
    for w in windows:
        sum_abs = volume_diff_abs.rolling(w).sum()
        sum_up = volume_diff_up.rolling(w).sum()
        features.append(sum_up / (sum_abs + 1e-12))
        feature_names.append(f'VSUMP{w}')
    for w in windows:
        sum_abs = volume_diff_abs.rolling(w).sum()
        sum_down = volume_diff_down.rolling(w).sum()
        features.append(sum_down / (sum_abs + 1e-12))
        feature_names.append(f'VSUMN{w}')

    # Combine all features into a new DataFrame
    feature_df = pd.concat(features, axis=1)
    feature_df.columns = feature_names
    
    # Merge with original df
    df = pd.concat([df, feature_df], axis=1)
    
    # 填充缺失值
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df
def process_single_stock(stock_row, data, features, sequence_length, date):
    """处理单只股票的数据，返回序列、目标值和股票索引"""
    stock_code = stock_row['instrument']
    # stock_idx = stock_row['stock_idx']
    
    # 获取该股票历史sequence_length天的数据（包括当天）
    stock_history = data[
        (data['instrument'] == stock_code) & 
        (data['datetime'] <= date)
    ].sort_values('datetime').tail(sequence_length)

    if len(stock_history) == sequence_length:
        seq = stock_history[features].values
        target = stock_row['label']  # 下一天的涨跌幅
        return seq, target, stock_code
    else:
        return None, None, None

def process_single_date(date, data, features, sequence_length):
    """处理单个日期的所有股票数据"""
    try:
        # 获取当天有target的股票（即有下一天数据的股票）
        day_data = data[data['datetime'] == date]
        day_data = day_data.dropna(subset=['label'])  # 确保有target
        
        if len(day_data) < 10:  # 确保至少有10只股票
            return None
            
        # 获取当天所有股票的特征序列
        day_sequences = []
        day_targets = []
        day_stock_indices = []
        
        # 对于单个日期内的股票处理，仍使用串行方式避免过度并行化
        # 因为多进程的开销可能超过收益
        for _, stock_row in day_data.iterrows():
            seq, target, stock_idx = process_single_stock(
                stock_row, data, features, sequence_length, date
            )
            if seq is not None:
                day_sequences.append(seq)
                day_targets.append(target)
                day_stock_indices.append(stock_idx)
        
        if len(day_sequences) >= 10:  # 确保有足够的股票
            # 创建排序标签：涨跌幅越高，相关性得分越高
            day_targets = np.array(day_targets)
            # 使用涨跌幅的排序作为相关性得分（值越大排名越高）
            sorted_indices = np.argsort(day_targets)[::-1]  # 降序排列
            relevance = np.zeros_like(day_targets, dtype=np.float32)
            for rank, idx in enumerate(sorted_indices):
                relevance[idx] = len(day_targets) - rank  # 最高涨跌幅得分最高
            
            return {
                'sequences': np.array(day_sequences),
                'targets': day_targets,
                'relevance': relevance,
                'stock_indices': day_stock_indices,
                'date': date
            }
        else:
            return None
            
    except Exception as e:
        print(f"处理日期 {date} 时出错: {e}")
        return None

def create_ranking_dataset_multiprocess(data, features, sequence_length, ranking_data_path=None, max_workers=None):
    """
    输入：股票历史数据 DataFrame，特征列名列表，序列长度，排名数据保存路径，最大工作进程数
    输出：排序数据集，格式为：(sequences, targets, relevance_scores, stock_indices)
    - sequences: List of np.array, 每个元素形状为 (num_stocks, sequence_length, num_features)
    - targets: List of np.array, 每个元素形状为 (num_stocks,)
    - relevance_scores: List of np.array, 每个元素形状为 (num_stocks,)
    - stock_indices: List of List, 每个元素为对应股票的索引列表
    """
    """多进程版本的排序数据集创建函数"""
    if ranking_data_path is not None:
        # 如果指定了ranking_data_path，尝试加载已有的数据集
        if os.path.exists(ranking_data_path):
            print(f"加载已有的排序数据集: {ranking_data_path}")
            return joblib.load(ranking_data_path)
    """
    创建排序数据集，按日期组织数据，每个样本包含同一天所有股票的特征和涨跌幅排序
    使用多线程加速处理
    """
    sequences = []
    targets = []
    relevance_scores = []
    stock_indices = []
    
    print("正在创建排序数据集（多线程版本）...")
    
    # 获取所有日期，确保有足够的历史数据
    all_dates = sorted(data['datetime'].unique())
    min_date_for_sequences = all_dates[sequence_length-1]  # 确保有足够历史数据
    
    # 只使用有足够历史数据的日期
    valid_dates = [date for date in all_dates if date >= min_date_for_sequences]
    
    print(f"总日期数: {len(all_dates)}, 有效日期数: {len(valid_dates)}")
    
    # 设置最大工作进程数
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    from tqdm import tqdm
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 10)
    
    print(f"使用 {max_workers} 个进程处理数据")
    
    # 分批处理日期以避免内存问题
    processed_count = 0
        
    # 使用进程池并行处理日期批次
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 创建处理函数的偏函数
            process_func = partial(process_single_date,
                                    data=data,
                                    features=features,
                                    sequence_length=sequence_length)
            
            # 并行处理批次中的所有日期
            futures = [executor.submit(process_func, date) for date in valid_dates]
            
            # 收集结果
            for future in tqdm(futures, desc="Processing dates", total=len(valid_dates)):
                try:
                    result = future.result(timeout=60)  # 设置超时
                    if result is not None:
                        sequences.append(result['sequences'])
                        targets.append(result['targets'])
                        relevance_scores.append(result['relevance'])
                        stock_indices.append(result['stock_indices'])
                        processed_count += 1
                except Exception as e:
                    print(f"处理某个日期时出错: {e}")
                    continue
                    
    except Exception as e:
        print(f"进程池处理出错，回退到串行处理: {e}")
        # 如果多进程出错，回退到串行处理
        for date in tqdm(valid_dates, desc="串行处理"):
            result = process_single_date(date, data, features, sequence_length)
            if result is not None:
                sequences.append(result['sequences'])
                targets.append(result['targets'])
                relevance_scores.append(result['relevance'])
                stock_indices.append(result['stock_indices'])
                processed_count += 1
    
    print(f"成功创建 {len(sequences)} 个训练样本")
    if len(sequences) > 0:
        print(f"每个样本平均包含 {np.mean([len(seq) for seq in sequences]):.1f} 只股票")
    
    # 将四个数据保存下来，下次直接读取
    if ranking_data_path:
        joblib.dump((sequences, targets, relevance_scores, stock_indices), ranking_data_path)
        print(f"数据集已保存到: {ranking_data_path}")
    
    return sequences, targets, relevance_scores, stock_indices

def create_dataset(data, features, sequence_length, ranking_data_path=None):
    """保持原有接口，但内部调用新的排序数据集创建函数"""
    return create_ranking_dataset_multiprocess(data, features, sequence_length, ranking_data_path)

def create_ranking_dataset_vectorized(
    data,
    features,
    sequence_length,
    ranking_data_path=None,
    min_window_end_date=None,
    max_window_end_date=None,
):
    """
    向量化加速版本：预计算每只股票的所有滑动窗口，再按日期聚合。
    返回每日序列、标签、相关性、股票索引和预测日期。
    """
    # if ranking_data_path and os.path.exists(ranking_data_path):
    #     print(f"加载已有的排序数据集: {ranking_data_path}")
    #     return joblib.load(ranking_data_path)

    print("正在创建排序数据集（向量化加速版本）...")
    # data.rename(columns={'stock_idx': 'instrument'}, inplace=True)
    data = data.copy()
    data.rename(columns={'日期': 'datetime'}, inplace=True)
    data['datetime'] = pd.to_datetime(data['datetime'])

    # 1. 确保数据按股票和时间排序
    data = data.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    
    target_columns = [
        'label',
        'risk_1d_target',
        'risk_3d_target',
        'regime_target',
    ]
    missing_targets = [
        column for column in target_columns
        if column not in data.columns
    ]
    if missing_targets:
        raise ValueError(f'排序数据集缺少训练目标: {missing_targets}')
    data = data.dropna(subset=target_columns)
    
    # 3. 为每只股票生成所有滑动窗口
    # 仅保留满足以下条件的 end_date：
    # - 历史窗口长度满足 sequence_length
    # - end_date 之后存在 5 条未来交易日数据
    all_windows = []  # 每个元素: (end_date, stock_code, sequence, target)

    print("Step 1: 为每只股票生成滑动窗口...")
    grouped = data.groupby('instrument')
    
    for stock_code, group in tqdm(grouped, desc="Processing stocks"):
        if len(group) < sequence_length:
            continue
        
        # 提取特征和 label
        feature_values = group[features].values.astype(np.float32)  # (T, F)
        labels = group['label'].values.astype(np.float32)           # (T,)
        risk_1d_labels = group['risk_1d_target'].values.astype(np.float32)
        risk_3d_labels = group['risk_3d_target'].values.astype(np.float32)
        regime_labels = group['regime_target'].values.astype(np.float32)
        dates = group['datetime'].values                             # (T,)

        # 生成滑动窗口：从第 sequence_length-1 行开始（0-indexed）
        num_windows = len(group) - sequence_length + 1
        for i in range(num_windows):
            end_idx = i + sequence_length - 1

            seq = feature_values[i : i + sequence_length]   # (L, F)
            target = labels[end_idx]                        # label 对应窗口最后一天的次日涨跌幅
            end_date = dates[end_idx]                       # 窗口结束日期（即预测日）
            all_windows.append((
                end_date,
                stock_code,
                seq,
                target,
                risk_1d_labels[end_idx],
                risk_3d_labels[end_idx],
                regime_labels[end_idx],
            ))

    # 4. 转为 DataFrame 便于按日期聚合
    print("Step 2: 按日期聚合窗口...")
    window_df = pd.DataFrame(
        all_windows,
        columns=[
            'date',
            'stock_code',
            'seq',
            'target',
            'risk_1d_target',
            'risk_3d_target',
            'regime_target',
        ],
    )

    # 5. 按 date 分组，构建每日样本
    sequences = []
    targets = []
    relevance_scores = []
    stock_indices = []
    prediction_dates = []
    risk_1d_targets = []
    risk_3d_targets = []
    regime_targets = []

    print("Step 3: 构建每日样本并计算 relevance...")
    grouped_by_date = window_df.groupby('date')

    if min_window_end_date is not None:
        min_window_end_date = pd.to_datetime(min_window_end_date)
    if max_window_end_date is not None:
        max_window_end_date = pd.to_datetime(max_window_end_date)
    
    for date, group in tqdm(grouped_by_date, desc="Aggregating by date"):
        if min_window_end_date is not None and pd.to_datetime(date) < min_window_end_date:
            continue
        if max_window_end_date is not None and pd.to_datetime(date) > max_window_end_date:
            continue

        if len(group) < 10:
            continue
        
        # 提取数据
        day_seqs = np.stack(group['seq'].values)          # (N, L, F)
        day_targets = group['target'].values              # (N,)
        day_stocks = group['stock_code'].tolist()         # [str]

        # 计算 relevance（与原逻辑一致）
        sorted_indices = np.argsort(day_targets)[::-1]
        relevance = np.zeros_like(day_targets, dtype=np.float32)
        for rank, idx in enumerate(sorted_indices):
            relevance[idx] = len(day_targets) - rank

        sequences.append(day_seqs)
        targets.append(day_targets)
        relevance_scores.append(relevance)
        stock_indices.append(day_stocks)
        prediction_dates.append(pd.Timestamp(date).strftime('%Y-%m-%d'))
        risk_1d_targets.append(
            group['risk_1d_target'].to_numpy(dtype=np.float32)
        )
        risk_3d_targets.append(
            group['risk_3d_target'].to_numpy(dtype=np.float32)
        )
        regime_targets.append(
            group['regime_target'].to_numpy(dtype=np.float32)
        )

    print(f"成功创建 {len(sequences)} 个训练样本")
    if len(sequences) > 0:
        avg_stocks = np.mean([len(seq) for seq in sequences])
        print(f"每个样本平均包含 {avg_stocks:.1f} 只股票")

    # 6. 保存
    # if ranking_data_path:
    #     joblib.dump((sequences, targets, relevance_scores, stock_indices), ranking_data_path)
    #     print(f"数据集已保存到: {ranking_data_path}")

    return (
        sequences,
        targets,
        relevance_scores,
        stock_indices,
        prediction_dates,
        risk_1d_targets,
        risk_3d_targets,
        regime_targets,
    )


def percentile_ranks(values):
    """将横截面分数转换为 [0, 1] 百分位，消除不同模型的分数尺度差异。"""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError('percentile_ranks 需要非空一维数组')
    if not np.isfinite(values).all():
        raise ValueError('排名分数包含 NaN 或无穷值')
    if values.size == 1:
        return np.ones(1, dtype=np.float64)
    return (rankdata(values, method='average') - 1.0) / (values.size - 1.0)


def _stable_softmax(values):
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.max(values)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(dtype=np.float64)


def extract_selection_risk_context(
    sequences,
    feature_names,
    scaler,
    lookback=20,
):
    """从已标准化序列恢复严格因果的反转与相关性输入。"""
    sequences = np.asarray(sequences, dtype=np.float64)
    if sequences.ndim != 3 or sequences.shape[0] == 0:
        raise ValueError('风险上下文序列必须为非空 [股票, 时间, 特征] 数组')
    if sequences.shape[2] != len(feature_names):
        raise ValueError('风险上下文序列维度与特征名称不一致')
    if lookback < 2 or sequences.shape[1] < lookback:
        raise ValueError('selection_risk_lookback 超出可用序列长度')
    required = (*SELECTION_MOMENTUM_FEATURES, SELECTION_RETURN_FEATURE)
    missing = [name for name in required if name not in feature_names]
    if missing:
        raise ValueError(f'风险选择缺少特征: {missing}')
    if not hasattr(scaler, 'mean_') or not hasattr(scaler, 'scale_'):
        raise ValueError('风险上下文需要已拟合的 StandardScaler')

    def inverse_feature(name):
        index = feature_names.index(name)
        return (
            sequences[:, :, index] * float(scaler.scale_[index])
            + float(scaler.mean_[index])
        )

    momentum_percentiles = np.stack([
        inverse_feature(name)[:, -1]
        for name in SELECTION_MOMENTUM_FEATURES
    ], axis=1)
    return_history = inverse_feature(SELECTION_RETURN_FEATURE)[:, -lookback:]
    momentum_percentiles = np.nan_to_num(
        momentum_percentiles,
        nan=0.5,
        posinf=1.0,
        neginf=0.0,
    )
    momentum_percentiles = np.clip(momentum_percentiles, 0.0, 1.0)
    return_history = np.nan_to_num(
        return_history,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return {
        'momentum_percentiles': momentum_percentiles,
        'return_history': return_history,
    }


def _positive_correlation_matrix(return_history):
    return_history = np.asarray(return_history, dtype=np.float64)
    if return_history.ndim != 2 or return_history.shape[1] < 2:
        raise ValueError('return_history 必须为 [股票, 时间] 二维数组')
    centered = return_history - return_history.mean(axis=1, keepdims=True)
    norms = np.sqrt(np.sum(centered ** 2, axis=1, keepdims=True))
    normalized = np.divide(
        centered,
        norms,
        out=np.zeros_like(centered),
        where=norms > 1e-12,
    )
    correlations = normalized @ normalized.T
    correlations = np.clip(correlations, -1.0, 1.0)
    np.fill_diagonal(correlations, 1.0)
    return np.maximum(correlations, 0.0)


def _mean_pairwise_correlation(correlations, indices):
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size < 2:
        return 0.0
    selected = correlations[np.ix_(indices, indices)]
    values = selected[np.triu_indices(indices.size, k=1)]
    return float(values.mean()) if values.size else 0.0


def select_risk_aware_top_indices(
    scores,
    momentum_percentiles,
    return_history,
    risk_gamma=0.0,
    candidate_k=20,
    top_k=5,
):
    """从排名候选中以反转风险和历史正相关惩罚贪心选择Top-k。"""
    scores = np.asarray(scores, dtype=np.float64)
    momentum_percentiles = np.asarray(
        momentum_percentiles,
        dtype=np.float64,
    )
    return_history = np.asarray(return_history, dtype=np.float64)
    if scores.ndim != 1 or scores.size < top_k:
        raise ValueError(f'风险选择至少需要 {top_k} 只股票')
    if momentum_percentiles.shape != (scores.size, 3):
        raise ValueError('momentum_percentiles 必须为 [股票, 3]')
    if return_history.ndim != 2 or return_history.shape[0] != scores.size:
        raise ValueError('return_history 股票维度与分数不一致')
    if risk_gamma < 0:
        raise ValueError('selection_risk_gamma 不能为负')
    candidate_k = min(int(candidate_k), scores.size)
    if candidate_k < top_k:
        raise ValueError('selection_candidate_k 不能小于 Top-k')
    if not (
        np.isfinite(scores).all()
        and np.isfinite(momentum_percentiles).all()
        and np.isfinite(return_history).all()
    ):
        raise ValueError('风险选择输入包含 NaN 或无穷值')

    raw_order = np.lexsort((np.arange(scores.size), -scores))
    raw_top_indices = raw_order[:top_k]
    candidate_indices = raw_order[:candidate_k]
    cs5, cs20, cs60 = momentum_percentiles.T
    reversal_risk = (
        0.6 * np.maximum(cs60 - cs5, 0.0)
        + 0.4 * np.maximum(cs20 - cs5, 0.0)
    )
    positive_correlations = _positive_correlation_matrix(return_history)

    if risk_gamma == 0.0:
        selected = raw_top_indices.tolist()
        selected_correlation_risks = [
            0.0 if position == 0 else float(
                positive_correlations[index, selected[:position]].mean()
            )
            for position, index in enumerate(selected)
        ]
    else:
        selected = []
        selected_correlation_risks = []
        remaining = set(int(index) for index in candidate_indices)
        while len(selected) < top_k:
            best_index = None
            best_utility = -float('inf')
            best_correlation_risk = 0.0
            for index in sorted(remaining):
                correlation_risk = (
                    0.0
                    if not selected
                    else float(
                        positive_correlations[index, selected].mean()
                    )
                )
                combined_risk = (
                    0.5 * reversal_risk[index]
                    if not selected
                    else (
                        0.5 * reversal_risk[index]
                        + 0.5 * correlation_risk
                    )
                )
                utility = scores[index] - risk_gamma * combined_risk
                if (
                    utility > best_utility
                    or (
                        np.isclose(utility, best_utility, rtol=0.0, atol=1e-15)
                        and (best_index is None or index < best_index)
                    )
                ):
                    best_index = index
                    best_utility = utility
                    best_correlation_risk = correlation_risk
            selected.append(best_index)
            selected_correlation_risks.append(best_correlation_risk)
            remaining.remove(best_index)

    selected = np.asarray(selected, dtype=np.int64)
    raw_rank_by_index = np.empty(scores.size, dtype=np.int64)
    raw_rank_by_index[raw_order] = np.arange(1, scores.size + 1)
    return {
        'top_indices': selected,
        'raw_top_indices': raw_top_indices,
        'selected_raw_ranks': raw_rank_by_index[selected],
        'reversal_risk': reversal_risk[selected],
        'correlation_risk': np.asarray(
            selected_correlation_risks,
            dtype=np.float64,
        ),
        'mean_positive_correlation': _mean_pairwise_correlation(
            positive_correlations,
            selected,
        ),
        'raw_mean_positive_correlation': _mean_pairwise_correlation(
            positive_correlations,
            raw_top_indices,
        ),
    }


def build_ensemble_portfolio(
    score_matrix,
    allocation_matrix,
    exposures,
    min_exposure,
    max_exposure,
    allocation_temperature=1.0,
    allocation_blend=1.0,
    disagreement_gamma=0.0,
    selection_risk_context=None,
    selection_risk_gamma=0.0,
    selection_candidate_k=20,
    risk_probability_matrix=None,
    regime_gates=None,
    risk_score_penalty=0.0,
    correlation_exposure_gamma=0.0,
    exposure_head_blend=1.0,
    fixed_exposure_baseline=0.6231689453125,
    top_k=5,
):
    """用 rank ensemble 选股，并按模型分歧将仓位向最低仓位收缩。"""
    score_matrix = np.asarray(score_matrix, dtype=np.float64)
    allocation_matrix = np.asarray(allocation_matrix, dtype=np.float64)
    exposures = np.asarray(exposures, dtype=np.float64).reshape(-1)
    if score_matrix.ndim != 2 or allocation_matrix.shape != score_matrix.shape:
        raise ValueError('score_matrix 与 allocation_matrix 必须是同形二维数组')
    num_models, num_stocks = score_matrix.shape
    if num_models == 0 or num_stocks < top_k:
        raise ValueError(f'集成模型为空或可选股票不足 {top_k} 只')
    if exposures.shape != (num_models,):
        raise ValueError('每个集成模型必须提供一个总仓位')
    if not (
        np.isfinite(score_matrix).all()
        and np.isfinite(allocation_matrix).all()
        and np.isfinite(exposures).all()
    ):
        raise ValueError('集成模型输出包含 NaN 或无穷值')
    if not 0.0 <= min_exposure < max_exposure < 1.0:
        raise ValueError('仓位范围必须满足 0 <= min_exposure < max_exposure < 1')
    if (
        allocation_temperature <= 0
        or disagreement_gamma < 0
        or risk_score_penalty < 0
        or correlation_exposure_gamma < 0
    ):
        raise ValueError('temperature 必须大于 0，gamma 不能为负')
    if not 0.0 <= allocation_blend <= 1.0:
        raise ValueError('allocation_blend 必须位于 [0, 1]')
    if not 0.0 <= exposure_head_blend <= 1.0:
        raise ValueError('exposure_head_blend 必须位于 [0, 1]')

    raw_percentile_matrix = np.stack(
        [percentile_ranks(scores) for scores in score_matrix],
        axis=0,
    )
    percentile_matrix = raw_percentile_matrix
    if risk_score_penalty > 0:
        risk_probability_matrix = np.asarray(
            risk_probability_matrix,
            dtype=np.float64,
        )
        regime_gates = np.asarray(regime_gates, dtype=np.float64).reshape(-1)
        if risk_probability_matrix.shape != score_matrix.shape:
            raise ValueError('风险概率矩阵与排名分数矩阵形状不一致')
        if regime_gates.shape != (num_models,):
            raise ValueError('每个模型必须提供一个市场压力门控值')
        if not (
            np.isfinite(risk_probability_matrix).all()
            and np.isfinite(regime_gates).all()
        ):
            raise ValueError('风险概率或市场压力包含 NaN/无穷值')
        risk_percentile_matrix = np.stack([
            percentile_ranks(probabilities)
            for probabilities in risk_probability_matrix
        ])
        percentile_matrix = (
            raw_percentile_matrix
            - risk_score_penalty
            * np.clip(regime_gates, 0.0, 1.0)[:, None]
            * risk_percentile_matrix
        )
    ensemble_scores = percentile_matrix.mean(axis=0)
    raw_ensemble_scores = raw_percentile_matrix.mean(axis=0)
    unadjusted_top_indices = np.lexsort(
        (np.arange(num_stocks), -raw_ensemble_scores)
    )[:top_k]
    if selection_risk_context is None:
        if selection_risk_gamma != 0.0:
            raise ValueError('非零 selection_risk_gamma 需要风险上下文')
        raw_top_indices = np.lexsort(
            (np.arange(num_stocks), -ensemble_scores)
        )[:top_k]
        risk_selection = {
            'top_indices': raw_top_indices,
            'raw_top_indices': raw_top_indices,
            'selected_raw_ranks': np.arange(1, top_k + 1),
            'reversal_risk': np.zeros(top_k, dtype=np.float64),
            'correlation_risk': np.zeros(top_k, dtype=np.float64),
            'mean_positive_correlation': 0.0,
            'raw_mean_positive_correlation': 0.0,
        }
    else:
        risk_selection = select_risk_aware_top_indices(
            ensemble_scores,
            selection_risk_context['momentum_percentiles'],
            selection_risk_context['return_history'],
            risk_gamma=selection_risk_gamma,
            candidate_k=selection_candidate_k,
            top_k=top_k,
        )
    top_indices = risk_selection['top_indices']

    learned_weights = np.stack([
        _stable_softmax(
            allocation_matrix[model_idx, top_indices] / allocation_temperature
        )
        for model_idx in range(num_models)
    ]).mean(axis=0)
    equal_weights = np.full(top_k, 1.0 / top_k, dtype=np.float64)
    relative_weights = (
        allocation_blend * learned_weights
        + (1.0 - allocation_blend) * equal_weights
    )
    relative_weights /= relative_weights.sum(dtype=np.float64)

    selected_disagreement = percentile_matrix[:, top_indices].std(axis=0)
    mean_disagreement = float(selected_disagreement.mean())
    head_base_exposure = float(np.median(exposures))
    head_base_exposure = float(
        np.clip(head_base_exposure, min_exposure, max_exposure)
    )
    fixed_exposure_baseline = float(
        np.clip(fixed_exposure_baseline, min_exposure, max_exposure)
    )
    base_exposure = (
        exposure_head_blend * head_base_exposure
        + (1.0 - exposure_head_blend) * fixed_exposure_baseline
    )
    base_exposure = float(np.clip(base_exposure, min_exposure, max_exposure))
    adjusted_exposure = min_exposure + (
        base_exposure - min_exposure
    ) * np.exp(-disagreement_gamma * mean_disagreement)
    adjusted_exposure = min_exposure + (
        adjusted_exposure - min_exposure
    ) * np.exp(
        -correlation_exposure_gamma
        * risk_selection['mean_positive_correlation']
    )
    adjusted_exposure = float(
        np.clip(adjusted_exposure, min_exposure, max_exposure)
    )
    positions = relative_weights * adjusted_exposure
    positions[np.argmax(positions)] += (
        adjusted_exposure - positions.sum(dtype=np.float64)
    )
    if (positions < 0).any() or positions.sum(dtype=np.float64) > 1.0:
        raise ValueError('集成仓位违反非负或总仓位不超过 1 的约束')

    return {
        'top_indices': top_indices,
        'positions': positions,
        'relative_weights': relative_weights,
        'ensemble_scores': ensemble_scores,
        'raw_ensemble_scores': raw_ensemble_scores,
        'percentile_matrix': percentile_matrix,
        'raw_percentile_matrix': raw_percentile_matrix,
        'selected_disagreement': selected_disagreement,
        'mean_disagreement': mean_disagreement,
        'head_base_exposure': head_base_exposure,
        'exposure_head_blend': float(exposure_head_blend),
        'fixed_exposure_baseline': fixed_exposure_baseline,
        'base_exposure': base_exposure,
        'exposure': adjusted_exposure,
        'raw_top_indices': risk_selection['raw_top_indices'],
        'unadjusted_top_indices': unadjusted_top_indices,
        'risk_score_penalty': float(risk_score_penalty),
        'correlation_exposure_gamma': float(
            correlation_exposure_gamma
        ),
        'selected_raw_ranks': risk_selection['selected_raw_ranks'],
        'selected_reversal_risk': risk_selection['reversal_risk'],
        'selected_correlation_risk': risk_selection['correlation_risk'],
        'mean_positive_correlation': risk_selection[
            'mean_positive_correlation'
        ],
        'raw_mean_positive_correlation': risk_selection[
            'raw_mean_positive_correlation'
        ],
    }


def align_oof_prediction_records(records_by_model, fold):
    """按日期和股票索引严格对齐同一折的多个随机种子 OOF 输出。"""
    if not records_by_model:
        raise ValueError('缺少 OOF 模型输出')
    sorted_records = [
        sorted(records, key=lambda record: record['prediction_date'])
        for records in records_by_model
    ]
    reference_dates = [
        record['prediction_date'] for record in sorted_records[0]
    ]
    if not reference_dates:
        raise ValueError(f'Fold {fold} 没有 OOF 记录')

    days = []
    for records in sorted_records[1:]:
        if [record['prediction_date'] for record in records] != reference_dates:
            raise ValueError(f'Fold {fold} 的多模型 OOF 日期不一致')
    for day_idx, prediction_date in enumerate(reference_dates):
        reference = sorted_records[0][day_idx]
        reference_stocks = np.asarray(reference['stock_indices'], dtype=np.int64)
        has_risk_context = all(
            key in reference
            for key in ('momentum_percentiles', 'return_history')
        )
        for records in sorted_records[1:]:
            stocks = np.asarray(records[day_idx]['stock_indices'], dtype=np.int64)
            if not np.array_equal(stocks, reference_stocks):
                raise ValueError(
                    f'Fold {fold} 在 {prediction_date} 的股票顺序不一致'
                )
            if not np.allclose(
                records[day_idx]['targets'],
                reference['targets'],
                rtol=0.0,
                atol=1e-8,
            ):
                raise ValueError(
                    f'Fold {fold} 在 {prediction_date} 的真实收益不一致'
                )
            for key in (
                'risk_1d_targets',
                'risk_3d_targets',
                'risk_5d_targets',
            ):
                if (
                    key in reference
                    and key in records[day_idx]
                    and not np.allclose(
                        records[day_idx][key],
                        reference[key],
                        rtol=0.0,
                        atol=1e-8,
                    )
                ):
                    raise ValueError(
                        f'Fold {fold} 在 {prediction_date} 的 {key} 不一致'
                    )
            if (
                'regime_target' in reference
                and 'regime_target' in records[day_idx]
                and not np.isclose(
                    records[day_idx]['regime_target'],
                    reference['regime_target'],
                    rtol=0.0,
                    atol=1e-8,
                )
            ):
                raise ValueError(
                    f'Fold {fold} 在 {prediction_date} 的状态标签不一致'
                )
            other_has_risk_context = all(
                key in records[day_idx]
                for key in ('momentum_percentiles', 'return_history')
            )
            if other_has_risk_context != has_risk_context:
                raise ValueError(
                    f'Fold {fold} 在 {prediction_date} 的风险上下文不一致'
                )
            if has_risk_context:
                for key in ('momentum_percentiles', 'return_history'):
                    if not np.allclose(
                        records[day_idx][key],
                        reference[key],
                        rtol=0.0,
                        atol=1e-8,
                    ):
                        raise ValueError(
                            f'Fold {fold} 在 {prediction_date} 的 {key} 不一致'
                        )
        day = {
            'fold': int(fold),
            'prediction_date': prediction_date,
            'stock_indices': reference_stocks,
            'targets': np.asarray(reference['targets'], dtype=np.float64),
            'scores': np.stack([
                np.asarray(records[day_idx]['scores'], dtype=np.float64)
                for records in sorted_records
            ]),
            'allocation_logits': np.stack([
                np.asarray(
                    records[day_idx]['allocation_logits'],
                    dtype=np.float64,
                )
                for records in sorted_records
            ]),
            'exposures': np.asarray([
                records[day_idx]['exposure'] for records in sorted_records
            ], dtype=np.float64),
            'regime_gates': np.asarray([
                records[day_idx].get('regime_gate', 0.0)
                for records in sorted_records
            ], dtype=np.float64),
            'risk_1d_probabilities': np.stack([
                np.asarray(
                    records[day_idx].get(
                        'risk_1d_probabilities',
                        np.full(reference_stocks.size, 0.5),
                    ),
                    dtype=np.float64,
                )
                for records in sorted_records
            ]),
            'risk_3d_probabilities': np.stack([
                np.asarray(
                    records[day_idx].get(
                        'risk_3d_probabilities',
                        np.full(reference_stocks.size, 0.5),
                    ),
                    dtype=np.float64,
                )
                for records in sorted_records
            ]),
            'risk_5d_probabilities': np.stack([
                np.asarray(
                    records[day_idx].get(
                        'risk_5d_probabilities',
                        np.full(reference_stocks.size, 0.5),
                    ),
                    dtype=np.float64,
                )
                for records in sorted_records
            ]),
            'risk_1d_targets': np.asarray(
                reference.get(
                    'risk_1d_targets',
                    np.full(reference_stocks.size, 0.5),
                ),
                dtype=np.float64,
            ),
            'risk_3d_targets': np.asarray(
                reference.get(
                    'risk_3d_targets',
                    np.full(reference_stocks.size, 0.5),
                ),
                dtype=np.float64,
            ),
            'risk_5d_targets': np.asarray(
                reference.get(
                    'risk_5d_targets',
                    np.full(reference_stocks.size, 0.5),
                ),
                dtype=np.float64,
            ),
            'regime_target': float(reference.get('regime_target', 0.5)),
        }
        if has_risk_context:
            day['selection_risk_context'] = {
                'momentum_percentiles': np.asarray(
                    reference['momentum_percentiles'],
                    dtype=np.float64,
                ),
                'return_history': np.asarray(
                    reference['return_history'],
                    dtype=np.float64,
                ),
            }
        days.append(day)
    return days


def summarize_ensemble_days(
    ensemble_days,
    min_exposure,
    max_exposure,
    allocation_temperature,
    allocation_blend,
    disagreement_gamma,
    selection_risk_gamma=0.0,
    selection_candidate_k=20,
    risk_score_penalty=0.0,
    risk_1d_blend=0.40,
    risk_3d_blend=0.60,
    risk_5d_blend=0.0,
    correlation_exposure_gamma=0.0,
    exposure_head_blend=1.0,
    fixed_exposure_baseline=0.6231689453125,
    downside_weight=0.5,
    top_k=5,
    include_daily=False,
):
    """计算 OOF ensemble 的收益分解、下行风险和 Rank IC。"""
    risk_blends = np.asarray(
        [risk_1d_blend, risk_3d_blend, risk_5d_blend],
        dtype=np.float64,
    )
    if (risk_blends < 0).any() or risk_blends.sum() <= 0:
        raise ValueError('风险头混合权重必须非负且权重和大于0')
    risk_blends /= risk_blends.sum()
    risk_1d_blend, risk_3d_blend, risk_5d_blend = risk_blends
    daily = []
    for day in ensemble_days:
        combined_risk_probabilities = (
            risk_1d_blend * day['risk_1d_probabilities']
            + risk_3d_blend * day['risk_3d_probabilities']
            + risk_5d_blend * day['risk_5d_probabilities']
        )
        portfolio = build_ensemble_portfolio(
            day['scores'],
            day['allocation_logits'],
            day['exposures'],
            min_exposure=min_exposure,
            max_exposure=max_exposure,
            allocation_temperature=allocation_temperature,
            allocation_blend=allocation_blend,
            disagreement_gamma=disagreement_gamma,
            selection_risk_context=day.get('selection_risk_context'),
            selection_risk_gamma=selection_risk_gamma,
            selection_candidate_k=selection_candidate_k,
            risk_probability_matrix=combined_risk_probabilities,
            regime_gates=day['regime_gates'],
            risk_score_penalty=risk_score_penalty,
            correlation_exposure_gamma=correlation_exposure_gamma,
            exposure_head_blend=exposure_head_blend,
            fixed_exposure_baseline=fixed_exposure_baseline,
            top_k=top_k,
        )
        selected = portfolio['top_indices']
        selected_returns = day['targets'][selected]
        selected_risk_1d = day['risk_1d_probabilities'][:, selected].mean()
        selected_risk_3d = day['risk_3d_probabilities'][:, selected].mean()
        selected_risk_5d = day['risk_5d_probabilities'][:, selected].mean()
        mean_risk_1d_prediction = day['risk_1d_probabilities'].mean(axis=0)
        mean_risk_3d_prediction = day['risk_3d_probabilities'].mean(axis=0)
        mean_risk_5d_prediction = day['risk_5d_probabilities'].mean(axis=0)
        regime_prediction = float(np.median(day['regime_gates']))
        equal_full_return = float(selected_returns.mean())
        allocation_only_return = float(
            np.dot(portfolio['relative_weights'], selected_returns)
        )
        weighted_return = float(
            np.dot(portfolio['positions'], selected_returns)
        )
        allocation_at_exposure_contribution = float(
            portfolio['exposure']
            * (allocation_only_return - equal_full_return)
        )
        fixed_exposure_return = float(
            fixed_exposure_baseline * allocation_only_return
        )
        rank_ic = spearmanr(
            portfolio['ensemble_scores'],
            day['targets'],
        ).statistic
        daily.append({
            'fold': int(day['fold']),
            'prediction_date': day['prediction_date'],
            'top5_return': equal_full_return,
            'equal_weight_at_exposure_return': (
                equal_full_return * portfolio['exposure']
            ),
            'allocation_only_return': allocation_only_return,
            'weighted_portfolio_return': weighted_return,
            'allocation_contribution': (
                allocation_only_return - equal_full_return
            ),
            'allocation_at_exposure_contribution': (
                allocation_at_exposure_contribution
            ),
            'exposure_contribution': (
                weighted_return - allocation_only_return
            ),
            'fixed_exposure_return': fixed_exposure_return,
            'gross_exposure': portfolio['exposure'],
            'head_gross_exposure': portfolio['head_base_exposure'],
            'exposure_head_blend': float(exposure_head_blend),
            'cash_weight': 1.0 - portfolio['exposure'],
            'model_disagreement': portfolio['mean_disagreement'],
            'regime_gate': regime_prediction,
            'regime_target': float(day['regime_target']),
            'selected_risk_1d': float(selected_risk_1d),
            'selected_risk_3d': float(selected_risk_3d),
            'selected_risk_5d': float(selected_risk_5d),
            'risk_1d_brier': float(np.mean(
                (
                    mean_risk_1d_prediction
                    - day['risk_1d_targets']
                ) ** 2
            )),
            'risk_3d_brier': float(np.mean(
                (
                    mean_risk_3d_prediction
                    - day['risk_3d_targets']
                ) ** 2
            )),
            'risk_5d_brier': float(np.mean(
                (
                    mean_risk_5d_prediction
                    - day['risk_5d_targets']
                ) ** 2
            )),
            'regime_brier': float(
                (regime_prediction - day['regime_target']) ** 2
            ),
            'mean_positive_correlation': portfolio[
                'mean_positive_correlation'
            ],
            'raw_mean_positive_correlation': portfolio[
                'raw_mean_positive_correlation'
            ],
            'mean_reversal_risk': float(
                np.mean(portfolio['selected_reversal_risk'])
            ),
            'risk_score_penalty': float(risk_score_penalty),
            'correlation_exposure_gamma': float(
                correlation_exposure_gamma
            ),
            'rank_ic': float(rank_ic) if np.isfinite(rank_ic) else 0.0,
        })

    if not daily:
        raise ValueError('没有可用于汇总的 ensemble OOF 日期')
    weighted_returns = np.asarray([
        row['weighted_portfolio_return'] for row in daily
    ], dtype=np.float64)
    negative_returns = np.minimum(weighted_returns, 0.0)
    downside_deviation = float(np.sqrt(np.mean(negative_returns ** 2)))
    fixed_exposure_returns = np.asarray([
        row['fixed_exposure_return'] for row in daily
    ], dtype=np.float64)
    fixed_negative_returns = np.minimum(fixed_exposure_returns, 0.0)
    fixed_downside_deviation = float(np.sqrt(
        np.mean(fixed_negative_returns ** 2)
    ))
    gross_exposures = np.asarray([
        row['gross_exposure'] for row in daily
    ], dtype=np.float64)
    regime_gates = np.asarray([
        row['regime_gate'] for row in daily
    ], dtype=np.float64)
    top5_returns = np.asarray([
        row['top5_return'] for row in daily
    ], dtype=np.float64)
    if regime_gates.std() < 1e-12 or top5_returns.std() < 1e-12:
        regime_return_correlation = 0.0
    else:
        regime_return_correlation = spearmanr(
            regime_gates,
            top5_returns,
        ).statistic
    if gross_exposures.std() < 1e-12 or top5_returns.std() < 1e-12:
        exposure_return_correlation = 0.0
    else:
        exposure_return_correlation = spearmanr(
            gross_exposures,
            top5_returns,
        ).statistic

    def mean(key):
        return float(np.mean([row[key] for row in daily]))

    fold_rows = {}
    for row in daily:
        fold_rows.setdefault(row['fold'], []).append(row)
    fold_summaries = [
        {
            'fold': int(fold),
            'mean_top5_return': float(np.mean([
                row['top5_return'] for row in rows
            ])),
            'mean_weighted_portfolio_return': float(np.mean([
                row['weighted_portfolio_return'] for row in rows
            ])),
            'worst_weighted_portfolio_return': float(np.min([
                row['weighted_portfolio_return'] for row in rows
            ])),
            'mean_rank_ic': float(np.mean([
                row['rank_ic'] for row in rows
            ])),
            'positive_rate': float(np.mean(np.asarray([
                row['weighted_portfolio_return'] for row in rows
            ]) > 0.0)),
            'num_evaluation_dates': len(rows),
        }
        for fold, rows in sorted(fold_rows.items())
    ]
    summary = {
        'num_evaluation_dates': len(daily),
        'mean_top5_return': mean('top5_return'),
        'mean_equal_weight_at_exposure_return': mean(
            'equal_weight_at_exposure_return'
        ),
        'mean_allocation_only_return': mean('allocation_only_return'),
        'mean_weighted_portfolio_return': float(weighted_returns.mean()),
        'worst_weighted_portfolio_return': float(weighted_returns.min()),
        'p10_weighted_portfolio_return': float(
            np.quantile(weighted_returns, 0.10)
        ),
        'std_weighted_portfolio_return': float(weighted_returns.std()),
        'positive_rate': float(np.mean(weighted_returns > 0.0)),
        'downside_deviation': downside_deviation,
        'mean_rank_ic': mean('rank_ic'),
        'worst_rank_ic': float(min(row['rank_ic'] for row in daily)),
        'mean_gross_exposure': mean('gross_exposure'),
        'mean_head_gross_exposure': mean('head_gross_exposure'),
        'mean_cash_weight': mean('cash_weight'),
        'mean_model_disagreement': mean('model_disagreement'),
        'mean_regime_gate': mean('regime_gate'),
        'regime_gate_std': float(regime_gates.std()),
        'mean_selected_risk_1d': mean('selected_risk_1d'),
        'mean_selected_risk_3d': mean('selected_risk_3d'),
        'mean_selected_risk_5d': mean('selected_risk_5d'),
        'mean_risk_1d_brier': mean('risk_1d_brier'),
        'mean_risk_3d_brier': mean('risk_3d_brier'),
        'mean_risk_5d_brier': mean('risk_5d_brier'),
        'mean_regime_brier': mean('regime_brier'),
        'regime_return_spearman': float(
            regime_return_correlation
            if np.isfinite(regime_return_correlation)
            else 0.0
        ),
        'mean_allocation_contribution': mean('allocation_contribution'),
        'mean_allocation_at_exposure_contribution': mean(
            'allocation_at_exposure_contribution'
        ),
        'mean_exposure_contribution': mean('exposure_contribution'),
        'exposure_std': float(gross_exposures.std()),
        'exposure_return_spearman': float(
            exposure_return_correlation
            if np.isfinite(exposure_return_correlation)
            else 0.0
        ),
        'mean_positive_correlation': mean('mean_positive_correlation'),
        'raw_mean_positive_correlation': mean(
            'raw_mean_positive_correlation'
        ),
        'mean_reversal_risk': mean('mean_reversal_risk'),
        'risk_score_penalty': float(risk_score_penalty),
        'correlation_exposure_gamma': float(correlation_exposure_gamma),
        'exposure_head_blend': float(exposure_head_blend),
        'worst_fold_weighted_portfolio_return': float(min(
            row['mean_weighted_portfolio_return'] for row in fold_summaries
        )),
        'worst_fold_top5_return': float(min(
            row['mean_top5_return'] for row in fold_summaries
        )),
        'policy_objective': float(
            weighted_returns.mean() - downside_weight * downside_deviation
        ),
        'fixed_exposure_policy_objective': float(
            fixed_exposure_returns.mean()
            - downside_weight * fixed_downside_deviation
        ),
        'folds': fold_summaries,
    }
    if include_daily:
        summary['daily'] = daily
    return summary


def calibrate_ensemble_policy(
    ensemble_days,
    min_exposure,
    max_exposure,
    allocation_temperature,
    allocation_blend_grid,
    disagreement_gamma_grid,
    selection_risk_gamma_grid=(0.0,),
    risk_score_penalty_grid=(0.0,),
    risk_1d_blend=0.40,
    risk_3d_blend=0.60,
    risk_5d_blend=0.0,
    correlation_exposure_gamma_grid=(0.0,),
    exposure_head_blend_grid=(1.0,),
    selection_candidate_k=20,
    fixed_exposure_baseline=0.6231689453125,
    downside_weight=0.5,
    top_k=5,
):
    """仅用 OOF 收益网格选择 allocation 混合与分歧降仓强度。"""
    risk_blends = np.asarray(
        [risk_1d_blend, risk_3d_blend, risk_5d_blend],
        dtype=np.float64,
    )
    if (risk_blends < 0).any() or risk_blends.sum() <= 0:
        raise ValueError('风险头混合权重必须非负且权重和大于0')
    risk_blends /= risk_blends.sum()
    risk_1d_blend, risk_3d_blend, risk_5d_blend = risk_blends
    candidates = []
    for allocation_blend in allocation_blend_grid:
        for disagreement_gamma in disagreement_gamma_grid:
            for selection_risk_gamma in selection_risk_gamma_grid:
                for risk_score_penalty in risk_score_penalty_grid:
                    for correlation_exposure_gamma in (
                        correlation_exposure_gamma_grid
                    ):
                        for exposure_head_blend in exposure_head_blend_grid:
                            metrics = summarize_ensemble_days(
                                ensemble_days,
                                min_exposure=min_exposure,
                                max_exposure=max_exposure,
                                allocation_temperature=allocation_temperature,
                                allocation_blend=float(allocation_blend),
                                disagreement_gamma=float(disagreement_gamma),
                                selection_risk_gamma=float(selection_risk_gamma),
                                selection_candidate_k=selection_candidate_k,
                                risk_score_penalty=float(risk_score_penalty),
                                risk_1d_blend=risk_1d_blend,
                                risk_3d_blend=risk_3d_blend,
                                risk_5d_blend=risk_5d_blend,
                                correlation_exposure_gamma=float(
                                    correlation_exposure_gamma
                                ),
                                exposure_head_blend=float(
                                    exposure_head_blend
                                ),
                                fixed_exposure_baseline=(
                                    fixed_exposure_baseline
                                ),
                                downside_weight=downside_weight,
                                top_k=top_k,
                            )
                            candidates.append({
                                'allocation_blend': float(allocation_blend),
                                'disagreement_gamma': float(
                                    disagreement_gamma
                                ),
                                'selection_risk_gamma': float(
                                    selection_risk_gamma
                                ),
                                'risk_score_penalty': float(
                                    risk_score_penalty
                                ),
                                'correlation_exposure_gamma': float(
                                    correlation_exposure_gamma
                                ),
                                'exposure_head_blend': float(
                                    exposure_head_blend
                                ),
                                'metrics': metrics,
                            })
    if not candidates:
        raise ValueError('ensemble policy 搜索网格不能为空')
    best = max(
        candidates,
        key=lambda candidate: (
            candidate['metrics']['policy_objective'],
            candidate['metrics']['mean_weighted_portfolio_return'],
            -candidate['metrics']['downside_deviation'],
            -candidate['disagreement_gamma'],
            -candidate['selection_risk_gamma'],
            -candidate['risk_score_penalty'],
            -candidate['correlation_exposure_gamma'],
            candidate['exposure_head_blend'],
            -abs(candidate['allocation_blend'] - 0.5),
        ),
    )
    best_metrics = summarize_ensemble_days(
        ensemble_days,
        min_exposure=min_exposure,
        max_exposure=max_exposure,
        allocation_temperature=allocation_temperature,
        allocation_blend=best['allocation_blend'],
        disagreement_gamma=best['disagreement_gamma'],
        selection_risk_gamma=best['selection_risk_gamma'],
        selection_candidate_k=selection_candidate_k,
        risk_score_penalty=best['risk_score_penalty'],
        risk_1d_blend=risk_1d_blend,
        risk_3d_blend=risk_3d_blend,
        risk_5d_blend=risk_5d_blend,
        correlation_exposure_gamma=best[
            'correlation_exposure_gamma'
        ],
        exposure_head_blend=best['exposure_head_blend'],
        fixed_exposure_baseline=fixed_exposure_baseline,
        downside_weight=downside_weight,
        top_k=top_k,
        include_daily=True,
    )
    return {
        'allocation_blend': best['allocation_blend'],
        'disagreement_gamma': best['disagreement_gamma'],
        'selection_risk_gamma': best['selection_risk_gamma'],
        'risk_score_penalty': best['risk_score_penalty'],
        'risk_1d_blend': float(risk_1d_blend),
        'risk_3d_blend': float(risk_3d_blend),
        'risk_5d_blend': float(risk_5d_blend),
        'correlation_exposure_gamma': best[
            'correlation_exposure_gamma'
        ],
        'exposure_head_blend': best['exposure_head_blend'],
        'selection_candidate_k': int(selection_candidate_k),
        'fixed_exposure_baseline': float(fixed_exposure_baseline),
        'min_exposure': float(min_exposure),
        'max_exposure': float(max_exposure),
        'allocation_temperature': float(allocation_temperature),
        'top_k': int(top_k),
        'downside_weight': float(downside_weight),
        'selection_metric': 'mean_return_minus_downside_weight_times_deviation',
        'oof_metrics': best_metrics,
        'grid_results': candidates,
    }
