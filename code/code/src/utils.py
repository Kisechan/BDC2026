import pandas as pd
import numpy as np
import joblib
import os
from tqdm import tqdm
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

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
INDUSTRY_RESIDUAL_FEATURE_SET = (
    '158+39_reduced25_relmarket12_risk15_indresid12'
)
INDUSTRY_RESIDUAL_FEATURES = (
    'indresid_market_return_1', 'indresid_market_return_3',
    'indresid_market_return_5', 'indresid_industry_return_1',
    'indresid_industry_return_3', 'indresid_industry_return_5',
    'indresid_market_return_1_pct', 'indresid_market_return_3_pct',
    'indresid_market_return_5_pct', 'indresid_industry_return_1_pct',
    'indresid_industry_return_3_pct', 'indresid_industry_return_5_pct',
)
INDUSTRY_ASOF_COLUMN = '_industry_asof'
INDUSTRY_NEUTRAL_TARGET = 'industry_neutral_label'
SELECTION_MOMENTUM_FEATURES = (
    'cs_return_5_pct',
    'cs_return_20_pct',
    'cs_return_60_pct',
)
SELECTION_RETURN_FEATURE = 'return_1'


def _industry_stock_key(values):
    """统一数值/零填充股票代码，供行业快照关联使用。"""
    raw = pd.Series(values, copy=False).astype('string').str.strip()
    numeric = pd.to_numeric(raw, errors='coerce')
    integer = numeric.notna() & np.isclose(numeric, np.round(numeric))
    result = raw.copy()
    result.loc[integer] = numeric.loc[integer].round().astype('Int64').astype('string')
    return result


def _prepare_industry_history(industry_history):
    if isinstance(industry_history, (str, os.PathLike)):
        industry_history = pd.read_csv(industry_history)
    required = {'effective_date', 'stock_id', 'industry'}
    missing = required.difference(industry_history.columns)
    if missing:
        raise ValueError(f'行业快照缺少必要列: {sorted(missing)}')
    history = industry_history.loc[:, ['effective_date', 'stock_id', 'industry']].copy()
    history['effective_date'] = pd.to_datetime(history['effective_date'], errors='coerce')
    history['_industry_key'] = _industry_stock_key(history['stock_id'])
    history['industry'] = history['industry'].astype('string').str.strip()
    history = history.dropna(subset=['effective_date', '_industry_key', 'industry'])
    duplicate = ['effective_date', '_industry_key']
    if (history.groupby(duplicate)['industry'].nunique() > 1).any():
        raise ValueError('同一股票和生效日存在冲突的行业快照')
    return history.drop_duplicates(duplicate).sort_values(
        ['effective_date', '_industry_key'], kind='mergesort'
    )


def attach_industry_asof(df, industry_history, industry_column=INDUSTRY_ASOF_COLUMN):
    """以最近且不晚于行情日的行业快照关联；绝不使用未来行业信息。"""
    required = {'股票代码', '日期'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'行业 as-of 关联缺少基础列: {sorted(missing)}')
    panel = df.copy()
    panel['_industry_order'] = np.arange(len(panel))
    panel['_industry_date'] = pd.to_datetime(panel['日期'], errors='coerce')
    if panel['_industry_date'].isna().any():
        raise ValueError('行业 as-of 关联包含无效日期')
    panel['_industry_key'] = _industry_stock_key(panel['股票代码'])
    history = _prepare_industry_history(industry_history)
    joined = pd.merge_asof(
        panel.sort_values(['_industry_date', '_industry_key'], kind='mergesort'),
        history.loc[:, ['effective_date', '_industry_key', 'industry']],
        left_on='_industry_date', right_on='effective_date', by='_industry_key',
        direction='backward', allow_exact_matches=True,
    ).sort_values('_industry_order', kind='mergesort')
    joined[industry_column] = joined['industry'].astype('string')
    return joined.drop(columns=[
        '_industry_order', '_industry_date', '_industry_key',
        'effective_date', 'industry',
    ])


def add_industry_residual_features(
    df, industry_history, min_industry_size=3,
    industry_column=INDUSTRY_ASOF_COLUMN,
):
    """添加12项当日市场/行业残差和百分位；缺失行业退化为市场统计。"""
    required = {'股票代码', '日期', '收盘'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'行业残差特征缺少基础列: {sorted(missing)}')
    if min_industry_size < 2:
        raise ValueError('min_industry_size 至少为2')
    panel = attach_industry_asof(df, industry_history, industry_column)
    panel['日期'] = pd.to_datetime(panel['日期'])
    panel['收盘'] = pd.to_numeric(panel['收盘'], errors='coerce')
    panel = panel.sort_values(['股票代码', '日期'], kind='mergesort')
    dates = panel['日期']
    closes = panel.groupby('股票代码', sort=False)['收盘']
    for period in (1, 3, 5):
        returns = closes.pct_change(periods=period, fill_method=None)
        market_mean = returns.groupby(dates).transform('mean')
        market_pct = returns.groupby(dates).rank(method='average', pct=True)
        group_keys = [dates, panel[industry_column]]
        industry_count = returns.notna().groupby(group_keys).transform('sum')
        valid = panel[industry_column].notna() & industry_count.ge(min_industry_size)
        industry_mean = returns.groupby(group_keys).transform('mean').where(valid, market_mean)
        industry_pct = returns.groupby(group_keys).rank(method='average', pct=True).where(valid, market_pct)
        panel[f'indresid_market_return_{period}'] = returns - market_mean
        panel[f'indresid_industry_return_{period}'] = returns - industry_mean
        panel[f'indresid_market_return_{period}_pct'] = market_pct
        panel[f'indresid_industry_return_{period}_pct'] = industry_pct
    panel[list(INDUSTRY_RESIDUAL_FEATURES)] = panel[
        list(INDUSTRY_RESIDUAL_FEATURES)
    ].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return panel.sort_index()


def add_industry_neutral_label(
    df, target_column='label', industry_column=INDUSTRY_ASOF_COLUMN,
    output_column=INDUSTRY_NEUTRAL_TARGET, min_industry_size=3,
):
    """用预测日所属行业同期均值去除五日收益；无行业时用市场均值。"""
    required = {'日期', target_column, industry_column}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'行业中性标签缺少必要列: {sorted(missing)}')
    dates = pd.to_datetime(df['日期'])
    targets = pd.to_numeric(df[target_column], errors='coerce')
    market_mean = targets.groupby(dates).transform('mean')
    group_keys = [dates, df[industry_column]]
    count = targets.notna().groupby(group_keys).transform('sum')
    valid = df[industry_column].notna() & count.ge(min_industry_size)
    industry_mean = targets.groupby(group_keys).transform('mean').where(valid, market_mean)
    result = df.copy()
    result[output_column] = (targets - industry_mean).replace([np.inf, -np.inf], np.nan)
    return result


def attach_label_end_dates(records, trading_dates, horizon=5):
    """为旧 OOF 产物补齐标签结束日，严格按市场交易日而非自然日推导。"""
    trading_dates = pd.DatetimeIndex(pd.to_datetime(trading_dates)).sort_values()
    trading_dates = trading_dates.unique()
    if horizon < 1:
        raise ValueError('标签期限必须大于0')
    positions = {
        pd.Timestamp(date): index
        for index, date in enumerate(trading_dates)
    }
    enriched = []
    for record in records:
        copied = dict(record)
        prediction_date = pd.Timestamp(record['prediction_date'])
        if 'label_end_date' in record:
            label_end_date = pd.Timestamp(record['label_end_date'])
        else:
            if prediction_date not in positions:
                raise ValueError(
                    f'OOF预测日不在交易日历中: {prediction_date.date()}'
                )
            end_position = positions[prediction_date] + int(horizon)
            if end_position >= len(trading_dates):
                raise ValueError(
                    f'无法为 {prediction_date.date()} 推导完整标签结束日'
                )
            label_end_date = pd.Timestamp(trading_dates[end_position])
        if label_end_date <= prediction_date:
            raise ValueError('标签结束日必须晚于预测日')
        copied['label_end_date'] = label_end_date.strftime('%Y-%m-%d')
        enriched.append(copied)
    return enriched


def _aggregate_probability_diagnostics(daily, head):
    """用整个 OOF 的常数发生率作为概率头基线，避免逐日标签泄漏。"""
    count = float(sum(row[f'{head}_count'] for row in daily))
    if count <= 0:
        raise ValueError(f'{head} 概率诊断没有有效样本')
    target_sum = float(sum(row[f'{head}_target_sum'] for row in daily))
    target_square_sum = float(sum(
        row[f'{head}_target_square_sum'] for row in daily
    ))
    brier_sum = float(sum(row[f'{head}_brier_sum'] for row in daily))
    event_rate = target_sum / count
    brier = brier_sum / count
    baseline_brier = max(
        target_square_sum / count - event_rate ** 2,
        0.0,
    )
    auc_weight = float(sum(row[f'{head}_auc_weight'] for row in daily))
    return {
        'event_rate': event_rate,
        'brier': brier,
        'baseline_brier': baseline_brier,
        'brier_skill': (
            float(1.0 - brier / baseline_brier)
            if baseline_brier > 1e-12 else 0.0
        ),
        'roc_auc': (
            float(sum(
                row[f'{head}_roc_auc'] * row[f'{head}_auc_weight']
                for row in daily
            ) / auc_weight)
            if auc_weight > 0 else 0.0
        ),
        'pr_auc': (
            float(sum(
                row[f'{head}_pr_auc'] * row[f'{head}_auc_weight']
                for row in daily
            ) / auc_weight)
            if auc_weight > 0 else event_rate
        ),
    }


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
    progress = tqdm(
        total=37,
        desc='相对/风险市场特征',
        unit='项',
        dynamic_ncols=True,
    )

    stock_codes = panel['股票代码']
    dates = panel['日期']
    close_by_stock = panel.groupby(stock_codes, sort=False)['收盘']
    volume_by_stock = panel.groupby(stock_codes, sort=False)['成交量']

    return_1 = close_by_stock.pct_change(periods=1, fill_method=None)
    progress.update()
    return_3 = close_by_stock.pct_change(periods=3, fill_method=None)
    progress.update()
    return_5 = close_by_stock.pct_change(periods=5, fill_method=None)
    progress.update()
    return_20 = close_by_stock.pct_change(periods=20, fill_method=None)
    progress.update()
    return_60 = close_by_stock.pct_change(periods=60, fill_method=None)
    progress.update()
    volatility_20 = return_1.groupby(
        stock_codes,
        sort=False,
    ).transform(lambda values: values.rolling(20, min_periods=20).std())
    progress.update()
    downside_squared = return_1.clip(upper=0.0).pow(2)
    downside_volatility_5 = downside_squared.groupby(
        stock_codes,
        sort=False,
    ).transform(
        lambda values: values.rolling(5, min_periods=5).mean().pow(0.5)
    )
    progress.update()
    downside_volatility_20 = downside_squared.groupby(
        stock_codes,
        sort=False,
    ).transform(
        lambda values: values.rolling(20, min_periods=20).mean().pow(0.5)
    )
    progress.update()
    volume_ma20 = volume_by_stock.transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    progress.update()
    close_ma20 = close_by_stock.transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    progress.update()
    close_ma60 = close_by_stock.transform(
        lambda values: values.rolling(60, min_periods=60).mean()
    )
    progress.update()
    close_high20 = close_by_stock.transform(
        lambda values: values.rolling(20, min_periods=20).max()
    )
    progress.update()
    volume_ratio_20 = panel['成交量'] / volume_ma20.replace(0.0, np.nan) - 1.0
    ma20_distance = panel['收盘'] / close_ma20.replace(0.0, np.nan) - 1.0
    ma60_distance = panel['收盘'] / close_ma60.replace(0.0, np.nan) - 1.0
    drawdown_20 = panel['收盘'] / close_high20.replace(0.0, np.nan) - 1.0
    momentum_gap_5_20 = return_5 - return_20
    momentum_gap_5_60 = return_5 - return_60
    progress.update()

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
        progress.update()

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
        progress.update()

    panel['market_return_1'] = return_1.groupby(dates).transform('mean')
    progress.update()
    panel['market_return_3'] = return_3.groupby(dates).transform('mean')
    progress.update()
    panel['market_return_5'] = return_5.groupby(dates).transform('mean')
    progress.update()
    panel['market_return_20'] = return_20.groupby(dates).transform('mean')
    progress.update()

    up_indicator = return_1.gt(0.0).astype(float).where(return_1.notna())
    above_ma20_indicator = (
        ma20_distance.gt(0.0).astype(float).where(ma20_distance.notna())
    )
    panel['market_breadth_up'] = up_indicator.groupby(dates).transform('mean')
    panel['market_breadth_above_ma20'] = (
        above_ma20_indicator.groupby(dates).transform('mean')
    )
    progress.update()

    market_return_20_mean = return_20.groupby(dates).transform('mean')
    panel['market_return_20_dispersion'] = (
        (return_20 - market_return_20_mean)
        .pow(2)
        .groupby(dates)
        .transform('mean')
        .pow(0.5)
    )
    progress.update()
    panel['market_downside_vol_5'] = (
        downside_volatility_5.groupby(dates).transform('mean')
    )
    panel['market_downside_vol_20'] = (
        downside_volatility_20.groupby(dates).transform('mean')
    )
    panel['market_drawdown_20'] = drawdown_20.groupby(dates).transform('mean')
    progress.update()
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
    progress.update()

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
    progress.update()

    generated_features = [*RELATIVE_MARKET_FEATURES, *RISK_MARKET_FEATURES]
    panel[generated_features] = (
        panel[generated_features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    progress.update()
    progress.close()
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
    df['volume_change'] = volume.pct_change(fill_method=None)
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


def project_long_only_weights(weights, min_weight=0.05, max_weight=0.35):
    """欧氏投影至和为一的 long-only 有界单纯形。"""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or weights.size == 0 or not np.isfinite(weights).all():
        raise ValueError('权重必须是有限的非空一维数组')
    if not (0.0 <= min_weight <= max_weight <= 1.0):
        raise ValueError('权重边界必须满足 0 <= min <= max <= 1')
    if weights.size * min_weight > 1.0 or weights.size * max_weight < 1.0:
        raise ValueError('权重边界与持仓数量不可行')
    lower = float(weights.min() - max_weight)
    upper = float(weights.max() - min_weight)
    for _ in range(80):
        offset = (lower + upper) / 2.0
        if np.clip(weights - offset, min_weight, max_weight).sum() > 1.0:
            lower = offset
        else:
            upper = offset
    projected = np.clip(weights - (lower + upper) / 2.0, min_weight, max_weight)
    projected[np.argmax(projected)] += 1.0 - projected.sum()
    return projected


def select_industry_capped_top_indices(
    scores, industries, top_k=5, max_stocks_per_industry=2, candidate_k=10,
):
    """仅在原始 Top-``candidate_k`` 中用行业上限替换名称；不可行则回退 Top-5。"""
    scores = np.asarray(scores, dtype=np.float64)
    industries = np.asarray(industries, dtype=object)
    if scores.ndim != 1 or industries.shape != scores.shape:
        raise ValueError('行业选择的分数和行业标签长度必须一致')
    if scores.size < top_k or max_stocks_per_industry < 1:
        raise ValueError('行业选择的股票数或行业上限无效')
    order = np.lexsort((np.arange(scores.size), -scores))
    raw_top = order[:top_k]
    selected, counts = [], {}
    for index in order[:min(max(int(candidate_k), top_k), scores.size)]:
        industry = industries[index]
        # 无行业快照不施加行业约束，避免快照缺失改变股票池可用性。
        constrained = industry is not None and not pd.isna(industry)
        key = str(industry) if constrained else None
        if constrained and counts.get(key, 0) >= max_stocks_per_industry:
            continue
        selected.append(int(index))
        if constrained:
            counts[key] = counts.get(key, 0) + 1
        if len(selected) == top_k:
            return {
                'top_indices': np.asarray(selected, dtype=np.int64),
                'raw_top_indices': raw_top,
                'applied': True,
                'fallback': False,
            }
    return {
        'top_indices': raw_top,
        'raw_top_indices': raw_top,
        'applied': False,
        'fallback': True,
    }


def build_exposure_confidence(scores, tail_probabilities, disagreement, market_pressure, top_k=5):
    """构造可校准的置信度：Top-5/6 间距减尾部风险、分歧和市场压力。"""
    scores = np.asarray(scores, dtype=np.float64)
    tail_probabilities = np.asarray(tail_probabilities, dtype=np.float64)
    if scores.ndim != 1 or tail_probabilities.shape != scores.shape:
        raise ValueError('置信度分数和尾部风险必须是一维且长度一致')
    if scores.size <= top_k or not (
        np.isfinite(scores).all() and np.isfinite(tail_probabilities).all()
    ):
        raise ValueError('置信度至少需要 Top-k+1 个有限股票分数')
    if disagreement < 0 or not np.isfinite(disagreement) or not np.isfinite(market_pressure):
        raise ValueError('分歧和市场压力必须为有限值，且分歧非负')
    ordered = np.sort(scores)[::-1]
    selected = np.argsort(scores)[::-1][:top_k]
    gap = float(ordered[top_k - 1] - ordered[top_k])
    tail_risk = float(tail_probabilities[selected].mean())
    return {
        'raw_confidence': gap - tail_risk - float(disagreement) - float(market_pressure),
        'top5_top6_gap': gap,
        'top5_tail_risk': tail_risk,
    }


def fit_strict_forward_confidence_calibrator(
    prediction_date, label_end_dates, confidence_values, utility_values,
    min_exposure=0.20, max_exposure=0.999999, min_samples=20,
):
    """仅用标签结束日早于 ``prediction_date`` 的 OOF 样本拟合单调仓位校准。"""
    from sklearn.isotonic import IsotonicRegression

    ends = pd.to_datetime(label_end_dates)
    confidence = np.asarray(confidence_values, dtype=np.float64)
    utility = np.asarray(utility_values, dtype=np.float64)
    if not (ends.size == confidence.size == utility.size):
        raise ValueError('置信度校准输入长度不一致')
    if not 0.0 <= min_exposure < max_exposure < 1.0:
        raise ValueError('置信度校准仓位范围无效')
    resolved = (ends < pd.Timestamp(prediction_date)) & np.isfinite(confidence) & np.isfinite(utility)
    if int(resolved.sum()) < min_samples:
        return None
    x = confidence[resolved]
    median = float(np.median(x))
    scale = float(np.quantile(x, .75) - np.quantile(x, .25))
    scale = max(scale, 1e-8)
    target = min_exposure + (max_exposure - min_exposure) * percentile_ranks(utility[resolved])
    model = IsotonicRegression(
        increasing=True, y_min=min_exposure, y_max=max_exposure,
        out_of_bounds='clip',
    ).fit((x - median) / scale, target)
    return {
        'model': model, 'median': median, 'scale': scale,
        'sample_count': int(resolved.sum()),
        'latest_label_end_date': pd.Timestamp(ends[resolved].max()).strftime('%Y-%m-%d'),
    }


def predict_calibrated_exposure(calibration, confidence):
    """应用 ``fit_strict_forward_confidence_calibrator`` 的已保存结果。"""
    if calibration is None:
        return None
    value = np.asarray([confidence], dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError('置信度必须为有限数值')
    return float(calibration['model'].predict(
        (value - calibration['median']) / calibration['scale']
    )[0])


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


def _multi_window_positive_correlation(return_history, lookbacks):
    """取多个严格因果窗口正相关的逐元素最大值。"""
    return_history = np.asarray(return_history, dtype=np.float64)
    lookbacks = tuple(sorted({int(value) for value in lookbacks}))
    if not lookbacks or lookbacks[0] < 2:
        raise ValueError('相关性窗口必须至少包含一个不小于2的整数')
    if lookbacks[-1] > return_history.shape[1]:
        raise ValueError('相关性窗口超过可用收益历史')
    return np.maximum.reduce([
        _positive_correlation_matrix(return_history[:, -lookback:])
        for lookback in lookbacks
    ])


def _candidate_correlation_clusters(
    correlations,
    candidate_indices,
    threshold,
):
    """按阈值图的连通分量构造确定性的相关簇。"""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError('相关簇阈值必须位于 [0, 1]')
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    cluster_by_index = np.full(correlations.shape[0], -1, dtype=np.int64)
    unvisited = set(int(index) for index in candidate_indices)
    cluster_id = 0
    while unvisited:
        start = min(unvisited)
        stack = [start]
        unvisited.remove(start)
        members = []
        while stack:
            current = stack.pop()
            members.append(current)
            neighbours = [
                index
                for index in sorted(unvisited)
                if correlations[current, index] >= threshold
            ]
            for index in neighbours:
                unvisited.remove(index)
                stack.append(index)
        cluster_by_index[members] = cluster_id
        cluster_id += 1
    return cluster_by_index


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
    correlation_lookbacks=None,
    cluster_cap_enabled=False,
    cluster_correlation_threshold=0.60,
    max_stocks_per_cluster=2,
    cluster_max_raw_rank=None,
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
    requested_candidate_k = min(int(candidate_k), scores.size)
    if requested_candidate_k < top_k:
        raise ValueError('selection_candidate_k 不能小于 Top-k')
    if max_stocks_per_cluster < 1:
        raise ValueError('max_stocks_per_cluster 必须大于0')
    if not (
        np.isfinite(scores).all()
        and np.isfinite(momentum_percentiles).all()
        and np.isfinite(return_history).all()
    ):
        raise ValueError('风险选择输入包含 NaN 或无穷值')

    raw_order = np.lexsort((np.arange(scores.size), -scores))
    raw_top_indices = raw_order[:top_k]
    cs5, cs20, cs60 = momentum_percentiles.T
    reversal_risk = (
        0.6 * np.maximum(cs60 - cs5, 0.0)
        + 0.4 * np.maximum(cs20 - cs5, 0.0)
    )
    if correlation_lookbacks is None:
        correlation_lookbacks = (return_history.shape[1],)
    positive_correlations = _multi_window_positive_correlation(
        return_history,
        correlation_lookbacks,
    )
    bounded_cluster_mode = (
        cluster_cap_enabled and cluster_max_raw_rank is not None
    )
    if bounded_cluster_mode:
        cluster_max_raw_rank = min(int(cluster_max_raw_rank), scores.size)
        if cluster_max_raw_rank < top_k:
            raise ValueError('cluster_max_raw_rank 不能小于 Top-k')
        effective_candidate_k = cluster_max_raw_rank
    else:
        effective_candidate_k = requested_candidate_k
    cluster_constraint_skipped = False
    while True:
        candidate_indices = raw_order[:effective_candidate_k]
        cluster_by_index = _candidate_correlation_clusters(
            positive_correlations,
            candidate_indices,
            cluster_correlation_threshold,
        )
        _, cluster_sizes = np.unique(
            cluster_by_index[candidate_indices],
            return_counts=True,
        )
        cluster_capacity = int(np.minimum(
            cluster_sizes,
            max_stocks_per_cluster,
        ).sum())
        if not cluster_cap_enabled or cluster_capacity >= top_k:
            break
        if bounded_cluster_mode:
            cluster_constraint_skipped = True
            break
        if effective_candidate_k >= scores.size:
            raise ValueError(
                '即使扩展到全部股票，相关簇硬约束仍无法选满Top-k；'
                '数据中可用相关簇数量不足'
            )
        effective_candidate_k = min(
            scores.size,
            max(
                effective_candidate_k + 10,
                int(np.ceil(effective_candidate_k * 1.25)),
            ),
        )

    cluster_constraint_applied = bool(
        cluster_cap_enabled and not cluster_constraint_skipped
    )
    if cluster_constraint_skipped:
        selected = raw_top_indices.tolist()
        selected_correlation_risks = [
            0.0 if position == 0 else float(
                positive_correlations[index, selected[:position]].mean()
            )
            for position, index in enumerate(selected)
        ]
    elif risk_gamma == 0.0 and not cluster_cap_enabled:
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
        selected_cluster_counts = {}
        remaining = set(int(index) for index in candidate_indices)
        while len(selected) < top_k:
            best_index = None
            best_utility = -float('inf')
            best_correlation_risk = 0.0
            for index in sorted(remaining):
                cluster_id = int(cluster_by_index[index])
                if (
                    cluster_constraint_applied
                    and selected_cluster_counts.get(cluster_id, 0)
                    >= max_stocks_per_cluster
                ):
                    continue
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
            if best_index is None:
                raise ValueError(
                    '相关簇硬约束下候选池不足，无法选满Top-k；'
                    '请扩大候选池或调整聚类阈值'
                )
            selected.append(best_index)
            selected_correlation_risks.append(best_correlation_risk)
            best_cluster = int(cluster_by_index[best_index])
            selected_cluster_counts[best_cluster] = (
                selected_cluster_counts.get(best_cluster, 0) + 1
            )
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
        'cluster_ids': cluster_by_index[selected],
        'raw_cluster_ids': cluster_by_index[raw_top_indices],
        'num_candidate_clusters': int(
            np.unique(cluster_by_index[candidate_indices]).size
        ),
        'requested_candidate_k': int(requested_candidate_k),
        'effective_candidate_k': int(effective_candidate_k),
        'candidate_pool_expanded': bool(
            effective_candidate_k > requested_candidate_k
        ),
        'cluster_cap_enabled': bool(cluster_cap_enabled),
        'cluster_constraint_applied': cluster_constraint_applied,
        'cluster_constraint_skipped': bool(cluster_constraint_skipped),
        'cluster_max_raw_rank': (
            int(cluster_max_raw_rank)
            if cluster_max_raw_rank is not None
            else None
        ),
        'max_selected_raw_rank': int(raw_rank_by_index[selected].max()),
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
    correlation_lookbacks=(20,),
    cluster_cap_enabled=False,
    cluster_correlation_threshold=0.60,
    max_stocks_per_cluster=2,
    cluster_max_raw_rank=None,
    risk_probability_matrix=None,
    regime_gates=None,
    risk_score_penalty=0.0,
    correlation_exposure_gamma=0.0,
    exposure_head_blend=1.0,
    fixed_exposure_baseline=0.6231689453125,
    industry_labels=None,
    max_stocks_per_industry=None,
    industry_candidate_k=10,
    position_weight_bounds=None,
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
    if max_stocks_per_industry is not None:
        if industry_labels is None:
            raise ValueError('行业上限需要提供行业标签')
        if int(max_stocks_per_industry) < 1:
            raise ValueError('max_stocks_per_industry 必须大于0')

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
        if selection_risk_gamma != 0.0 or cluster_cap_enabled:
            raise ValueError('风险惩罚或相关簇约束需要风险上下文')
        raw_top_indices = np.lexsort(
            (np.arange(num_stocks), -ensemble_scores)
        )[:top_k]
        risk_selection = {
            'top_indices': raw_top_indices,
            'raw_top_indices': raw_top_indices,
            'selected_raw_ranks': np.arange(1, top_k + 1),
            'reversal_risk': np.zeros(top_k, dtype=np.float64),
            'correlation_risk': np.zeros(top_k, dtype=np.float64),
            'cluster_ids': np.arange(top_k, dtype=np.int64),
            'raw_cluster_ids': np.arange(top_k, dtype=np.int64),
            'num_candidate_clusters': top_k,
            'requested_candidate_k': int(selection_candidate_k),
            'effective_candidate_k': int(selection_candidate_k),
            'candidate_pool_expanded': False,
            'cluster_cap_enabled': False,
            'cluster_constraint_applied': False,
            'cluster_constraint_skipped': False,
            'cluster_max_raw_rank': None,
            'max_selected_raw_rank': int(top_k),
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
            correlation_lookbacks=correlation_lookbacks,
            cluster_cap_enabled=cluster_cap_enabled,
            cluster_correlation_threshold=cluster_correlation_threshold,
            max_stocks_per_cluster=max_stocks_per_cluster,
            cluster_max_raw_rank=cluster_max_raw_rank,
            top_k=top_k,
        )
    top_indices = risk_selection['top_indices']
    industry_selection = {
        'raw_top_indices': top_indices,
        'applied': False,
        'fallback': False,
    }
    if max_stocks_per_industry is not None:
        industry_selection = select_industry_capped_top_indices(
            ensemble_scores,
            industry_labels,
            top_k=top_k,
            max_stocks_per_industry=int(max_stocks_per_industry),
            candidate_k=industry_candidate_k,
        )
        top_indices = industry_selection['top_indices']
        score_order = np.lexsort((np.arange(num_stocks), -ensemble_scores))
        raw_rank = np.empty(num_stocks, dtype=np.int64)
        raw_rank[score_order] = np.arange(1, num_stocks + 1)
        risk_selection = dict(risk_selection)
        risk_selection['selected_raw_ranks'] = raw_rank[top_indices]
        risk_selection['max_selected_raw_rank'] = int(raw_rank[top_indices].max())

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
    if position_weight_bounds is not None:
        if len(position_weight_bounds) != 2:
            raise ValueError('position_weight_bounds 必须为 (min_weight, max_weight)')
        relative_weights = project_long_only_weights(
            relative_weights,
            min_weight=float(position_weight_bounds[0]),
            max_weight=float(position_weight_bounds[1]),
        )

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
        'selected_cluster_ids': risk_selection['cluster_ids'],
        'raw_cluster_ids': risk_selection['raw_cluster_ids'],
        'num_candidate_clusters': risk_selection[
            'num_candidate_clusters'
        ],
        'requested_candidate_k': risk_selection[
            'requested_candidate_k'
        ],
        'effective_candidate_k': risk_selection[
            'effective_candidate_k'
        ],
        'candidate_pool_expanded': risk_selection[
            'candidate_pool_expanded'
        ],
        'cluster_cap_enabled': risk_selection['cluster_cap_enabled'],
        'cluster_constraint_applied': risk_selection[
            'cluster_constraint_applied'
        ],
        'cluster_constraint_skipped': risk_selection[
            'cluster_constraint_skipped'
        ],
        'cluster_max_raw_rank': risk_selection['cluster_max_raw_rank'],
        'max_selected_raw_rank': risk_selection['max_selected_raw_rank'],
        'industry_constraint_applied': industry_selection['applied'],
        'industry_constraint_fallback': industry_selection['fallback'],
        'max_stocks_per_industry': (
            None if max_stocks_per_industry is None
            else int(max_stocks_per_industry)
        ),
        'industry_candidate_k': int(industry_candidate_k),
        'position_weight_bounds': (
            None if position_weight_bounds is None
            else [float(value) for value in position_weight_bounds]
        ),
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
        if 'label_end_date' not in reference:
            raise ValueError(
                f'Fold {fold} 在 {prediction_date} 缺少 label_end_date'
            )
        label_end_date = reference['label_end_date']
        reference_stocks = np.asarray(reference['stock_indices'], dtype=np.int64)
        has_risk_context = all(
            key in reference
            for key in ('momentum_percentiles', 'return_history')
        )
        for records in sorted_records[1:]:
            if records[day_idx].get('label_end_date') != label_end_date:
                raise ValueError(
                    f'Fold {fold} 在 {prediction_date} 的标签结束日不一致'
                )
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
                'tail_5d_targets',
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
            'label_end_date': label_end_date,
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
            'tail_5d_probabilities': np.stack([
                np.asarray(
                    records[day_idx].get(
                        'tail_5d_probabilities',
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
            'tail_5d_targets': np.asarray(
                reference.get(
                    'tail_5d_targets',
                    np.zeros(reference_stocks.size),
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
    tail_5d_blend=0.0,
    correlation_exposure_gamma=0.0,
    exposure_head_blend=1.0,
    correlation_lookbacks=(20,),
    cluster_cap_enabled=False,
    cluster_correlation_threshold=0.60,
    max_stocks_per_cluster=2,
    cluster_max_raw_rank=None,
    tail_5d_threshold=-0.03,
    fixed_exposure_baseline=0.6231689453125,
    downside_weight=0.5,
    top_k=5,
    include_daily=False,
):
    """计算 OOF ensemble 的收益分解、下行风险和 Rank IC。"""
    def probability_diagnostics(predictions, targets, binary=False):
        predictions = np.asarray(predictions, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        event_rate = float(targets.mean())
        brier = float(np.mean((predictions - targets) ** 2))
        baseline_brier = float(np.mean((targets - event_rate) ** 2))
        result = {
            'count': int(targets.size),
            'target_sum': float(targets.sum()),
            'target_square_sum': float(np.square(targets).sum()),
            'brier_sum': float(np.square(predictions - targets).sum()),
            'event_rate': event_rate,
            'brier': brier,
            'baseline_brier': baseline_brier,
            'brier_skill': (
                float(1.0 - brier / baseline_brier)
                if baseline_brier > 1e-12 else 0.0
            ),
            'roc_auc': 0.0,
            'pr_auc': event_rate,
            'auc_weight': 0,
        }
        if binary and np.unique(targets).size == 2:
            result['roc_auc'] = float(
                roc_auc_score(targets, predictions)
            )
            result['pr_auc'] = float(
                average_precision_score(targets, predictions)
            )
            result['auc_weight'] = int(targets.size)
        return result

    risk_blends = np.asarray(
        [
            risk_1d_blend,
            risk_3d_blend,
            risk_5d_blend,
            tail_5d_blend,
        ],
        dtype=np.float64,
    )
    if (risk_blends < 0).any() or risk_blends.sum() <= 0:
        raise ValueError('风险头混合权重必须非负且权重和大于0')
    risk_blends /= risk_blends.sum()
    (
        risk_1d_blend,
        risk_3d_blend,
        risk_5d_blend,
        tail_5d_blend,
    ) = risk_blends
    daily = []
    for day in ensemble_days:
        combined_risk_probabilities = (
            risk_1d_blend * day['risk_1d_probabilities']
            + risk_3d_blend * day['risk_3d_probabilities']
            + risk_5d_blend * day['risk_5d_probabilities']
            + tail_5d_blend * day['tail_5d_probabilities']
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
            correlation_lookbacks=correlation_lookbacks,
            cluster_cap_enabled=cluster_cap_enabled,
            cluster_correlation_threshold=cluster_correlation_threshold,
            max_stocks_per_cluster=max_stocks_per_cluster,
            cluster_max_raw_rank=cluster_max_raw_rank,
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
        selected_tail_5d = day['tail_5d_probabilities'][:, selected].mean()
        selected_combined_risk = combined_risk_probabilities[
            :, selected
        ].mean()
        mean_risk_1d_prediction = day['risk_1d_probabilities'].mean(axis=0)
        mean_risk_3d_prediction = day['risk_3d_probabilities'].mean(axis=0)
        mean_risk_5d_prediction = day['risk_5d_probabilities'].mean(axis=0)
        mean_tail_5d_prediction = day['tail_5d_probabilities'].mean(axis=0)
        risk_diagnostics = {
            'risk_1d': probability_diagnostics(
                mean_risk_1d_prediction,
                day['risk_1d_targets'],
            ),
            'risk_3d': probability_diagnostics(
                mean_risk_3d_prediction,
                day['risk_3d_targets'],
            ),
            'risk_5d': probability_diagnostics(
                mean_risk_5d_prediction,
                day['risk_5d_targets'],
            ),
            'tail_5d': probability_diagnostics(
                mean_tail_5d_prediction,
                day['tail_5d_targets'],
                binary=True,
            ),
        }
        regime_prediction = float(np.median(day['regime_gates']))
        equal_full_return = float(selected_returns.mean())
        raw_top5_return = float(
            day['targets'][portfolio['raw_top_indices']].mean()
        )
        market_future_return = float(day['targets'].mean())
        market_tail_share = float(
            np.mean(day['targets'] <= tail_5d_threshold)
        )
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
            'label_end_date': day['label_end_date'],
            'top5_return': equal_full_return,
            'selected_stock_indices': [
                int(value) for value in day['stock_indices'][selected]
            ],
            'raw_top5_return': raw_top5_return,
            'diversification_return_contribution': (
                equal_full_return - raw_top5_return
            ),
            'market_future_return': market_future_return,
            'market_tail_share': market_tail_share,
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
            'exposure_policy_contribution': (
                weighted_return - fixed_exposure_return
            ),
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
            'selected_tail_5d': float(selected_tail_5d),
            'selected_combined_risk': float(selected_combined_risk),
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
            'tail_5d_brier': float(np.mean(
                (
                    mean_tail_5d_prediction
                    - day['tail_5d_targets']
                ) ** 2
            )),
            'regime_brier': float(
                (regime_prediction - day['regime_target']) ** 2
            ),
            **{
                f'{head}_{metric}': float(value)
                for head, diagnostics in risk_diagnostics.items()
                for metric, value in diagnostics.items()
            },
            'mean_positive_correlation': portfolio[
                'mean_positive_correlation'
            ],
            'raw_mean_positive_correlation': portfolio[
                'raw_mean_positive_correlation'
            ],
            'mean_reversal_risk': float(
                np.mean(portfolio['selected_reversal_risk'])
            ),
            'selected_cluster_ids': [
                int(value) for value in portfolio['selected_cluster_ids']
            ],
            'raw_cluster_ids': [
                int(value) for value in portfolio['raw_cluster_ids']
            ],
            'num_candidate_clusters': int(
                portfolio['num_candidate_clusters']
            ),
            'requested_candidate_k': int(
                portfolio['requested_candidate_k']
            ),
            'effective_candidate_k': int(
                portfolio['effective_candidate_k']
            ),
            'candidate_pool_expanded': bool(
                portfolio['candidate_pool_expanded']
            ),
            'cluster_constraint_applied': bool(
                portfolio['cluster_constraint_applied']
            ),
            'cluster_constraint_skipped': bool(
                portfolio['cluster_constraint_skipped']
            ),
            'max_selected_raw_rank': int(
                portfolio['max_selected_raw_rank']
            ),
            'max_selected_cluster_count': int(max(
                np.unique(
                    portfolio['selected_cluster_ids'],
                    return_counts=True,
                )[1]
            )),
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
    top5_negative_returns = np.minimum(top5_returns, 0.0)
    top5_downside_deviation = float(np.sqrt(
        np.mean(top5_negative_returns ** 2)
    ))
    market_future_returns = np.asarray([
        row['market_future_return'] for row in daily
    ], dtype=np.float64)
    market_tail_shares = np.asarray([
        row['market_tail_share'] for row in daily
    ], dtype=np.float64)
    selected_tail_risks = np.asarray([
        row['selected_tail_5d'] for row in daily
    ], dtype=np.float64)
    selected_combined_risks = np.asarray([
        row['selected_combined_risk'] for row in daily
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

    def safe_spearman(left, right):
        if left.std() < 1e-12 or right.std() < 1e-12:
            return 0.0
        value = spearmanr(left, right).statistic
        return float(value) if np.isfinite(value) else 0.0

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
        'mean_raw_top5_return': mean('raw_top5_return'),
        'mean_diversification_return_contribution': mean(
            'diversification_return_contribution'
        ),
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
        'top5_downside_deviation': top5_downside_deviation,
        'mean_rank_ic': mean('rank_ic'),
        'worst_daily_rank_ic': float(min(row['rank_ic'] for row in daily)),
        # 兼容旧报告读取器；新代码应使用语义明确的字段。
        'worst_rank_ic': float(min(row['rank_ic'] for row in daily)),
        'worst_fold_mean_rank_ic': float(min(
            row['mean_rank_ic'] for row in fold_summaries
        )),
        'mean_gross_exposure': mean('gross_exposure'),
        'mean_head_gross_exposure': mean('head_gross_exposure'),
        'mean_cash_weight': mean('cash_weight'),
        'mean_model_disagreement': mean('model_disagreement'),
        'mean_regime_gate': mean('regime_gate'),
        'regime_gate_std': float(regime_gates.std()),
        'mean_selected_risk_1d': mean('selected_risk_1d'),
        'mean_selected_risk_3d': mean('selected_risk_3d'),
        'mean_selected_risk_5d': mean('selected_risk_5d'),
        'mean_selected_tail_5d': mean('selected_tail_5d'),
        'mean_selected_combined_risk': mean('selected_combined_risk'),
        'mean_risk_1d_brier': mean('risk_1d_brier'),
        'mean_risk_3d_brier': mean('risk_3d_brier'),
        'mean_risk_5d_brier': mean('risk_5d_brier'),
        'mean_tail_5d_brier': mean('tail_5d_brier'),
        'mean_regime_brier': mean('regime_brier'),
        'regime_return_spearman': float(
            regime_return_correlation
            if np.isfinite(regime_return_correlation)
            else 0.0
        ),
        'regime_market_return_spearman': safe_spearman(
            regime_gates,
            market_future_returns,
        ),
        'regime_tail_share_spearman': safe_spearman(
            regime_gates,
            market_tail_shares,
        ),
        'tail_risk_return_spearman': safe_spearman(
            selected_tail_risks,
            top5_returns,
        ),
        'combined_risk_return_spearman': safe_spearman(
            selected_combined_risks,
            top5_returns,
        ),
        'mean_allocation_contribution': mean('allocation_contribution'),
        'mean_allocation_at_exposure_contribution': mean(
            'allocation_at_exposure_contribution'
        ),
        'mean_exposure_contribution': mean('exposure_contribution'),
        'mean_exposure_policy_contribution': mean(
            'exposure_policy_contribution'
        ),
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
        'mean_candidate_clusters': mean('num_candidate_clusters'),
        'mean_effective_candidate_k': mean('effective_candidate_k'),
        'candidate_pool_expansion_rate': float(np.mean([
            row['candidate_pool_expanded'] for row in daily
        ])),
        'max_effective_candidate_k': int(max(
            row['effective_candidate_k'] for row in daily
        )),
        'max_selected_cluster_count': int(max(
            row['max_selected_cluster_count'] for row in daily
        )),
        'cluster_constraint_application_rate': float(np.mean([
            row['cluster_constraint_applied'] for row in daily
        ])),
        'cluster_constraint_skip_rate': float(np.mean([
            row['cluster_constraint_skipped'] for row in daily
        ])),
        'max_selected_raw_rank': int(max(
            row['max_selected_raw_rank'] for row in daily
        )),
        'cluster_cap_enabled': bool(cluster_cap_enabled),
        'cluster_correlation_threshold': float(
            cluster_correlation_threshold
        ),
        'max_stocks_per_cluster': int(max_stocks_per_cluster),
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
        'ranking_policy_objective': float(
            top5_returns.mean()
            - downside_weight * top5_downside_deviation
        ),
        'fixed_exposure_policy_objective': float(
            fixed_exposure_returns.mean()
            - downside_weight * fixed_downside_deviation
        ),
        'exposure_policy_objective_delta': float(
            weighted_returns.mean()
            - downside_weight * downside_deviation
            - (
                fixed_exposure_returns.mean()
                - downside_weight * fixed_downside_deviation
            )
        ),
        'folds': fold_summaries,
    }
    for head in ('risk_1d', 'risk_3d', 'risk_5d', 'tail_5d'):
        diagnostics = _aggregate_probability_diagnostics(daily, head)
        for metric, value in diagnostics.items():
            summary[f'mean_{head}_{metric}'] = value
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
    tail_5d_blend=0.0,
    correlation_exposure_gamma_grid=(0.0,),
    exposure_head_blend_grid=(1.0,),
    selection_candidate_k=20,
    correlation_lookbacks=(20,),
    cluster_cap_enabled=False,
    cluster_correlation_threshold=0.60,
    max_stocks_per_cluster=2,
    tail_5d_threshold=-0.03,
    fixed_exposure_baseline=0.6231689453125,
    downside_weight=0.5,
    top_k=5,
):
    """仅用 OOF 收益网格选择 allocation 混合与分歧降仓强度。"""
    allocation_blend_grid = tuple(allocation_blend_grid)
    disagreement_gamma_grid = tuple(disagreement_gamma_grid)
    selection_risk_gamma_grid = tuple(selection_risk_gamma_grid)
    risk_score_penalty_grid = tuple(risk_score_penalty_grid)
    correlation_exposure_gamma_grid = tuple(
        correlation_exposure_gamma_grid
    )
    exposure_head_blend_grid = tuple(exposure_head_blend_grid)
    grid_size = int(np.prod([
        len(allocation_blend_grid),
        len(disagreement_gamma_grid),
        len(selection_risk_gamma_grid),
        len(risk_score_penalty_grid),
        len(correlation_exposure_gamma_grid),
        len(exposure_head_blend_grid),
    ]))
    calibration_progress = tqdm(
        total=grid_size,
        desc=f'OOF策略校准({len(ensemble_days)}日)',
        unit='组',
        dynamic_ncols=True,
    )
    risk_blends = np.asarray(
        [
            risk_1d_blend,
            risk_3d_blend,
            risk_5d_blend,
            tail_5d_blend,
        ],
        dtype=np.float64,
    )
    if (risk_blends < 0).any() or risk_blends.sum() <= 0:
        raise ValueError('风险头混合权重必须非负且权重和大于0')
    risk_blends /= risk_blends.sum()
    (
        risk_1d_blend,
        risk_3d_blend,
        risk_5d_blend,
        tail_5d_blend,
    ) = risk_blends
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
                                tail_5d_blend=tail_5d_blend,
                                correlation_exposure_gamma=float(
                                    correlation_exposure_gamma
                                ),
                                exposure_head_blend=float(
                                    exposure_head_blend
                                ),
                                correlation_lookbacks=correlation_lookbacks,
                                cluster_cap_enabled=cluster_cap_enabled,
                                cluster_correlation_threshold=(
                                    cluster_correlation_threshold
                                ),
                                max_stocks_per_cluster=(
                                    max_stocks_per_cluster
                                ),
                                tail_5d_threshold=tail_5d_threshold,
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
                            calibration_progress.update()
    calibration_progress.close()
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
        tail_5d_blend=tail_5d_blend,
        correlation_exposure_gamma=best[
            'correlation_exposure_gamma'
        ],
        exposure_head_blend=best['exposure_head_blend'],
        correlation_lookbacks=correlation_lookbacks,
        cluster_cap_enabled=cluster_cap_enabled,
        cluster_correlation_threshold=cluster_correlation_threshold,
        max_stocks_per_cluster=max_stocks_per_cluster,
        tail_5d_threshold=tail_5d_threshold,
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
        'tail_5d_blend': float(tail_5d_blend),
        'correlation_exposure_gamma': best[
            'correlation_exposure_gamma'
        ],
        'exposure_head_blend': best['exposure_head_blend'],
        'selection_candidate_k': int(selection_candidate_k),
        'correlation_lookbacks': [
            int(value) for value in correlation_lookbacks
        ],
        'cluster_cap_enabled': bool(cluster_cap_enabled),
        'cluster_correlation_threshold': float(
            cluster_correlation_threshold
        ),
        'max_stocks_per_cluster': int(max_stocks_per_cluster),
        'tail_5d_threshold': float(tail_5d_threshold),
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


def evaluate_ensemble_policy(ensemble_days, policy, include_daily=False):
    """使用已经选定的策略评估一组未参与调参的 OOF 日期。"""
    return summarize_ensemble_days(
        ensemble_days,
        min_exposure=policy['min_exposure'],
        max_exposure=policy['max_exposure'],
        allocation_temperature=policy['allocation_temperature'],
        allocation_blend=policy['allocation_blend'],
        disagreement_gamma=policy['disagreement_gamma'],
        selection_risk_gamma=policy['selection_risk_gamma'],
        selection_candidate_k=policy['selection_candidate_k'],
        risk_score_penalty=policy['risk_score_penalty'],
        risk_1d_blend=policy['risk_1d_blend'],
        risk_3d_blend=policy['risk_3d_blend'],
        risk_5d_blend=policy['risk_5d_blend'],
        tail_5d_blend=policy.get('tail_5d_blend', 0.0),
        correlation_exposure_gamma=policy[
            'correlation_exposure_gamma'
        ],
        exposure_head_blend=policy['exposure_head_blend'],
        correlation_lookbacks=policy.get(
            'correlation_lookbacks',
            [20],
        ),
        cluster_cap_enabled=policy.get('cluster_cap_enabled', False),
        cluster_correlation_threshold=policy.get(
            'cluster_correlation_threshold',
            0.60,
        ),
        max_stocks_per_cluster=policy.get(
            'max_stocks_per_cluster',
            2,
        ),
        cluster_max_raw_rank=policy.get('cluster_max_raw_rank'),
        tail_5d_threshold=policy.get('tail_5d_threshold', -0.03),
        fixed_exposure_baseline=policy['fixed_exposure_baseline'],
        downside_weight=policy['downside_weight'],
        top_k=policy['top_k'],
        include_daily=include_daily,
    )


def _summarize_cross_fitted_daily(daily, downside_weight):
    """汇总每折用独立策略得到的留出日记录。"""
    if not daily:
        raise ValueError('嵌套 OOF 没有留出评估记录')

    def values(key):
        return np.asarray([row[key] for row in daily], dtype=np.float64)

    def mean(key):
        return float(values(key).mean())

    def safe_spearman(left_key, right_key):
        left = values(left_key)
        right = values(right_key)
        if left.std() < 1e-12 or right.std() < 1e-12:
            return 0.0
        value = spearmanr(left, right).statistic
        return float(value) if np.isfinite(value) else 0.0

    weighted_returns = values('weighted_portfolio_return')
    negative_returns = np.minimum(weighted_returns, 0.0)
    downside_deviation = float(np.sqrt(np.mean(negative_returns ** 2)))
    top5_returns = values('top5_return')
    top5_downside = float(np.sqrt(np.mean(
        np.minimum(top5_returns, 0.0) ** 2
    )))
    fixed_returns = values('fixed_exposure_return')
    fixed_downside = float(np.sqrt(np.mean(
        np.minimum(fixed_returns, 0.0) ** 2
    )))
    fold_rows = {}
    for row in daily:
        fold_rows.setdefault(int(row['fold']), []).append(row)
    folds = [{
        'fold': fold,
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
        'positive_rate': float(np.mean([
            row['weighted_portfolio_return'] > 0.0 for row in rows
        ])),
        'num_evaluation_dates': len(rows),
    } for fold, rows in sorted(fold_rows.items())]
    scalar_mean_keys = (
        'top5_return',
        'raw_top5_return',
        'diversification_return_contribution',
        'equal_weight_at_exposure_return',
        'allocation_only_return',
        'allocation_contribution',
        'allocation_at_exposure_contribution',
        'exposure_contribution',
        'exposure_policy_contribution',
        'gross_exposure',
        'head_gross_exposure',
        'cash_weight',
        'model_disagreement',
        'regime_gate',
        'selected_risk_1d',
        'selected_risk_3d',
        'selected_risk_5d',
        'selected_tail_5d',
        'selected_combined_risk',
        'risk_1d_brier',
        'risk_3d_brier',
        'risk_5d_brier',
        'tail_5d_brier',
        'regime_brier',
        'risk_1d_event_rate',
        'risk_1d_baseline_brier',
        'risk_1d_brier_skill',
        'risk_1d_roc_auc',
        'risk_1d_pr_auc',
        'risk_3d_event_rate',
        'risk_3d_baseline_brier',
        'risk_3d_brier_skill',
        'risk_3d_roc_auc',
        'risk_3d_pr_auc',
        'risk_5d_event_rate',
        'risk_5d_baseline_brier',
        'risk_5d_brier_skill',
        'risk_5d_roc_auc',
        'risk_5d_pr_auc',
        'tail_5d_event_rate',
        'tail_5d_baseline_brier',
        'tail_5d_brier_skill',
        'tail_5d_roc_auc',
        'tail_5d_pr_auc',
        'mean_positive_correlation',
        'raw_mean_positive_correlation',
        'mean_reversal_risk',
        'num_candidate_clusters',
        'effective_candidate_k',
        'rank_ic',
    )
    summary = {
        f'mean_{key}': mean(key)
        for key in scalar_mean_keys
    }
    summary.update({
        'num_evaluation_dates': len(daily),
        'mean_weighted_portfolio_return': float(
            weighted_returns.mean()
        ),
        'worst_weighted_portfolio_return': float(
            weighted_returns.min()
        ),
        'p10_weighted_portfolio_return': float(
            np.quantile(weighted_returns, 0.10)
        ),
        'std_weighted_portfolio_return': float(
            weighted_returns.std()
        ),
        'positive_rate': float(np.mean(weighted_returns > 0.0)),
        'downside_deviation': downside_deviation,
        'top5_downside_deviation': top5_downside,
        'worst_daily_rank_ic': float(values('rank_ic').min()),
        'worst_rank_ic': float(values('rank_ic').min()),
        'worst_fold_mean_rank_ic': float(min(
            row['mean_rank_ic'] for row in folds
        )),
        'worst_fold_weighted_portfolio_return': float(min(
            row['mean_weighted_portfolio_return'] for row in folds
        )),
        'worst_fold_top5_return': float(min(
            row['mean_top5_return'] for row in folds
        )),
        'exposure_std': float(values('gross_exposure').std()),
        'regime_gate_std': float(values('regime_gate').std()),
        'regime_return_spearman': safe_spearman(
            'regime_gate',
            'top5_return',
        ),
        'regime_market_return_spearman': safe_spearman(
            'regime_gate',
            'market_future_return',
        ),
        'regime_tail_share_spearman': safe_spearman(
            'regime_gate',
            'market_tail_share',
        ),
        'exposure_return_spearman': safe_spearman(
            'gross_exposure',
            'top5_return',
        ),
        'tail_risk_return_spearman': safe_spearman(
            'selected_tail_5d',
            'top5_return',
        ),
        'combined_risk_return_spearman': safe_spearman(
            'selected_combined_risk',
            'top5_return',
        ),
        'max_selected_cluster_count': int(max(
            row['max_selected_cluster_count'] for row in daily
        )),
        'cluster_constraint_application_rate': float(np.mean([
            row.get('cluster_constraint_applied', False) for row in daily
        ])),
        'cluster_constraint_skip_rate': float(np.mean([
            row.get('cluster_constraint_skipped', False) for row in daily
        ])),
        'max_selected_raw_rank': int(max(
            row.get('max_selected_raw_rank', 5) for row in daily
        )),
        'policy_objective': float(
            weighted_returns.mean()
            - downside_weight * downside_deviation
        ),
        'ranking_policy_objective': float(
            top5_returns.mean() - downside_weight * top5_downside
        ),
        'fixed_exposure_policy_objective': float(
            fixed_returns.mean() - downside_weight * fixed_downside
        ),
        'exposure_policy_objective_delta': float(
            weighted_returns.mean()
            - downside_weight * downside_deviation
            - (
                fixed_returns.mean()
                - downside_weight * fixed_downside
            )
        ),
        'folds': folds,
        'daily': daily,
    })
    for head in ('risk_1d', 'risk_3d', 'risk_5d', 'tail_5d'):
        diagnostics = _aggregate_probability_diagnostics(daily, head)
        for metric, value in diagnostics.items():
            summary[f'mean_{head}_{metric}'] = value
    # 与既有报告字段兼容。
    aliases = {
        'mean_top5_return': 'mean_top5_return',
        'mean_raw_top5_return': 'mean_raw_top5_return',
        'mean_rank_ic': 'mean_rank_ic',
        'mean_gross_exposure': 'mean_gross_exposure',
        'mean_head_gross_exposure': 'mean_head_gross_exposure',
        'mean_cash_weight': 'mean_cash_weight',
        'mean_model_disagreement': 'mean_model_disagreement',
        'mean_regime_gate': 'mean_regime_gate',
        'mean_selected_risk_1d': 'mean_selected_risk_1d',
        'mean_selected_risk_3d': 'mean_selected_risk_3d',
        'mean_selected_risk_5d': 'mean_selected_risk_5d',
        'mean_selected_tail_5d': 'mean_selected_tail_5d',
        'mean_selected_combined_risk': (
            'mean_selected_combined_risk'
        ),
        'mean_risk_1d_brier': 'mean_risk_1d_brier',
        'mean_risk_3d_brier': 'mean_risk_3d_brier',
        'mean_risk_5d_brier': 'mean_risk_5d_brier',
        'mean_tail_5d_brier': 'mean_tail_5d_brier',
        'mean_regime_brier': 'mean_regime_brier',
        'mean_allocation_contribution': (
            'mean_allocation_contribution'
        ),
        'mean_allocation_at_exposure_contribution': (
            'mean_allocation_at_exposure_contribution'
        ),
        'mean_exposure_contribution': 'mean_exposure_contribution',
        'mean_exposure_policy_contribution': (
            'mean_exposure_policy_contribution'
        ),
        'mean_positive_correlation': 'mean_mean_positive_correlation',
        'raw_mean_positive_correlation': (
            'mean_raw_mean_positive_correlation'
        ),
        'mean_reversal_risk': 'mean_mean_reversal_risk',
        'mean_candidate_clusters': 'mean_num_candidate_clusters',
        'mean_effective_candidate_k': 'mean_effective_candidate_k',
        'mean_diversification_return_contribution': (
            'mean_diversification_return_contribution'
        ),
    }
    for public_name, generated_name in aliases.items():
        summary[public_name] = summary[generated_name]
    summary['candidate_pool_expansion_rate'] = float(np.mean([
        row['candidate_pool_expanded'] for row in daily
    ]))
    summary['max_effective_candidate_k'] = int(max(
        row['effective_candidate_k'] for row in daily
    ))
    return summary


def cross_fit_ensemble_policy(ensemble_days, **calibration_kwargs):
    """两折选策略、一折留出评估，循环覆盖全部 OOF 折。"""
    fold_ids = sorted({int(day['fold']) for day in ensemble_days})
    if len(fold_ids) < 3:
        raise ValueError('嵌套 OOF 至少需要三折')
    held_out_daily = []
    fold_policies = []
    policy_fields = (
        'allocation_blend',
        'disagreement_gamma',
        'selection_risk_gamma',
        'risk_score_penalty',
        'correlation_exposure_gamma',
        'exposure_head_blend',
    )
    for held_out_fold in fold_ids:
        calibration_days = [
            day for day in ensemble_days
            if int(day['fold']) != held_out_fold
        ]
        evaluation_days = [
            day for day in ensemble_days
            if int(day['fold']) == held_out_fold
        ]
        calibration_fold_ids = sorted({
            int(day['fold']) for day in calibration_days
        })
        if held_out_fold in calibration_fold_ids:
            raise AssertionError('留出折泄漏进策略校准数据')
        calibration_dates = sorted({
            day['prediction_date'] for day in calibration_days
        })
        evaluation_dates = sorted({
            day['prediction_date'] for day in evaluation_days
        })
        if set(calibration_dates).intersection(evaluation_dates):
            raise AssertionError('留出折预测日期泄漏进策略校准数据')
        policy = calibrate_ensemble_policy(
            calibration_days,
            **calibration_kwargs,
        )
        held_metrics = evaluate_ensemble_policy(
            evaluation_days,
            policy,
            include_daily=True,
        )
        held_out_daily.extend(held_metrics['daily'])
        fold_policies.append({
            'held_out_fold': held_out_fold,
            'calibration_folds': calibration_fold_ids,
            'calibration_dates': calibration_dates,
            'evaluation_dates': evaluation_dates,
            'policy': {
                field: policy[field] for field in policy_fields
            },
            'held_out_metrics': {
                key: value
                for key, value in held_metrics.items()
                if key != 'daily'
            },
        })
    downside_weight = float(calibration_kwargs.get(
        'downside_weight',
        0.5,
    ))
    metrics = _summarize_cross_fitted_daily(
        sorted(
            held_out_daily,
            key=lambda row: (row['prediction_date'], row['fold']),
        ),
        downside_weight,
    )
    stability = {
        field: {
            'values': [
                row['policy'][field] for row in fold_policies
            ],
            'num_unique': len({
                row['policy'][field] for row in fold_policies
            }),
        }
        for field in policy_fields
    }
    return {
        'method': 'two_folds_calibrate_one_fold_evaluate',
        'metrics': metrics,
        'fold_policies': fold_policies,
        'policy_stability': stability,
    }


_MODULE_POLICY_FIELDS = {
    'risk_score': ('risk_score_penalty', 0.0),
    'reversal': ('selection_risk_gamma', 0.0),
    'correlation_cluster': ('cluster_cap_enabled', False),
    'allocation': ('allocation_blend', 0.25),
    'exposure_head': ('exposure_head_blend', 0.25),
    'correlation_exposure': ('correlation_exposure_gamma', 0.0),
}


def _module_policy_base(calibration_kwargs):
    """构造可直接传给 evaluate_ensemble_policy 的保守策略基线。"""
    risk_blends = np.asarray([
        calibration_kwargs.get('risk_1d_blend', 0.40),
        calibration_kwargs.get('risk_3d_blend', 0.60),
        calibration_kwargs.get('risk_5d_blend', 0.0),
        calibration_kwargs.get('tail_5d_blend', 0.0),
    ], dtype=np.float64)
    if (risk_blends < 0).any() or risk_blends.sum() <= 0:
        raise ValueError('风险头混合权重必须非负且权重和大于0')
    risk_blends /= risk_blends.sum()
    minimum_allocation_blend = float(calibration_kwargs.get(
        'minimum_allocation_blend',
        0.25,
    ))
    minimum_exposure_blend = float(calibration_kwargs.get(
        'minimum_exposure_blend',
        0.25,
    ))
    return {
        'allocation_blend': minimum_allocation_blend,
        'disagreement_gamma': 0.0,
        'selection_risk_gamma': 0.0,
        'risk_score_penalty': 0.0,
        'risk_1d_blend': float(risk_blends[0]),
        'risk_3d_blend': float(risk_blends[1]),
        'risk_5d_blend': float(risk_blends[2]),
        'tail_5d_blend': float(risk_blends[3]),
        'correlation_exposure_gamma': 0.0,
        'exposure_head_blend': minimum_exposure_blend,
        'selection_candidate_k': int(calibration_kwargs.get(
            'selection_candidate_k',
            20,
        )),
        'correlation_lookbacks': [
            int(value) for value in calibration_kwargs.get(
                'correlation_lookbacks',
                [20],
            )
        ],
        'cluster_cap_enabled': False,
        'cluster_correlation_threshold': float(calibration_kwargs.get(
            'cluster_correlation_threshold',
            0.60,
        )),
        'max_stocks_per_cluster': int(calibration_kwargs.get(
            'max_stocks_per_cluster',
            2,
        )),
        'cluster_max_raw_rank': int(calibration_kwargs.get(
            'cluster_max_raw_rank',
            10,
        )),
        'tail_5d_threshold': float(calibration_kwargs.get(
            'tail_5d_threshold',
            -0.03,
        )),
        'fixed_exposure_baseline': float(calibration_kwargs.get(
            'fixed_exposure_baseline',
            0.6231689453125,
        )),
        'min_exposure': float(calibration_kwargs['min_exposure']),
        'max_exposure': float(calibration_kwargs['max_exposure']),
        'allocation_temperature': float(calibration_kwargs[
            'allocation_temperature'
        ]),
        'top_k': int(calibration_kwargs.get('top_k', 5)),
        'downside_weight': float(calibration_kwargs.get(
            'downside_weight',
            0.5,
        )),
    }


def _paired_module_gate(
    candidate_metrics,
    fallback_metrics,
    return_key,
    minimum_positive_fold_fraction,
):
    """按配对日收益及折级稳健性判定单一策略模块是否可启用。"""
    candidate_daily = candidate_metrics['daily']
    fallback_daily = fallback_metrics['daily']
    if len(candidate_daily) != len(fallback_daily):
        raise AssertionError('模块门控的候选与回退日期数量不一致')
    candidate_by_key = {
        (int(row['fold']), row['prediction_date']): row
        for row in candidate_daily
    }
    fallback_by_key = {
        (int(row['fold']), row['prediction_date']): row
        for row in fallback_daily
    }
    if candidate_by_key.keys() != fallback_by_key.keys():
        raise AssertionError('模块门控的候选与回退日期不一致')
    ordered_keys = sorted(candidate_by_key)
    candidate_returns = np.asarray([
        candidate_by_key[key][return_key] for key in ordered_keys
    ], dtype=np.float64)
    fallback_returns = np.asarray([
        fallback_by_key[key][return_key] for key in ordered_keys
    ], dtype=np.float64)
    paired = candidate_returns - fallback_returns
    fold_contributions = {}
    for fold in sorted({key[0] for key in ordered_keys}):
        fold_values = [
            paired[index]
            for index, key in enumerate(ordered_keys)
            if key[0] == fold
        ]
        fold_contributions[str(fold)] = float(np.mean(fold_values))
    required_positive_folds = int(np.ceil(
        minimum_positive_fold_fraction * len(fold_contributions)
    ))
    positive_folds = sum(
        value > 0.0 for value in fold_contributions.values()
    )
    candidate_p10 = float(np.quantile(candidate_returns, 0.10))
    fallback_p10 = float(np.quantile(fallback_returns, 0.10))
    candidate_worst_fold = min(
        float(np.mean([
            candidate_by_key[key][return_key]
            for key in ordered_keys
            if key[0] == fold
        ]))
        for fold in {key[0] for key in ordered_keys}
    )
    fallback_worst_fold = min(
        float(np.mean([
            fallback_by_key[key][return_key]
            for key in ordered_keys
            if key[0] == fold
        ]))
        for fold in {key[0] for key in ordered_keys}
    )
    checks = {
        'mean_contribution_positive': bool(paired.mean() > 0.0),
        'positive_fold_fraction': bool(
            positive_folds >= required_positive_folds
        ),
        'p10_not_worse': bool(candidate_p10 >= fallback_p10 - 1e-12),
        'worst_fold_not_worse': bool(
            candidate_worst_fold >= fallback_worst_fold - 1e-12
        ),
    }
    return {
        'enabled': bool(all(checks.values())),
        'return_key': return_key,
        'mean_paired_contribution': float(paired.mean()),
        'fold_contributions': fold_contributions,
        'positive_folds': int(positive_folds),
        'required_positive_folds': int(required_positive_folds),
        'candidate_p10': candidate_p10,
        'fallback_p10': fallback_p10,
        'p10_change': candidate_p10 - fallback_p10,
        'candidate_worst_fold': candidate_worst_fold,
        'fallback_worst_fold': fallback_worst_fold,
        'worst_fold_change': candidate_worst_fold - fallback_worst_fold,
        'checks': checks,
    }


def _choose_simple_candidate(candidates, objective_key, tolerance):
    """在最优目标容差内优先选择候选声明的低复杂度策略。"""
    if not candidates:
        raise ValueError('策略阶段候选不能为空')
    best_objective = max(
        candidate['metrics'][objective_key] for candidate in candidates
    )
    eligible = [
        candidate for candidate in candidates
        if candidate['metrics'][objective_key] >= best_objective - tolerance
    ]
    return min(
        eligible,
        key=lambda candidate: (
            candidate['simplicity'],
            -candidate['metrics'][objective_key],
        ),
    )


def calibrate_module_gated_policy(
    ensemble_days,
    policy_simplicity_tolerance=0.001,
    module_min_positive_fold_fraction=2 / 3,
    cluster_cap_grid=(False, True),
    **calibration_kwargs,
):
    """分 Ranking、Allocation、Exposure 三阶段校准并逐模块门控。"""
    if not ensemble_days:
        raise ValueError('模块门控策略缺少 OOF 日期')
    policy = _module_policy_base(calibration_kwargs)
    baseline_metrics = evaluate_ensemble_policy(
        ensemble_days,
        policy,
        include_daily=True,
    )
    stage_reports = {}
    module_eligibility = {}
    allocation_grid_size = len({
        max(
            float(calibration_kwargs.get(
                'minimum_allocation_blend',
                0.25,
            )),
            float(value),
        )
        for value in calibration_kwargs.get(
            'allocation_blend_grid',
            [0.25],
        )
    })
    exposure_grid_size = len({
        max(
            float(calibration_kwargs.get(
                'minimum_exposure_blend',
                0.25,
            )),
            float(value),
        )
        for value in calibration_kwargs.get(
            'exposure_head_blend_grid',
            [0.25],
        )
    })
    total_candidates = (
        len(calibration_kwargs.get('risk_score_penalty_grid', [0.0]))
        * len(calibration_kwargs.get('selection_risk_gamma_grid', [0.0]))
        * len(tuple(cluster_cap_grid))
        + allocation_grid_size
        + exposure_grid_size
        * len(calibration_kwargs.get(
            'correlation_exposure_gamma_grid',
            [0.0],
        ))
    )
    progress = tqdm(
        total=total_candidates,
        desc=f'分阶段策略校准({len(ensemble_days)}日)',
        unit='组',
        dynamic_ncols=True,
    )

    ranking_candidates = []
    for risk_penalty in calibration_kwargs.get(
        'risk_score_penalty_grid',
        [0.0],
    ):
        for reversal_gamma in calibration_kwargs.get(
            'selection_risk_gamma_grid',
            [0.0],
        ):
            for cluster_enabled in cluster_cap_grid:
                candidate_policy = dict(policy)
                candidate_policy.update({
                    'risk_score_penalty': float(risk_penalty),
                    'selection_risk_gamma': float(reversal_gamma),
                    'cluster_cap_enabled': bool(cluster_enabled),
                })
                metrics = evaluate_ensemble_policy(
                    ensemble_days,
                    candidate_policy,
                )
                ranking_candidates.append({
                    'policy': candidate_policy,
                    'metrics': metrics,
                    'simplicity': (
                        int(float(risk_penalty) > 0.0)
                        + int(float(reversal_gamma) > 0.0)
                        + int(bool(cluster_enabled)),
                        float(risk_penalty) + float(reversal_gamma),
                        int(bool(cluster_enabled)),
                    ),
                })
                progress.update()
    ranking_choice = _choose_simple_candidate(
        ranking_candidates,
        'ranking_policy_objective',
        policy_simplicity_tolerance,
    )
    policy.update({
        field: ranking_choice['policy'][field]
        for field in (
            'risk_score_penalty',
            'selection_risk_gamma',
            'cluster_cap_enabled',
        )
    })
    for module in ('risk_score', 'reversal', 'correlation_cluster'):
        field, fallback = _MODULE_POLICY_FIELDS[module]
        if policy[field] == fallback:
            module_eligibility[module] = {
                'enabled': False,
                'reason': 'simplicity_selected_fallback',
            }
            continue
        candidate_metrics = evaluate_ensemble_policy(
            ensemble_days,
            policy,
            include_daily=True,
        )
        fallback_policy = dict(policy)
        fallback_policy[field] = fallback
        fallback_metrics = evaluate_ensemble_policy(
            ensemble_days,
            fallback_policy,
            include_daily=True,
        )
        gate = _paired_module_gate(
            candidate_metrics,
            fallback_metrics,
            'top5_return',
            module_min_positive_fold_fraction,
        )
        module_eligibility[module] = gate
        if not gate['enabled']:
            policy[field] = fallback
    ranking_metrics = evaluate_ensemble_policy(
        ensemble_days,
        policy,
        include_daily=True,
    )
    stage_reports['ranking'] = {
        'baseline_metrics': {
            key: baseline_metrics[key] for key in (
                'mean_top5_return',
                'top5_downside_deviation',
                'ranking_policy_objective',
                'p10_weighted_portfolio_return',
                'worst_fold_top5_return',
            )
        },
        'selected_policy': {
            field: policy[field] for field in (
                'risk_score_penalty',
                'selection_risk_gamma',
                'cluster_cap_enabled',
            )
        },
        'selected_metrics': {
            key: ranking_metrics[key] for key in (
                'mean_top5_return',
                'top5_downside_deviation',
                'ranking_policy_objective',
                'mean_positive_correlation',
                'raw_mean_positive_correlation',
                'cluster_constraint_application_rate',
                'cluster_constraint_skip_rate',
                'max_selected_raw_rank',
            )
        },
    }

    minimum_allocation_blend = _MODULE_POLICY_FIELDS['allocation'][1]
    allocation_grid = sorted({
        max(minimum_allocation_blend, float(value))
        for value in calibration_kwargs.get(
            'allocation_blend_grid',
            [minimum_allocation_blend],
        )
    })
    allocation_candidates = []
    for allocation_blend in allocation_grid:
        candidate_policy = dict(policy)
        candidate_policy['allocation_blend'] = allocation_blend
        metrics = evaluate_ensemble_policy(ensemble_days, candidate_policy)
        allocation_candidates.append({
            'policy': candidate_policy,
            'metrics': metrics,
            'simplicity': (
                abs(allocation_blend - minimum_allocation_blend),
            ),
        })
        progress.update()
    allocation_choice = _choose_simple_candidate(
        allocation_candidates,
        'policy_objective',
        policy_simplicity_tolerance,
    )
    policy['allocation_blend'] = allocation_choice['policy'][
        'allocation_blend'
    ]
    if policy['allocation_blend'] > minimum_allocation_blend:
        candidate_metrics = evaluate_ensemble_policy(
            ensemble_days,
            policy,
            include_daily=True,
        )
        fallback_policy = dict(policy)
        fallback_policy['allocation_blend'] = minimum_allocation_blend
        fallback_metrics = evaluate_ensemble_policy(
            ensemble_days,
            fallback_policy,
            include_daily=True,
        )
        gate = _paired_module_gate(
            candidate_metrics,
            fallback_metrics,
            'weighted_portfolio_return',
            module_min_positive_fold_fraction,
        )
        module_eligibility['allocation'] = gate
        if not gate['enabled']:
            policy['allocation_blend'] = minimum_allocation_blend
    else:
        module_eligibility['allocation'] = {
            'enabled': False,
            'reason': 'simplicity_selected_fallback',
        }
    stage_reports['allocation'] = {
        'selected_blend': policy['allocation_blend'],
        'module_gate': module_eligibility['allocation'],
    }

    minimum_exposure_blend = _MODULE_POLICY_FIELDS['exposure_head'][1]
    exposure_blend_grid = sorted({
        max(minimum_exposure_blend, float(value))
        for value in calibration_kwargs.get(
            'exposure_head_blend_grid',
            [minimum_exposure_blend],
        )
    })
    exposure_candidates = []
    for exposure_blend in exposure_blend_grid:
        for correlation_gamma in calibration_kwargs.get(
            'correlation_exposure_gamma_grid',
            [0.0],
        ):
            candidate_policy = dict(policy)
            candidate_policy.update({
                'exposure_head_blend': exposure_blend,
                'correlation_exposure_gamma': float(correlation_gamma),
            })
            metrics = evaluate_ensemble_policy(
                ensemble_days,
                candidate_policy,
            )
            exposure_candidates.append({
                'policy': candidate_policy,
                'metrics': metrics,
                'simplicity': (
                    int(float(correlation_gamma) > 0.0),
                    abs(exposure_blend - minimum_exposure_blend),
                    float(correlation_gamma),
                ),
            })
            progress.update()
    progress.close()
    exposure_choice = _choose_simple_candidate(
        exposure_candidates,
        'policy_objective',
        policy_simplicity_tolerance,
    )
    policy.update({
        field: exposure_choice['policy'][field]
        for field in ('exposure_head_blend', 'correlation_exposure_gamma')
    })
    for module in ('exposure_head', 'correlation_exposure'):
        field, fallback = _MODULE_POLICY_FIELDS[module]
        if policy[field] == fallback:
            module_eligibility[module] = {
                'enabled': False,
                'reason': 'simplicity_selected_fallback',
            }
            continue
        candidate_metrics = evaluate_ensemble_policy(
            ensemble_days,
            policy,
            include_daily=True,
        )
        fallback_policy = dict(policy)
        fallback_policy[field] = fallback
        fallback_metrics = evaluate_ensemble_policy(
            ensemble_days,
            fallback_policy,
            include_daily=True,
        )
        gate = _paired_module_gate(
            candidate_metrics,
            fallback_metrics,
            'weighted_portfolio_return',
            module_min_positive_fold_fraction,
        )
        module_eligibility[module] = gate
        if not gate['enabled']:
            policy[field] = fallback
    final_metrics = evaluate_ensemble_policy(
        ensemble_days,
        policy,
        include_daily=True,
    )
    stage_reports['exposure'] = {
        'selected_policy': {
            field: policy[field] for field in (
                'exposure_head_blend',
                'correlation_exposure_gamma',
            )
        },
        'selected_metrics': {
            key: final_metrics[key] for key in (
                'mean_weighted_portfolio_return',
                'downside_deviation',
                'policy_objective',
                'p10_weighted_portfolio_return',
                'worst_fold_weighted_portfolio_return',
            )
        },
    }
    module_value_grids = {
        'risk_score': calibration_kwargs.get(
            'risk_score_penalty_grid',
            [0.0],
        ),
        'reversal': calibration_kwargs.get(
            'selection_risk_gamma_grid',
            [0.0],
        ),
        'correlation_cluster': tuple(cluster_cap_grid),
        'allocation': allocation_grid,
        'exposure_head': exposure_blend_grid,
        'correlation_exposure': calibration_kwargs.get(
            'correlation_exposure_gamma_grid',
            [0.0],
        ),
    }
    alternative_reports = {}
    for module, (field, fallback) in _MODULE_POLICY_FIELDS.items():
        values = [
            value for value in module_value_grids[module]
            if value != fallback
        ]
        if not values:
            alternative_reports[module] = {
                'available': False,
                'deployed': policy[field] != fallback,
            }
            continue
        return_key = (
            'top5_return'
            if module in (
                'risk_score',
                'reversal',
                'correlation_cluster',
            )
            else 'weighted_portfolio_return'
        )
        objective_key = (
            'ranking_policy_objective'
            if return_key == 'top5_return'
            else 'policy_objective'
        )
        alternatives = []
        for value in values:
            candidate_policy = dict(policy)
            candidate_policy[field] = value
            candidate_metrics = evaluate_ensemble_policy(
                ensemble_days,
                candidate_policy,
                include_daily=True,
            )
            alternatives.append((value, candidate_metrics))
        best_value, best_metrics = max(
            alternatives,
            key=lambda row: row[1][objective_key],
        )
        fallback_policy = dict(policy)
        fallback_policy[field] = fallback
        fallback_metrics = evaluate_ensemble_policy(
            ensemble_days,
            fallback_policy,
            include_daily=True,
        )
        alternative_reports[module] = {
            'available': True,
            'deployed': policy[field] != fallback,
            'fallback_value': fallback,
            'best_alternative_value': best_value,
            'best_alternative_objective': best_metrics[objective_key],
            'fallback_objective': fallback_metrics[objective_key],
            'gate': _paired_module_gate(
                best_metrics,
                fallback_metrics,
                return_key,
                module_min_positive_fold_fraction,
            ),
        }
    policy.update({
        'selection_metric': (
            'staged_ranking_then_allocation_then_exposure_with_module_gates'
        ),
        'policy_simplicity_tolerance': float(
            policy_simplicity_tolerance
        ),
        'module_min_positive_fold_fraction': float(
            module_min_positive_fold_fraction
        ),
        'module_eligibility': module_eligibility,
        'module_alternative_reports': alternative_reports,
        'module_fallbacks': {
            module: fallback
            for module, (_, fallback) in _MODULE_POLICY_FIELDS.items()
        },
        'stage_reports': stage_reports,
        'oof_metrics': final_metrics,
    })
    return policy


def cross_fit_module_gated_policy(ensemble_days, **calibration_kwargs):
    """两折分阶段校准、一折评估，并以留出贡献约束最终部署模块。"""
    fold_ids = sorted({int(day['fold']) for day in ensemble_days})
    if len(fold_ids) < 3:
        raise ValueError('模块门控嵌套 OOF 至少需要三折')
    held_out_daily = []
    held_module_pairs = {module: [] for module in _MODULE_POLICY_FIELDS}
    fold_policies = []
    policy_fields = tuple(
        field for field, _ in _MODULE_POLICY_FIELDS.values()
    ) + ('disagreement_gamma',)
    for held_out_fold in fold_ids:
        calibration_days = [
            day for day in ensemble_days if int(day['fold']) != held_out_fold
        ]
        evaluation_days = [
            day for day in ensemble_days if int(day['fold']) == held_out_fold
        ]
        calibration_folds = sorted({
            int(day['fold']) for day in calibration_days
        })
        if held_out_fold in calibration_folds:
            raise AssertionError('留出折泄漏进模块门控校准数据')
        calibration_dates = {
            day['prediction_date'] for day in calibration_days
        }
        evaluation_dates = {
            day['prediction_date'] for day in evaluation_days
        }
        if calibration_dates.intersection(evaluation_dates):
            raise AssertionError('留出折日期泄漏进模块门控校准数据')
        policy = calibrate_module_gated_policy(
            calibration_days,
            **calibration_kwargs,
        )
        held_metrics = evaluate_ensemble_policy(
            evaluation_days,
            policy,
            include_daily=True,
        )
        held_out_daily.extend(held_metrics['daily'])
        held_module_reports = {}
        for module, (field, fallback) in _MODULE_POLICY_FIELDS.items():
            if policy[field] == fallback:
                held_module_reports[module] = {
                    'active': False,
                    'mean_paired_contribution': 0.0,
                }
                continue
            fallback_policy = dict(policy)
            fallback_policy[field] = fallback
            fallback_metrics = evaluate_ensemble_policy(
                evaluation_days,
                fallback_policy,
                include_daily=True,
            )
            return_key = (
                'top5_return'
                if module in (
                    'risk_score',
                    'reversal',
                    'correlation_cluster',
                )
                else 'weighted_portfolio_return'
            )
            candidate_returns = np.asarray([
                row[return_key] for row in held_metrics['daily']
            ])
            fallback_returns = np.asarray([
                row[return_key] for row in fallback_metrics['daily']
            ])
            report = {
                'active': True,
                'return_key': return_key,
                'mean_paired_contribution': float(
                    (candidate_returns - fallback_returns).mean()
                ),
                'candidate_returns': candidate_returns.tolist(),
                'fallback_returns': fallback_returns.tolist(),
            }
            held_module_reports[module] = report
            held_module_pairs[module].append({
                'fold': held_out_fold,
                **report,
            })
        fold_policies.append({
            'held_out_fold': held_out_fold,
            'calibration_folds': calibration_folds,
            'calibration_dates': sorted(calibration_dates),
            'evaluation_dates': sorted(evaluation_dates),
            'policy': {field: policy[field] for field in policy_fields},
            'calibration_module_eligibility': policy[
                'module_eligibility'
            ],
            'held_out_module_contributions': held_module_reports,
            'held_out_metrics': {
                key: value for key, value in held_metrics.items()
                if key != 'daily'
            },
        })
    downside_weight = float(calibration_kwargs.get(
        'downside_weight',
        0.5,
    ))
    cross_fitted_metrics = _summarize_cross_fitted_daily(
        sorted(
            held_out_daily,
            key=lambda row: (row['prediction_date'], row['fold']),
        ),
        downside_weight,
    )
    deployment_eligibility = {}
    for module, rows in held_module_pairs.items():
        enabled_count = len(rows)
        if enabled_count < 2:
            deployment_eligibility[module] = {
                'eligible': False,
                'cross_fitted_enable_count': enabled_count,
                'reason': 'enabled_in_fewer_than_two_cross_fitted_policies',
            }
            continue
        paired = np.concatenate([
            np.asarray(row['candidate_returns'])
            - np.asarray(row['fallback_returns'])
            for row in rows
        ])
        fold_contributions = {
            str(row['fold']): row['mean_paired_contribution']
            for row in rows
        }
        candidate_returns = np.concatenate([
            np.asarray(row['candidate_returns']) for row in rows
        ])
        fallback_returns = np.concatenate([
            np.asarray(row['fallback_returns']) for row in rows
        ])
        checks = {
            'mean_contribution_positive': bool(paired.mean() > 0.0),
            'all_enabled_folds_positive': bool(all(
                value > 0.0 for value in fold_contributions.values()
            )),
            'p10_not_worse': bool(
                np.quantile(candidate_returns, 0.10)
                >= np.quantile(fallback_returns, 0.10) - 1e-12
            ),
            'worst_fold_not_worse': bool(min(
                fold_contributions.values()
            ) >= -1e-12),
        }
        deployment_eligibility[module] = {
            'eligible': bool(all(checks.values())),
            'cross_fitted_enable_count': enabled_count,
            'mean_paired_contribution': float(paired.mean()),
            'fold_contributions': fold_contributions,
            'checks': checks,
        }
    all_oof_candidate = calibrate_module_gated_policy(
        ensemble_days,
        **calibration_kwargs,
    )
    robust_deployment = dict(all_oof_candidate)
    for module, (field, fallback) in _MODULE_POLICY_FIELDS.items():
        if not deployment_eligibility[module]['eligible']:
            robust_deployment[field] = fallback
    robust_deployment['disagreement_gamma'] = 0.0
    robust_deployment['oof_metrics'] = evaluate_ensemble_policy(
        ensemble_days,
        robust_deployment,
        include_daily=True,
    )
    stability = {
        field: {
            'values': [row['policy'][field] for row in fold_policies],
            'num_unique': len({
                row['policy'][field] for row in fold_policies
            }),
        }
        for field in policy_fields
    }
    return {
        'method': 'module_gated_two_folds_calibrate_one_fold_evaluate',
        'metrics': cross_fitted_metrics,
        'fold_policies': fold_policies,
        'policy_stability': stability,
        'module_eligibility': deployment_eligibility,
        'all_oof_candidate_policy': all_oof_candidate,
        'robust_deployment_policy': robust_deployment,
    }


def forward_fit_module_gated_policy(
    ensemble_days,
    forward_module_max_fold_loss=0.0025,
    forward_module_max_p10_loss=0.005,
    **calibration_kwargs,
):
    """仅用历史折校准后续折，避免策略层使用未来市场阶段。"""
    fold_ids = sorted({int(day['fold']) for day in ensemble_days})
    if len(fold_ids) < 3:
        raise ValueError('严格前向模块门控至少需要三折')
    if forward_module_max_fold_loss < 0 or forward_module_max_p10_loss < 0:
        raise ValueError('前向模块容许损失必须非负')
    for day in ensemble_days:
        if 'label_end_date' not in day:
            raise ValueError('严格前向策略需要 label_end_date')
        if pd.Timestamp(day['label_end_date']) <= pd.Timestamp(
            day['prediction_date']
        ):
            raise ValueError('OOF标签结束日必须晚于预测日')

    held_out_daily = []
    fold_policies = []
    module_pairs = {module: [] for module in _MODULE_POLICY_FIELDS}
    policy_fields = tuple(
        field for field, _ in _MODULE_POLICY_FIELDS.values()
    ) + ('disagreement_gamma',)
    fallback_policy = _module_policy_base(calibration_kwargs)

    for fold_position, held_out_fold in enumerate(fold_ids):
        evaluation_days = sorted(
            [
                day for day in ensemble_days
                if int(day['fold']) == held_out_fold
            ],
            key=lambda day: day['prediction_date'],
        )
        if not evaluation_days:
            raise ValueError(f'Fold {held_out_fold} 没有前向评估日期')
        evaluation_start = pd.Timestamp(
            evaluation_days[0]['prediction_date']
        )
        calibration_days = [
            day for day in ensemble_days
            if int(day['fold']) < held_out_fold
            and pd.Timestamp(day['label_end_date']) < evaluation_start
        ]
        if fold_position < 2:
            policy = dict(fallback_policy)
            calibration_mode = 'warmup_fallback_requires_two_earlier_folds'
        else:
            if len({int(day['fold']) for day in calibration_days}) < 2:
                raise ValueError(
                    f'Fold {held_out_fold} 前没有两折已完成标签的校准日期'
                )
            policy = calibrate_module_gated_policy(
                calibration_days,
                **calibration_kwargs,
            )
            calibration_mode = 'historical_folds_only'

        for day in calibration_days:
            if int(day['fold']) >= held_out_fold:
                raise AssertionError('未来折泄漏进前向策略校准')
            if pd.Timestamp(day['label_end_date']) >= evaluation_start:
                raise AssertionError('未完成标签泄漏进前向策略校准')

        held_metrics = evaluate_ensemble_policy(
            evaluation_days,
            policy,
            include_daily=True,
        )
        held_out_daily.extend(held_metrics['daily'])
        held_module_reports = {}
        for module, (field, fallback) in _MODULE_POLICY_FIELDS.items():
            if policy[field] == fallback:
                held_module_reports[module] = {
                    'active': False,
                    'mean_paired_contribution': 0.0,
                }
                continue
            module_fallback = dict(policy)
            module_fallback[field] = fallback
            fallback_metrics = evaluate_ensemble_policy(
                evaluation_days,
                module_fallback,
                include_daily=True,
            )
            return_key = (
                'top5_return'
                if module in (
                    'risk_score',
                    'reversal',
                    'correlation_cluster',
                )
                else 'weighted_portfolio_return'
            )
            candidate_returns = np.asarray([
                row[return_key] for row in held_metrics['daily']
            ])
            fallback_returns = np.asarray([
                row[return_key] for row in fallback_metrics['daily']
            ])
            report = {
                'active': True,
                'return_key': return_key,
                'mean_paired_contribution': float(
                    (candidate_returns - fallback_returns).mean()
                ),
                'candidate_returns': candidate_returns.tolist(),
                'fallback_returns': fallback_returns.tolist(),
            }
            held_module_reports[module] = report
            module_pairs[module].append({
                'fold': held_out_fold,
                **report,
            })
        fold_policies.append({
            'held_out_fold': held_out_fold,
            'calibration_mode': calibration_mode,
            'calibration_folds': sorted({
                int(day['fold']) for day in calibration_days
            }),
            'calibration_dates': [
                day['prediction_date'] for day in calibration_days
            ],
            'evaluation_dates': [
                day['prediction_date'] for day in evaluation_days
            ],
            'policy': {field: policy[field] for field in policy_fields},
            'held_out_module_contributions': held_module_reports,
            'held_out_metrics': {
                key: value for key, value in held_metrics.items()
                if key != 'daily'
            },
        })

    downside_weight = float(calibration_kwargs.get(
        'downside_weight',
        0.25,
    ))
    forward_metrics = _summarize_cross_fitted_daily(
        sorted(
            held_out_daily,
            key=lambda row: (row['prediction_date'], row['fold']),
        ),
        downside_weight,
    )
    deployment_eligibility = {}
    for module, rows in module_pairs.items():
        if not rows:
            deployment_eligibility[module] = {
                'eligible': False,
                'forward_enable_count': 0,
                'reason': 'never_enabled_by_historical_calibration',
            }
            continue
        paired = np.concatenate([
            np.asarray(row['candidate_returns'])
            - np.asarray(row['fallback_returns'])
            for row in rows
        ])
        candidate_returns = np.concatenate([
            np.asarray(row['candidate_returns']) for row in rows
        ])
        fallback_returns = np.concatenate([
            np.asarray(row['fallback_returns']) for row in rows
        ])
        fold_contributions = {
            str(row['fold']): row['mean_paired_contribution']
            for row in rows
        }
        checks = {
            'mean_contribution_positive': bool(paired.mean() > 0.0),
            'at_least_one_fold_positive': bool(any(
                value > 0.0 for value in fold_contributions.values()
            )),
            'fold_loss_within_tolerance': bool(min(
                fold_contributions.values()
            ) >= -float(forward_module_max_fold_loss)),
            'p10_loss_within_tolerance': bool(
                np.quantile(candidate_returns, 0.10)
                >= np.quantile(fallback_returns, 0.10)
                - float(forward_module_max_p10_loss)
            ),
        }
        deployment_eligibility[module] = {
            'eligible': bool(all(checks.values())),
            'forward_enable_count': len(rows),
            'mean_paired_contribution': float(paired.mean()),
            'fold_contributions': fold_contributions,
            'p10_change': float(
                np.quantile(candidate_returns, 0.10)
                - np.quantile(fallback_returns, 0.10)
            ),
            'checks': checks,
        }

    all_oof_candidate = calibrate_module_gated_policy(
        ensemble_days,
        **calibration_kwargs,
    )
    robust_deployment = dict(all_oof_candidate)
    for module, (field, fallback) in _MODULE_POLICY_FIELDS.items():
        if not deployment_eligibility[module]['eligible']:
            robust_deployment[field] = fallback
    robust_deployment['disagreement_gamma'] = 0.0
    robust_deployment['oof_metrics'] = evaluate_ensemble_policy(
        ensemble_days,
        robust_deployment,
        include_daily=True,
    )
    stability = {
        field: {
            'values': [row['policy'][field] for row in fold_policies],
            'num_unique': len({
                row['policy'][field] for row in fold_policies
            }),
        }
        for field in policy_fields
    }
    return {
        'method': 'strict_forward_historical_folds_only',
        'metrics': forward_metrics,
        'fold_policies': fold_policies,
        'policy_stability': stability,
        'module_eligibility': deployment_eligibility,
        'all_oof_candidate_policy': all_oof_candidate,
        'robust_deployment_policy': robust_deployment,
    }
