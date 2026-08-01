#!/usr/bin/env python3
"""
获取沪深300指数成分股历史数据
- 获取2026年2月20日沪深300的300个成分股
- 抓取每只股票从2015年至今的历史量价数据
- 使用baostock平台
- 保存格式: 股票代码,日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌额,换手率,涨跌幅
"""

import argparse

import baostock as bs
import pandas as pd
from datetime import datetime
import os
import time


def login():
    """登录baostock"""
    lg = bs.login()
    if lg.error_code != '0':
        raise Exception(f"登录失败: {lg.error_msg}")
    print("baostock登录成功")
    return lg


def logout():
    """登出baostock"""
    bs.logout()
    print("baostock已登出")


def get_hs300_stocks():
    """获取沪深300成分股列表"""
    print("正在获取沪深300成分股列表...")
    
    rs = bs.query_hs300_stocks()
    
    if rs.error_code != '0':
        raise Exception(f"获取成分股失败: {rs.error_msg}")
    
    stocks = []
    while (rs.error_code == '0') & rs.next():
        stocks.append(rs.get_row_data())
    
    df = pd.DataFrame(stocks, columns=rs.fields)
    print(f"获取到 {len(df)} 只沪深300成分股")
    return df


def get_stock_history(bs_code, start_date, end_date):
    """获取单只股票历史数据"""
    rs = bs.query_history_k_data_plus(bs_code,
        "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg",
        start_date=start_date, end_date=end_date,
        frequency="d", adjustflag="1")  # adjustflag="1"表示后复权
    
    if rs.error_code != '0':
        raise Exception(f"查询失败: {rs.error_msg}")
    
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    
    if not data_list:
        return None
    
    df = pd.DataFrame(data_list, columns=rs.fields)
    
    # 转换数据类型
    numeric_cols = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pctChg']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 计算振幅和涨跌额
    df['振幅'] = ((df['high'] - df['low']) / df['preclose'] * 100).round(2)
    df['涨跌额'] = (df['close'] - df['preclose']).round(2)
    
    # 转换日期格式 YYYY/M/D
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y/%-m/%-d')
    
    # 提取纯数字股票代码（统一为6位格式，不足前面补0）
    df['code'] = df['code'].str.replace('sh.', '').str.replace('sz.', '')
    df['code'] = df['code'].str.zfill(6)
    
    # 重命名列
    df = df.rename(columns={
        'code': '股票代码',
        'date': '日期',
        'open': '开盘',
        'close': '收盘',
        'high': '最高',
        'low': '最低',
        'volume': '成交量',
        'amount': '成交额',
        'turn': '换手率',
        'pctChg': '涨跌幅'
    })
    
    columns = ['股票代码', '日期', '开盘', '收盘', '最高', '最低', 
               '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅']
    df = df[columns]
    
    return df


def get_stock_industry_snapshot(snapshot_date, stock_codes):
    """获取指定日期可见的行业分类，并限制到当前训练股票池。"""
    rs = bs.query_stock_industry(date=snapshot_date)
    if rs.error_code != '0':
        raise RuntimeError(
            f"获取 {snapshot_date} 行业分类失败: {rs.error_msg}"
        )
    rows = []
    while (rs.error_code == '0') & rs.next():
        rows.append(rs.get_row_data())
    snapshot = pd.DataFrame(rows, columns=rs.fields)
    required = {'code', 'industry', 'industryClassification'}
    missing = required.difference(snapshot.columns)
    if missing:
        raise ValueError(f"行业接口缺少字段: {sorted(missing)}")
    snapshot['stock_id'] = (
        snapshot['code'].astype(str).str.extract(r'(\d{6})$')[0]
    )
    snapshot = snapshot[
        snapshot['stock_id'].isin(set(stock_codes))
    ].copy()
    snapshot['effective_date'] = pd.Timestamp(
        snapshot_date
    ).strftime('%Y-%m-%d')
    snapshot['industry'] = snapshot['industry'].fillna('').str.strip()
    snapshot['industry_classification'] = (
        snapshot['industryClassification'].fillna('').str.strip()
    )
    snapshot.loc[
        snapshot['industry'].eq(''),
        'industry',
    ] = 'UNKNOWN'
    return snapshot[[
        'effective_date',
        'stock_id',
        'industry',
        'industry_classification',
    ]]


def update_industry_history(
    output_dir,
    stock_codes,
    start_date,
    end_date,
    extra_dates,
):
    """按年度和实验边界增量维护行业历史快照。"""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    annual_dates = [
        pd.Timestamp(year=year, month=12, day=31)
        for year in range(start.year, end.year)
    ]
    requested_dates = {
        date.strftime('%Y-%m-%d')
        for date in [start, *annual_dates, end, *extra_dates]
        if start <= date <= end
    }
    output_path = os.path.join(output_dir, 'stock_industry_history.csv')
    if os.path.isfile(output_path):
        existing = pd.read_csv(
            output_path,
            dtype={'stock_id': str},
        )
        existing['stock_id'] = existing['stock_id'].str.zfill(6)
    else:
        existing = pd.DataFrame(columns=[
            'effective_date',
            'stock_id',
            'industry',
            'industry_classification',
        ])
    completed_dates = set(existing['effective_date'].astype(str))
    missing_dates = sorted(requested_dates.difference(completed_dates))
    snapshots = [existing]
    for snapshot_date in missing_dates:
        print(f"获取行业分类快照: {snapshot_date}")
        snapshots.append(get_stock_industry_snapshot(
            snapshot_date,
            stock_codes,
        ))
    history = pd.concat(snapshots, ignore_index=True)
    history = history.drop_duplicates(
        subset=['effective_date', 'stock_id'],
        keep='last',
    ).sort_values(['effective_date', 'stock_id'])
    history.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(
        f"行业分类已写入: {output_path} "
        f"({history['effective_date'].nunique()} 个快照)"
    )
    return output_path


def get_existing_stocks(output_path):
    """获取已经保存的股票代码列表"""
    if not os.path.exists(output_path):
        return set()
    try:
        df = pd.read_csv(output_path, dtype={'股票代码': str})
        if '股票代码' in df.columns and len(df) > 0:
            return set(df['股票代码'].str.zfill(6).unique())
    except:
        pass
    return set()


def filter_data_by_date_range(df, start_date, end_date):
    """过滤DataFrame，仅保留目标时间窗内的数据"""
    if df is None or df.empty:
        return df

    if '日期' not in df.columns:
        return df

    filtered = df.copy()
    filtered.loc[:, '日期_dt'] = pd.to_datetime(
        filtered['日期'],
        format='mixed',
        errors='coerce',
    )
    filtered = filtered.dropna(subset=['日期_dt'])

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    filtered = filtered[(filtered['日期_dt'] >= start_dt) & (filtered['日期_dt'] <= end_dt)].copy()
    filtered = filtered.drop(columns=['日期_dt'])
    return filtered


def parse_args():
    parser = argparse.ArgumentParser(description="下载或增量更新沪深 300 历史日线数据")
    parser.add_argument(
        "--start-date",
        default="2023-01-01",
        help="下载起始日期，格式 YYYY-MM-DD，默认 2023-01-01",
    )
    parser.add_argument(
        "--end-date",
        default="2026-07-20",
        help="下载结束日期，格式 YYYY-MM-DD，默认 2026-07-20",
    )
    parser.add_argument(
        "--output-dir",
        default="./data",
        help="数据输出目录，默认 ./data",
    )
    parser.add_argument(
        "--industry-only",
        action="store_true",
        help="只增量获取行业快照，不下载日线行情",
    )
    parser.add_argument(
        "--industry-snapshot-dates",
        default="2026-01-06,2026-03-06,2026-05-06,2026-07-06",
        help=(
            "额外行业快照日期，逗号分隔 YYYY-MM-DD；默认覆盖三折 "
            "train_end 与全量训练截止日"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    start_date = args.start_date
    end_date = args.end_date
    
    output_path = os.path.join(save_dir, "stock_data.csv")
    
    print(f"目标数据时间范围: {start_date} 至 {end_date}")
    print(f"输出文件: {output_path}")
    print("=" * 60)
    
    # 检查已有的数据
    existing_stocks = get_existing_stocks(output_path)
    if existing_stocks:
        print(f"发现已有数据，包含 {len(existing_stocks)} 只股票，将检查每只股票是否需要增量更新")
    
    # 登录baostock
    login()
    
    try:
        # 获取沪深300成分股
        hs300_df = get_hs300_stocks()
        
        # 保存成分股列表
        hs300_list_path = os.path.join(save_dir, "hs300_stock_list.csv")
        hs300_df.to_csv(hs300_list_path, index=False, encoding='utf-8-sig')
        pure_stock_codes = (
            hs300_df['code']
            .str.extract(r'(\d{6})$')[0]
            .dropna()
            .str.zfill(6)
            .tolist()
        )
        extra_industry_dates = [
            pd.Timestamp(value.strip())
            for value in args.industry_snapshot_dates.split(',')
            if value.strip()
        ]
        update_industry_history(
            save_dir,
            pure_stock_codes,
            start_date,
            end_date,
            extra_industry_dates,
        )
        if args.industry_only:
            return
        
        # 读取现有数据（用于增量合并）
        existing_df = None
        if os.path.exists(output_path) and len(existing_stocks) > 0:
            try:
                existing_df = pd.read_csv(
                    output_path,
                    dtype={'股票代码': str},
                )
                raw_len = len(existing_df)
                existing_df = filter_data_by_date_range(existing_df, start_date, end_date)
                existing_df['股票代码'] = (
                    existing_df['股票代码'].str.zfill(6)
                )
                filtered_len = len(existing_df)
                print(f"  已加载现有数据: {len(existing_df)} 条记录")
                if filtered_len != raw_len:
                    print(f"  已按目标区间过滤旧数据: {raw_len} -> {filtered_len}")
            except Exception as e:
                print(f"  警告: 读取现有数据失败: {e}")

        existing_ranges = {}
        if existing_df is not None and not existing_df.empty:
            date_values = pd.to_datetime(
                existing_df['日期'],
                format='mixed',
                errors='coerce',
            )
            existing_ranges = (
                pd.DataFrame({
                    '股票代码': existing_df['股票代码'],
                    '日期_dt': date_values,
                })
                .dropna(subset=['日期_dt'])
                .groupby('股票代码')['日期_dt']
                .agg(['min', 'max'])
                .apply(lambda row: (
                    row['min'].strftime('%Y-%m-%d'),
                    row['max'].strftime('%Y-%m-%d'),
                ), axis=1)
                .to_dict()
            )
        
        # 准备处理所有股票（统一为6位字符串格式）
        hs300_df['纯代码'] = hs300_df['code'].str.replace('sh.', '').str.replace('sz.', '').str.zfill(6)
        
        # 统计信息
        failed_stocks = []
        total = len(hs300_df)
        success_count = 0
        new_stock_count = 0
        incremental_count = 0
        total_new_records = 0
        pending_updates = []
        
        for idx, row in hs300_df.iterrows():
            bs_code = row.get('code', '')
            stock_name = row.get('code_name', '')
            pure_code = row.get('纯代码', '')
            
            # 检查该股票是否已存在数据
            existing_min_date, existing_max_date = existing_ranges.get(
                pure_code,
                (None, None),
            )
            
            if existing_min_date and existing_max_date:
                # 已有数据，检查是否需要增量
                need_early = existing_min_date > start_date
                need_late = existing_max_date < end_date
                
                if not need_early and not need_late:
                    print(f"\n[{idx+1}/{total}] {bs_code} {stock_name} - 数据已完整 ({existing_min_date} 至 {existing_max_date})，跳过")
                    continue
                
                print(f"\n[{idx+1}/{total}] {bs_code} {stock_name} - 增量更新")
                print(f"  现有数据范围: {existing_min_date} 至 {existing_max_date}")
                
                # 计算需要获取的日期范围
                fetch_ranges = []
                if need_early:
                    fetch_start = start_date
                    fetch_end = (datetime.strptime(existing_min_date, '%Y-%m-%d') - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    fetch_ranges.append((fetch_start, fetch_end, "早期"))
                if need_late:
                    late_start = datetime.strptime(existing_max_date, '%Y-%m-%d') + pd.Timedelta(days=1)
                    fetch_start = max(pd.to_datetime(start_date), pd.to_datetime(late_start)).strftime('%Y-%m-%d')
                    fetch_end = end_date
                    fetch_ranges.append((fetch_start, fetch_end, "近期"))
            else:
                # 全新股票
                print(f"\n[{idx+1}/{total}] {bs_code} {stock_name} - 全新获取")
                fetch_ranges = [(start_date, end_date, "全量")]
            
            try:
                all_new_data = []
                for fetch_start, fetch_end, period_name in fetch_ranges:
                    print(f"  获取{period_name}数据: {fetch_start} 至 {fetch_end}")
                    stock_data = get_stock_history(bs_code, fetch_start, fetch_end)
                    if stock_data is not None and not stock_data.empty:
                        all_new_data.append(stock_data)
                
                if all_new_data:
                    new_data = pd.concat(all_new_data, ignore_index=True)
                    
                    if existing_df is not None and len(existing_df) > 0:
                        pending_updates.append(new_data)
                        incremental_count += 1
                    else:
                        pending_updates.append(new_data)
                        new_stock_count += 1
                    
                    total_new_records += len(new_data)
                    success_count += 1
                    print(f"  ✓ 获取成功，新增 {len(new_data)} 条记录")
                else:
                    print(f"  ✗ 无新数据")
                    
            except Exception as e:
                print(f"  ✗ 失败: {e}")
                failed_stocks.append((bs_code, stock_name))
            
            # 每10只成功获取的股票暂停一下
            if success_count > 0 and success_count % 10 == 0:
                print(f"\n  --- 已处理 {success_count} 只，暂停2秒 ---")
                time.sleep(2)
        
        if pending_updates:
            frames = ([] if existing_df is None else [existing_df])
            frames.extend(pending_updates)
            existing_df = pd.concat(frames, ignore_index=True)
            existing_df['股票代码'] = existing_df['股票代码'].astype(str).str.zfill(6)
            existing_df['日期_dt'] = pd.to_datetime(
                existing_df['日期'], format='mixed', errors='raise'
            )
            existing_df = (
                existing_df.drop_duplicates(
                    subset=['股票代码', '日期_dt'], keep='last'
                )
                .sort_values(['股票代码', '日期_dt'])
                .drop(columns=['日期_dt'])
            )
            existing_df.to_csv(output_path, index=False, encoding='utf-8-sig')

        # 显示结果
        print("\n" + "=" * 60)
        print("本次运行完成!")
        print(f"  - 全新获取: {new_stock_count} 只股票")
        print(f"  - 增量更新: {incremental_count} 只股票")
        print(f"  - 失败: {len(failed_stocks)} 只股票")
        print(f"  - 新增记录: {total_new_records}")
        
        # 验证总数据
        if os.path.exists(output_path):
            df = pd.read_csv(output_path)
            print(f"\n文件总览:")
            print(f"  - 文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
            print(f"  - 总行数: {len(df)}")
            print(f"  - 股票数量: {df['股票代码'].nunique()}")
            if len(df) > 0:
                print(f"  - 时间范围: {df['日期'].min()} 至 {df['日期'].max()}")
                
                # 验证同一股票数据是否相邻
                stock_blocks = df.groupby('股票代码').apply(lambda x: x.index.max() - x.index.min() + 1).sum()
                if stock_blocks == len(df):
                    print("  - 数据组织: ✓ 同一股票数据相邻")
                else:
                    print(f"  - 数据组织: 警告，股票数据块总长度({stock_blocks})与总行数({len(df)})不一致")
                
                print("\n前3行数据预览:")
                print(df.head(3).to_string(index=False))
                print("\n最后3行数据预览:")
                print(df.tail(3).to_string(index=False))
        
        # 保存失败列表
        if failed_stocks:
            failed_df = pd.DataFrame(failed_stocks, columns=['股票代码', '股票名称'])
            failed_path = os.path.join(save_dir, "failed_stocks.csv")
            failed_df.to_csv(failed_path, index=False, encoding='utf-8-sig')
            print(f"\n失败股票列表已保存至: {failed_path}")
    
    finally:
        logout()


if __name__ == "__main__":
    main()
