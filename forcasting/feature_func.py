# /usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd


def prev_uv_days_func(x):
    def prev_uv_days_sku_func(xx):
        cur_date = xx['dt']
        prev_data = x[x.dt < cur_date][-1:]
        if len(prev_data) == 0:
            return np.nan
        return (pd.to_datetime(cur_date) - pd.to_datetime(prev_data.iloc[0]['dt'])).days

    return x.apply(prev_uv_days_sku_func, axis=1)


def prev_uv_hour_func(x):
    def prev_uv_hour_sku_func(xx):
        cur_date = xx['dt']
        prev_data = x[x.dt < cur_date][-1:]
        if len(prev_data) == 0:
            return np.nan
        return prev_data.iloc[0].instant_hour

    return x.apply(prev_uv_hour_sku_func, axis=1)


def prev3_uv_median_func(x, max_days=90):
    def prev3_uv_median_sku_func(xx):
        cur_date, prev_uv_brand = xx['dt'], xx['prev_uv_brand']
        prev_date_max = str(pd.to_datetime(cur_date) - pd.Timedelta(days=max_days))[:10]
        prev_data = x[x.dt < cur_date][-3:]
        if len(prev_data) == 0:
            return prev_uv_brand
        elif len(prev_data) == 1:
            return prev_data['uv'].iloc[0]
        uv_list = []
        n = len(prev_data)
        for i in range(n):
            row = prev_data.iloc[i]
            if row['dt'] >= prev_date_max:
                uv_list.append(row['uv'])
                continue
            uv_list.append(prev_uv_brand)
        return np.median(uv_list + [prev_uv_brand]) if n == 2 else np.median(uv_list)

    return x.apply(prev3_uv_median_sku_func, axis=1)


def prev_hour_uv_func(x, data):
    def prev_hour_uv_sku_func(xx):
        cur_date, cur_hour = xx['dt'], xx['instant_hour']
        x_sub = x[x.dt < cur_date].sort_values('dt')
        if len(x_sub) == 0:
            return None
        x_sub_hour = x_sub[x_sub.instant_hour == cur_hour]
        if len(x_sub_hour) > 0:
            return x_sub_hour.iloc[-1].uv
        cur_date_month = str(pd.to_datetime(cur_date) - pd.Timedelta(days=30))[:10]
        return data[(data.dt.between(cur_date_month, cur_date)) & (data.instant_hour == cur_hour)].uv.median()

    return x.apply(prev_hour_uv_sku_func, axis=1)


def count_last_month_func(x):
    def count_last_month_sku_func(xx):
        cur_date = xx['dt']
        yesterday = str(pd.to_datetime(cur_date) - pd.Timedelta(days=1))[:10]
        last_month = str(pd.to_datetime(cur_date) - pd.Timedelta(days=30))[:10]
        return len(x[x.dt.between(last_month, yesterday)])

    return x.apply(lambda x: count_last_month_sku_func(x), axis=1)


def days_lowest_price_func(x, price_df):
    def days_lowest_price_sku_func(xx, price_df_sub):
        min_price = xx.instant_price
        redprice_arr = np.array(price_df_sub[price_df_sub.dt < xx['dt']].redprice_min)
        ind_list = np.where(redprice_arr[::-1] - min_price < 0)[0]
        if len(ind_list) == 0:
            return min(365, len(redprice_arr))
        return min(365, ind_list[0] + 1)

    item_sku_id = x.iloc[0].item_sku_id_
    price_df_sub = price_df[price_df.item_sku_id == item_sku_id]
    return x.apply(lambda x: days_lowest_price_sku_func(x, price_df_sub), axis=1)