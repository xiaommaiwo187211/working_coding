# /usr/bin/env python3
# -*- coding:utf-8 -*-

from feature_func import *


class Feature_Flow:


    def __init__(self, data, sku_brand_df, price_histroy_df, uv_channel_total_df, trans_history_df):
        self.sku_brand_df = sku_brand_df
        self.data = data
        self.price_history_df = price_histroy_df
        self.uv_channel_total_df = uv_channel_total_df
        self.trans_history_df = trans_history_df
        self.end_date = data.dt.max()

        self.data_full = None


    @staticmethod
    def date_features():
        date_all = pd.DataFrame(pd.date_range('2018-01-01', '2021-12-31').astype(str), columns=['dt'])
        date_all['month'] = date_all.dt.str[5:7].astype(int)
        date_all['year'] = date_all.dt.str[:4].astype(int)
        date_all['weekday'] = pd.to_datetime(date_all.dt).dt.weekday
        date_all['week_num'] = pd.to_datetime(date_all.dt).dt.weekofyear
        return date_all


    def history_features(self):
        # 品牌粒度前1次秒杀的UV中位数
        data_brand = self.data.sort_values('dt').groupby(['cid3', 'brand_code', 'dt']).uv.median().reset_index(level=[0, 1, 2])
        prev_uv_brand = data_brand.groupby(['cid3', 'brand_code']).rolling(1).uv.median().rename('prev_uv_brand').reset_index(level=[0, 1]).groupby(['cid3', 'brand_code']).shift(1)
        prev_uv_brand = pd.concat([data_brand, prev_uv_brand], axis=1).drop('uv', axis=1)
        # 前3次秒杀UV中位数
        self.data = self.data.merge(prev_uv_brand, on=['cid3', 'brand_code', 'dt'], how='left')
        prev3_uv_median = self.data.groupby('item_sku_id').apply(lambda x: prev3_uv_median_func(x)).rename('prev3_uv_median').reset_index(drop=True)
        # 前1次秒杀UV
        prev_uv = self.data.groupby('item_sku_id').rolling(1).uv.median().rename('prev_uv').reset_index(level=0).groupby(
            'item_sku_id').shift(1)
        prev_uv_days = self.data.groupby('item_sku_id').apply(prev_uv_days_func).rename('prev_uv_days').reset_index(
            drop=True)
        prev_uv_hour = self.data.groupby('item_sku_id').apply(prev_uv_hour_func).rename('prev_uv_hour').reset_index(
            drop=True)
        # 前1次同时段秒杀UV
        prev_hour_uv = self.data.groupby('item_sku_id').apply(lambda x: prev_hour_uv_func(x, self.data)).rename('prev_hour_uv').astype(float).reset_index(drop=True)
        # 过去30天参加秒杀次数
        count_last_month = self.data.groupby('item_sku_id').apply(count_last_month_func).rename('count_last_month').reset_index(drop=True)
        # 前1次秒杀价格
        instantprice_rolling1 = self.data.groupby('item_sku_id').rolling(1).instant_price.median().rename('instantprice_rolling1').reset_index(level=0).groupby('item_sku_id').shift(1)
        # 多少天的最低价
        # 为了引用groupby中的列
        self.data['item_sku_id_'] = self.data.item_sku_id
        days_lowest_price = self.data.groupby('item_sku_id').apply(lambda x: days_lowest_price_func(x, self.price_history_df)).rename('days_lowest_price').reset_index(drop=True)
        self.data.drop(['item_sku_id_'], axis=1, inplace=True)
        # 汇总
        data_history = pd.concat([self.data, prev_uv, prev_uv_days, prev_uv_hour, prev_hour_uv, prev3_uv_median, count_last_month, instantprice_rolling1, days_lowest_price], axis=1)
        return data_history


    def sku_uv_features(self):
        # 前7天常规流量平均
        uv_normal_rolling7 = self.uv_channel_total_df.groupby('item_sku_id').rolling(7, min_periods=1).uv_normal.median().rename(
            'uv_normal_rolling7').reset_index(level=0).groupby('item_sku_id').shift(63)
        # 前7天总流量平均
        uv_total_rolling7 = self.uv_channel_total_df.groupby('item_sku_id').rolling(7, min_periods=1).uv_total.median().rename(
            'uv_total_rolling7').reset_index(level=0).groupby('item_sku_id').shift(63)
        # 汇总
        data_uv = pd.concat([self.uv_channel_total_df[['item_sku_id', 'dt']], uv_normal_rolling7, uv_total_rolling7], axis=1)
        return data_uv


    def price_features(self):
        # 前1天红价
        redprice_rolling1 = self.price_history_df.groupby('item_sku_id').rolling(1).redprice.median().rename(
            'redprice_rolling1').reset_index(level=0).groupby('item_sku_id').shift(63)
        # 前7天红价中位数
        redprice_rolling7 = self.price_history_df.groupby('item_sku_id').rolling(7, min_periods=1).redprice.median().rename(
            'redprice_rolling7').reset_index(level=0).groupby('item_sku_id').shift(63)
        # 当天最高红价、最长时间红价、成交价、后台京东价
        price_df = self.price_history_df[['item_sku_id', 'redprice_max', 'redprice', 'netprice', 'jdprice', 'dt']]
        price_df['jdprice'] = price_df.groupby('item_sku_id').jdprice.fillna(method='ffill').fillna(method='bfill')
        # 汇总
        end_date = self.end_date
        data_price = pd.concat([self.price_history_df[['item_sku_id', 'dt']], redprice_rolling1, redprice_rolling7], axis=1) \
            .query("'2018-01-01' <= dt <= '@end_date'")
        data_price = data_price.merge(price_df, on=['item_sku_id', 'dt'], how='left')
        return data_price


    def brand_uv_features(self):
        brand_uv_channel_total_df = self.uv_channel_total_df.merge(self.sku_brand_df[['item_sku_id', 'cid3', 'brand_code']],
                                                              on='item_sku_id')
        brand_uv_channel_total_df = brand_uv_channel_total_df.groupby(['dt', 'cid3', 'brand_code']).apply(
            lambda x: pd.Series([x.uv_normal.median(), x.uv_total.median()],
                                index=['uv_normal', 'uv_total'])).reset_index().sort_values(['cid3', 'brand_code', 'dt'])
        # 前7天品牌常规流量中位数
        brand_uv_normal_rolling7 = brand_uv_channel_total_df.groupby(['cid3', 'brand_code']).uv_normal.rolling(7,
                                                                                                     min_periods=1).median().rename(
            'brand_uv_normal_rolling7').reset_index(level=[0, 1]).groupby(['cid3', 'brand_code']).shift(63)
        # 前7天品牌总流量中位数
        brand_uv_total_rolling7 = brand_uv_channel_total_df.groupby(['cid3', 'brand_code']).uv_total.rolling(7,
                                                                                                   min_periods=1).median().rename(
            'brand_uv_total_rolling7').reset_index(level=[0, 1]).groupby(['cid3', 'brand_code']).shift(63)
        # 汇总
        end_date = self.end_date
        data_brand_uv = pd.concat(
            [brand_uv_channel_total_df[['cid3', 'brand_code', 'dt']], brand_uv_normal_rolling7, brand_uv_total_rolling7], axis=1) \
            .query("'2018-01-01' <= dt <= '@end_date'")
        return data_brand_uv


    def cid3_uv_features(self):
        cid3_uv_channel_total_df = self.uv_channel_total_df.merge(self.sku_brand_df[['item_sku_id', 'cid3']],
                                                              on='item_sku_id').groupby(['dt', 'cid3']).apply(
            lambda x: pd.Series([x.uv_normal.median(), x.uv_total.median()],
                                index=['uv_normal', 'uv_total'])).reset_index().sort_values(['cid3', 'dt'])
        # 前7天品类常规流量中位数
        cid3_uv_normal_rolling7 = cid3_uv_channel_total_df.groupby('cid3').uv_normal.rolling(7,
                                                                                             min_periods=1).median().rename(
            'cid3_uv_normal_rolling7').reset_index(level=0).groupby('cid3').shift(63)
        # 前7天品类总流量中位数
        cid3_uv_total_rolling7 = cid3_uv_channel_total_df.groupby('cid3').uv_total.rolling(7,
                                                                                           min_periods=1).median().rename(
            'cid3_uv_total_rolling7').reset_index(level=0).groupby('cid3').shift(63)
        # 汇总
        end_date = self.end_date
        data_cid3_uv = pd.concat([cid3_uv_channel_total_df[['cid3', 'dt']], cid3_uv_normal_rolling7, cid3_uv_total_rolling7], axis=1) \
            .query("'2018-01-01' <= dt <= @end_date")
        return data_cid3_uv


    def transaction_features(self):
        # 前7天销量平均
        sale_qtty_rolling7 = self.trans_history_df.groupby('item_sku_id').rolling(7, min_periods=1).sale_qtty.median().rename(
            'sale_qtty_rolling7').reset_index(level=0).groupby('item_sku_id').shift(63)
        # 前7天订单量平均
        ord_count_rolling7 = self.trans_history_df.groupby('item_sku_id').rolling(7, min_periods=1).ord_count.median().rename(
            'ord_count_rolling7').reset_index(level=0).groupby('item_sku_id').shift(63)
        data_trans = pd.concat(
            [self.trans_history_df[['item_sku_id', 'dt']], sale_qtty_rolling7, ord_count_rolling7], axis=1)
        return data_trans


    def fill_nan(self):
        self.data_full = self.data_full[self.data_full.dt >= '2018-03-10']
        self.data_full['netprice'] = self.data_full.netprice.fillna(self.data_full.redprice)
        self.data_full['jdprice'] = self.data_full.jdprice.fillna(self.data_full.redprice_max)


    def remove_abnormal(self):
        # 当日最大红价不能等于秒杀价
        self.data_full = self.data_full[self.data_full.redprice_max > self.data_full.instant_price]


    def combine_features(self):
        self.data_full['sin_week'] = np.sin(2 * np.pi * self.data_full.week_num / 52)
        self.data_full['cos_week'] = np.cos(2 * np.pi * self.data_full.week_num / 52)

        # 当日折扣金额（基于红价）
        self.data_full['red_instant_diff'] = self.data_full.redprice_max - self.data_full.instant_price
        # 当日折扣率（基于红价）
        self.data_full['instant_discount_red'] = 1 - self.data_full.instant_price / self.data_full.redprice_max
        # 当日折扣率（基于京东价）
        self.data_full['instant_discount_jd'] = 1 - self.data_full.instant_price / self.data_full.jdprice

        # 相比上1次秒杀的折扣率变化率
        self.data_full['instant_discount_jd_rolling1'] = self.data_full.groupby('item_sku_id').apply(
            lambda x: 1 - x.instant_price / x.jdprice).rename('instant_discount_jd_rolling1').reset_index(
            level=0).groupby('item_sku_id').shift(1)
        self.data_full[
            'instant_discount_jd_diff_percent'] = 1 - self.data_full.instant_discount_jd_rolling1 / self.data_full.instant_discount_jd
        # 相比上1次秒杀的秒杀价变化率
        self.data_full['instantprice_diff'] = self.data_full.instant_price - self.data_full.instantprice_rolling1
        self.data_full['instantprice_diff_percent'] = self.data_full.instantprice_diff / self.data_full.instantprice_rolling1
        # self.data_full.replace([-np.inf, np.inf], [-1, 1], inplace=True)


    def return_features(self):
        date_all = self.date_features()
        data_history = self.history_features()
        data_uv = self.sku_uv_features()
        data_price = self.price_features()
        data_brand_uv = self.brand_uv_features()
        data_cid3_uv = self.cid3_uv_features()
        data_trans = self.transaction_features()

        self.data_full = data_history.merge(data_uv, on=['item_sku_id', 'dt'], how='left') \
              .merge(data_trans, on=['item_sku_id', 'dt'], how='left') \
              .merge(data_price, on=['item_sku_id', 'dt'], how='left') \
              .merge(data_cid3_uv, on=['cid3', 'dt'], how='left') \
              .merge(data_brand_uv, on=['cid3', 'brand_code', 'dt'], how='left') \
              .merge(date_all, on='dt') \
              .sort_values(['item_sku_id', 'dt']).reset_index(drop=True)

        self.fill_nan()
        self.remove_abnormal()
        self.combine_features()
        # self.data_full.dropna(inplace=True)

        return self.data_full.sort_values(['item_sku_id', 'dt']).reset_index(drop=True)



if __name__ == '__main__':
    from data_flow.base_spark import read_data
    data_flow = read_data(channel=2, threshold=50)
    end_date = data_flow[0]['dt'].max()
    feature_flow = Feature_Flow(end_date, *(data_flow[:-1]))
    features_df = feature_flow.return_features()
    features_df.to_csv('data/features_df.csv', index=False)
    print(features_df)
