# /usr/bin/env python3
# -*- coding:utf-8 -*-

from base_spark import *
from feature_flow import Feature_Flow
from model_train_test import Model_Train_Test
from feature_flow_offline import Feature_Flow_Offline


FEATURE_COLS = ['instant_price', 'prev_uv', 'prev_hour_uv', 'prev_uv_days', 'prev_uv_hour', 'prev3_uv_median',
                'count_last_month', 'instantprice_rolling1', 'days_lowest_price', 'redprice_rolling7',
                'redprice_max', 'netprice', 'jdprice', 'instant_discount_jd', 'instant_discount_real_jd',
                'jd_instant_diff', 'instant_discount_jd_diff_percent', 'instantprice_diff_percent', 'month',
                'weekday', 'sin_week', 'cos_week', 'instant_hour', 'brand_code', 'cid3', 'year', 'uv_total_rolling7',
                'sale_qtty_rolling7', 'ord_count_rolling7', 'brand_uv_total_rolling7', 'cid3_uv_total_rolling7',
                'uv_normal_rolling7', 'brand_uv_normal_rolling7', 'cid3_uv_normal_rolling7', 'item_sku_id',
                'brand_code', 'dt', 'uv']

CATE_COLS = ['item_sku_id', 'brand_code', 'cid3']

FEATURES_OFFLINE_COLS = ['item_sku_id', 'instant_channel', 'brand_code', 'brand_uv_normal_rolling7', 'brand_uv_total_rolling7',
         'cid3_uv_normal_rolling7', 'cid3_uv_total_rolling7', 'count_last_month', 'days_lowest_price',
         'instant_discount_jd_rolling1', 'instantprice_rolling1', 'jdprice', 'ord_count_rolling7',
         'prev3_uv_median', 'prev_hour_uv_0', 'prev_hour_uv_10', 'prev_hour_uv_12', 'prev_hour_uv_14',
         'prev_hour_uv_16', 'prev_hour_uv_18', 'prev_hour_uv_20', 'prev_hour_uv_22', 'prev_hour_uv_6',
         'prev_hour_uv_8', 'prev_uv', 'prev_uv_brand_last', 'prev_uv_dt', 'redprice_rolling1',
         'redprice_rolling7', 'sale_qtty_rolling7', 'uv_hour0_median', 'uv_hour10_median', 'uv_hour12_median',
         'uv_hour14_median', 'uv_hour16_median', 'uv_hour18_median', 'uv_hour20_median', 'uv_hour22_median',
         'uv_hour6_median', 'uv_hour8_median', 'uv_interval', 'uv_normal_rolling7', 'uv_total_rolling7', 'dt']


def main_data_flow(channel_list, threshold_list):
    read_data_spark()
    data_flow = read_data(mode='data_flow')
    end_date = data_flow[0]['dt'].max()

    data_list, features_df_offline_list = [], []
    for i in range(len(channel_list)):
        channel, threshold = channel_list[i], threshold_list[i]
        data = data_flow[0].query("instant_channel == @channel and uv >= @threshold") \
            .drop('instant_channel', axis=1).sort_values(['item_sku_id', 'dt']).reset_index(drop=True)
        data_flow_ = [data] + data_flow[1:]
        data_list.append(data.assign(instant_channel=channel))
        feature_flow = Feature_Flow(*(data_flow_[:-1]))
        features_df = feature_flow.return_features()
        print('Channel: %s    Feature Flow Finished...' % channel)

        feature_flow_offline = Feature_Flow_Offline(*data_flow_, features_df, end_date, channel)
        features_df_offline = feature_flow_offline.return_features()
        features_df_offline_list.append(features_df_offline)
        print('Channel: %s    Feature Flow Offline Finished...' % channel)

    data = pd.concat(data_list)
    save_offline_features(*features_df_offline_list, FEATURES_OFFLINE_COLS)
    save_ref(data)




def main_model_flow(channel_list, threshold_list, end_date=None):
    data_flow = read_data(mode='model_flow')
    end_date = data_flow[0]['dt'].max() if end_date is None else end_date

    features_df_offline_list = []
    for i in range(len(channel_list)):
        channel, threshold = channel_list[i], threshold_list[i]
        data = data_flow[0].query("instant_channel == @channel and uv >= @threshold") \
                           .drop('instant_channel', axis=1).sort_values(['item_sku_id', 'dt']).reset_index(drop=True)
        data_flow_ = [data] + data_flow[1:]
        feature_flow = Feature_Flow(*(data_flow_[:-1]))
        features_df = feature_flow.return_features()
        print('Channel: %s    Feature Flow Finished...' % channel)

        mtt = Model_Train_Test(data_flow_, channel, end_date, FEATURE_COLS, CATE_COLS, threshold)
        mod = mtt.model_train(features_df)
        print('Channel: %s    Model Train and Test Finished...' % channel)
        feature_flow_offline = Feature_Flow_Offline(*data_flow_, features_df, end_date, channel)
        features_df_offline = feature_flow_offline.return_features()
        features_df_offline_list.append(features_df_offline)
        print('Channel: %s    Feature Flow Offline Finished...' % channel)

    offline_features = save_offline_features(*features_df_offline_list, FEATURES_OFFLINE_COLS)

    test_result_list = []
    for i in range(len(channel_list)):
        channel, threshold = channel_list[i], threshold_list[i]
        mtt = Model_Train_Test(data_flow, channel, end_date, FEATURE_COLS, threshold)
        test_result = mtt.model_test(offline_features)
        test_result_list.append(test_result)
    save_test_result(*test_result_list)




if __name__ == '__main__':
    import sys
    from time import time

    start = time()
    end_date = str(datetime.today() - pd.Timedelta(days=62))[:10]

    if sys.argv[1] == 'data_flow':
        main_data_flow([1, 2], [10, 50])
    elif sys.argv[1] == 'model_flow':
        main_model_flow([1, 2], [10, 50], end_date)

    print("%s time spent: %.2f" % (sys.argv[1], time() - start))
