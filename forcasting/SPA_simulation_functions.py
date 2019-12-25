#!/usr/bin/env python3
# coding:utf-8
__author__ = 'lishikun4'

from pyspark.sql import functions as F
import numpy as np
import pandas as pd
import xgboost
from sklearn.linear_model import ElasticNetCV, HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import codecs
from fbprophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
from fbprophet.plot import add_changepoints_to_plot

def add_fourier_terms(df, period, col, degree_fourier):
    """根据输入的周期和列的值，添加相应的傅里叶项来描述周期性
    """
    for i in range(1, degree_fourier + 1):
        df = df.withColumn(col + '_fourier_sin_' + str(i),
                           F.sin((2 * np.pi * F.col(col) / period) * i))
        df = df.withColumn(col + '_fourier_cos_' + str(i),
                           F.cos((2 * np.pi * F.col(col) / period) * i))
    return df


def add_datediff(df, date_col, start_date):
    """添加日期序号
    """
    df = df.withColumn('datediff', F.datediff(F.col(date_col), F.lit(start_date)))
    df = df.withColumn('datediff_square', F.pow(F.col('datediff'), 2))
    df = df.withColumn('datediff_square_root', F.pow(F.col('datediff'), 0.5))
    return df


def adjust_promo_features(df, threshold):
    threshold_coupon = 1 - threshold
    # 比较成交价与基线价，成交价过高（大于基线价的95%）即认为是假促销
    # 可直接比较优惠后金额与基于基线价格的优惠前金额
    # 假促销的情况下，促销标记变为零（无促销）
    df = df \
        .withColumn('sku_offer_flag', F.when(
        (F.col('sku_offer_flag').isNotNull()) &
        (F.col('after_prefr_amount') < threshold * F.col('synthetic_before_prefr_amount')) &
        (F.col('sku_offer_flag')), 1).otherwise(0))
    df = df \
        .withColumn('full_minus_offer_flag', F.when(
        (F.col('full_minus_offer_flag').isNotNull()) &
        (F.col('after_prefr_amount') < threshold * F.col('synthetic_before_prefr_amount')) &
        (F.col('full_minus_offer_flag')), 1).otherwise(0))
    df = df \
        .withColumn('suit_offer_flag', F.when(
        (F.col('suit_offer_flag').isNotNull()) &
        (F.col('after_prefr_amount') < threshold * F.col('synthetic_before_prefr_amount')) &
        (F.col('suit_offer_flag')), 1).otherwise(0))
    df = df \
        .withColumn('ghost_offer_flag', F.when(
        (F.col('ghost_offer_flag').isNotNull()) &
        (F.col('after_prefr_amount') < threshold * F.col('synthetic_before_prefr_amount')) &
        (F.col('ghost_offer_flag')), 1).otherwise(0))
    # 假促销的情况下，促销折扣变为零
    df = df \
        .withColumn('sku_offer_discount_rate', F.when(
        F.col('after_prefr_amount') < threshold * F.col('synthetic_before_prefr_amount'),
        F.col('sku_offer_discount_rate')).otherwise(0))
    df = df \
        .withColumn('full_minus_offer_discount_rate', F.when(
        F.col('after_prefr_amount') < threshold * F.col('synthetic_before_prefr_amount'),
        F.col('full_minus_offer_discount_rate')).otherwise(0))
    df = df \
        .withColumn('suit_offer_discount_rate', F.when(
        F.col('after_prefr_amount') < threshold * F.col('synthetic_before_prefr_amount'),
        F.col('suit_offer_discount_rate')).otherwise(0))
    df = df \
        .withColumn('ghost_offer_discount_rate', F.when(
        F.col('after_prefr_amount') < threshold * F.col('synthetic_before_prefr_amount'),
        F.col('ghost_offer_discount_rate')).otherwise(0))
    # 假促销的情况下，满减与套装促销参与度变为零
    df = df \
        .withColumn('participation_rate_full_minus_and_suit_offer', F.when(
        F.col('after_prefr_amount') < threshold * F.col('synthetic_before_prefr_amount'),
        F.col('participation_rate_full_minus_and_suit_offer')).otherwise(0))
    # 优惠券的优惠过小的情况下，不考虑优惠券的影响
    df = df \
        .withColumn('dq_and_jq_pay_flag', F.when(
        (F.col('dq_and_jq_pay_amount') >= threshold_coupon * F.col('after_prefr_amount')) &
        (F.col('dq_and_jq_pay_flag')), 1).otherwise(0))
    df = df \
        .withColumn('dq_and_jq_pay_discount_rate', F.when(
        F.col('dq_and_jq_pay_amount') >= threshold_coupon * F.col('after_prefr_amount'),
        F.col('dq_and_jq_pay_discount_rate')).otherwise(0))
    df = df \
        .withColumn('dq_pay_flag', F.when(
        (F.col('dq_pay_amount') >= threshold_coupon * F.col('after_prefr_amount')) &
        (F.col('dq_pay_flag')), 1).otherwise(0))
    df = df \
        .withColumn('dq_pay_discount_rate', F.when(
        F.col('dq_pay_amount') >= threshold_coupon * F.col('after_prefr_amount'),
        F.col('dq_pay_discount_rate')).otherwise(0))
    df = df \
        .withColumn('jq_pay_flag', F.when(
        (F.col('jq_pay_amount') >= threshold_coupon * F.col('after_prefr_amount')) &
        (F.col('jq_pay_flag')), 1).otherwise(0))
    df = df \
        .withColumn('jq_pay_discount_rate', F.when(
        F.col('jq_pay_amount') >= threshold_coupon * F.col('after_prefr_amount'),
        F.col('jq_pay_discount_rate')).otherwise(0))
    # 非促销标记做相应调整
    df = df \
        .withColumn('non_promo_flag', F.when(
        ((F.col('after_prefr_amount') < threshold * F.col('synthetic_before_prefr_amount')) |
         (F.col('jq_pay_amount') >= threshold_coupon * F.col('after_prefr_amount'))) &
        F.col('non_promo_flag'), 1).otherwise(0))
    return df


def convert_boolean_to_int(df, cols):
    if type(cols) is not list:
        cols = [cols]
    for col_to_change in cols:
        df = df.withColumn(col_to_change, F.when(F.col(col_to_change), 1).otherwise(0))
    return df


# def adjust_out_of_stock(df_train, df_test, prediction):
#     """调整矛盾的缺货标记，即有销量但标志为缺货的天
#     """
#     stock_min = df_train['stock_status'].min()
#     df_test['replace_stock_flag'] = df_test['stock_status'].apply(lambda x: int(x < stock_min))
#     prediction = prediction * df_test['replace_stock_flag']
#     return prediction

def adjust_out_of_stock(df_train, df_test, prediction):
    """调整矛盾的缺货标记，即有销量但标志为缺货的天
    """
    stock_min = df_train['stock_status'].min()
    def exchange_low_stock_value(out_of_stock_flag, stock_min):
        if out_of_stock_flag < stock_min:
            return out_of_stock_flag
        else:
            return 1
    df_test['replace_stock_flag'] = df_test['stock_status'] \
    .apply(lambda x: exchange_low_stock_value(x, stock_min))
    prediction = prediction * df_test['replace_stock_flag']
    return prediction



def judge_feature_bound_lr(df_train, df_test, feature_list):
    """如果测试期特征的范围，超过训练期特征，则使用线性回归进行预测，可能导致预测结果，数值异常。
       这种情况下，需要使用随机森林，进行预测。
    """
    threshold = 0

    def feature_judge_bound(df_train, df_test, feature):
        train_max = df_train[feature].max()
        train_min = df_train[feature].min()
        test_max = df_test[feature].max()
        test_min = df_test[feature].min()
        if (test_max > train_max * (1 + threshold)) | (test_min < train_min * (1 - threshold)):
            return True
        else:
            return False

    feature_result = [feature_judge_bound(df_train, df_test, i) for i in feature_list]
    return any(feature_result)


def judge_feature_bound_rf(df_train, df_test, feature_list):
    """如果测试期特征的范围，距离训练期特征过远，则使用随机森林进行预测，可能导致预测结果为附近节点的数值。
       这种情况下，需要使用线性回归，进行预测。
    """
    threshold = 0.15

    def judge_overstep(judge_data, bound):
        return sum([(judge_data > i) & (judge_data < j) for (i, j) in bound])

    def feature_judge_overstep(df_train, df_test, feature):
        train_feature_all_data = df_train[feature].tolist()
        test_feature_all_data = df_test[feature].tolist()
        train_feature_all_data_bound = [(i * (1 - threshold) - 0.001, i * (1 + threshold) + 0.001) for i in
                                        train_feature_all_data]
        test_feature_overstep_bound = [judge_overstep(feature_data, train_feature_all_data_bound)
                                       for feature_data in test_feature_all_data]
        return min(test_feature_overstep_bound)

    return min([feature_judge_overstep(df_train, df_test, feature) for feature in feature_list]) == 0


def calculate_baseline_sku(row, model_list, selected_columns, X_SCHEMA_SKU, rolling_columns, data_source='self'):
    model_type = 0
    raw_data = row[1]
    return model_fit(raw_data, data_source, model_list, model_type, selected_columns, X_SCHEMA_SKU, rolling_columns)


def calculate_baseline_cid3(row, data_source='self', fi_pd=None):
    model_type = 1
    raw_data = row[1]
    return model_fit(raw_data, data_source, model_type, fi_pd)


'''
def model_fit(raw_data, model_type):
    """数据转换为Pandas DataFrame，并按天排序
    """
    if model_type == 0:
        dataset = pd.DataFrame(list(raw_data), columns = DATASET_SCHEMA_SKU).sort_values('dt').drop_duplicates().reset_index(drop = True)
        result = fit_baseline_sku(dataset)
    elif  model_type == 1:
        dataset = pd.DataFrame(list(raw_data), columns = DATASET_SCHEMA_CID3).sort_values('date').drop_duplicates().reset_index(drop = True)
        result = fit_baseline_cid3(dataset)
    return result.to_dict(orient = 'records')
'''


def model_fit(raw_data, data_source, model_list, model_type, selected_columns, X_SCHEMA_SKU, rolling_columns, fi_pd=None):
    """数据转换为Pandas DataFrame，并按天排序
    """

    if model_type == 0:
        sku_input = input_sku_baseline(raw_data, model_list, selected_columns, X_SCHEMA_SKU, rolling_columns, data_source=data_source)
        result = fit_baseline_sku(sku_input, model_list)
    elif model_type == 1:
        cid_input = input_cid_baseline(raw_data, data_source=data_source)
        result = fit_baseline_cid3(cid_input) if data_source == '7fresh' else revised_fit_baseline_cid3(cid_input,
                                                                                                        fi_pd)
    return result.to_dict(orient='records')


def fit_baseline_sku(sku_input, model_list):
    """计算sku的基线销量
    """
    output_df = baseline_model_sku(model_list, sku_input=sku_input, moderator_=1).output_df
    return output_df


'''
没有删除该函数，但是新的方法不适用该函数
'''


def fit_baseline_cid3(cid_input):
    """计算品类的基线
    """
    dataset = cid_input.dataset
    try:
        x_df = dataset[cid_input.X_SCHEMA_CID3]
        y_df = dataset[cid_input.Y_SCHEMA_CID3]
        x_np = x_df.values
        y_np = y_df.values.ravel()
        reg = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=6)
        fitted_model = reg.fit(x_np, y_np)
        initial_fitting = fitted_model.predict(x_np)
        initial_fitting = np.exp(initial_fitting)
        # 去除促销相关的变量的影响，把相关变量设为0
        x_non_promo = non_promo_inputs(x_df, cid_input.cid3_promotion_features).values
        initial_base_line = fitted_model.predict(x_non_promo)
        initial_base_line = np.exp(initial_base_line)
        final_base_line = sql_ewma(initial_base_line)[0]
        output_df = dataset.copy()
        output_df['initial_fitting'] = initial_fitting
        output_df['initial_baseline'] = initial_base_line
        output_df['final_baseline'] = final_base_line
        output_df = output_df[cid_input.OUTPUT_SCHEMA_CID3]
    except:
        output_df = dataset[cid_input.OUTPUT_KEYS_CID3]
        for col in cid_input.OUTPUT_VALUES_CID3:
            output_df[col] = np.nan
    return output_df


def revised_fit_baseline_cid3(cid_input, fi_pd):
    """修改三级品类计算的算法,
    其中新增两个函数local_randomforest()与local_elasticnet()
    local_randomforest()用于拟合计算节假日相关曲线
    与local_elasticnet()用于拟合计算平常日相关曲线
    """
    dataset = cid_input.dataset
    try:
        # df_list用于将拆分的数据收集，df_list[-1]代表非节假日天的数据集
        df_list = []
        # i = 'newyear'
        for i in cid_input.SPLIT_FESTIVAL:
            current_list = fi_pd[fi_pd['festival_name'] == i].reset_index(drop=True)
            # 建立空的dataframe，之后进行concat方便
            df_empty = pd.DataFrame(columns=dataset.columns)
            for j in range(len(current_list)):
                current_date = dataset[(dataset['date'] >= str(current_list.loc[j, 'cal_start_date'])) & (
                            dataset['date'] <= str(current_list.loc[j, 'cal_end_date']))]
                df_empty = pd.concat([df_empty, current_date])
                dataset = dataset.drop(current_date.index)
            df_list.append(df_empty)

        # 将normal数据加入数据
        df_list.append(dataset)
        empty_df = pd.DataFrame(columns=cid_input.LOCAL_SCHEMA_CID3)
        # circle_randomforest_al
        for i in df_list[0:-1]:
            empty_df = pd.concat([empty_df, local_randomforest(i, cid_input)])

        empty_df = pd.concat([empty_df, local_elasticnet(df_list[-1], cid_input)])
        empty_df = empty_df.sort_values(by="date")
        final_base_line = sql_ewma(empty_df['initial_base_line'])[0]
        empty_df['final_baseline'] = final_base_line
        empty_df = empty_df[cid_input.OUTPUT_SCHEMA_CID3]
    except:
        empty_df = dataset[cid_input.OUTPUT_KEYS_CID3]
        for col in cid_input.OUTPUT_VALUES_CID3:
            empty_df[col] = np.nan
    return empty_df


def local_randomforest(dataset, cid_input):
    x_df = dataset[cid_input.X_SCHEMA_CID3]
    y_df = dataset[cid_input.Y_SCHEMA_CID3]
    x_np = x_df.values
    y_np = y_df.values.ravel()
    reg = RandomForestRegressor(n_estimators=200, criterion='mse', max_depth=10, oob_score=True, random_state=1)
    fitted_model = reg.fit(x_np, y_np)
    initial_fitting = fitted_model.predict(x_np)
    initial_fitting = np.exp(initial_fitting)
    # 去除促销相关的变量的影响，把相关变量设为0
    x_non_promo = non_promo_inputs(x_df, cid_input.cid3_promotion_features).values
    initial_base_line = fitted_model.predict(x_non_promo)
    initial_base_line = np.exp(initial_base_line)
    current_dt = dataset.copy()
    current_dt['initial_base_line'] = initial_base_line
    return current_dt[cid_input.LOCAL_SCHEMA_CID3]


def local_elasticnet(dataset, cid_input):
    x_df = dataset[cid_input.X_SCHEMA_CID3]
    y_df = dataset[cid_input.Y_SCHEMA_CID3]
    # elasticnet绝对不能用上面方法进行数据归一化
    x_np = x_df.values
    y_np = y_df.values.ravel()
    lm = ElasticNetCV(random_state=1)
    lm.fit(x_np, y_np)
    coefficients = lm.coef_
    intercept = lm.intercept_
    initial_fitting = lm.predict(x_df)
    initial_fitting = np.exp(initial_fitting)
    x_non_promo = non_promo_inputs(x_df, cid_input.cid3_promotion_features).values
    s_initial_fitting = lm.predict(x_non_promo)
    s_initial_fitting = np.exp(s_initial_fitting)
    current_dt = dataset.copy()
    current_dt['initial_base_line'] = s_initial_fitting
    return current_dt[cid_input.LOCAL_SCHEMA_CID3]


def sql_ewma(vol):
    """ OW平滑函数
    Calculate a non-quite-exponential weighted moving average; the input must be an ordered sequence
    """
    # NB: pd.algos.roll_sum was deprecated, therefore we have to revert to the other version,
    # which is also slated for deprecation
    vol = vol.astype('float64')
    # need to convert NaNs to zeros, otherwise we get NaNs instead of
    # zeros when the sliding window contains only NaNs
    pad = np.zeros(16)
    vol_exp = np.nan_to_num(np.hstack((vol, pad)))
    sum1 = pd.rolling_sum(vol_exp, 34, 0)[8:-8]
    sum2 = pd.rolling_sum(vol_exp, 22, 0)[5:-11]
    sum3 = pd.rolling_sum(vol_exp, 14, 0)[3:-13]
    sum2 = sum2 * 0.75
    sum3 = sum3 * 0.75
    # we use the count of non-NaN weeks as the weights
    # obtain the counts by converting NaN to 0, other to 1, then summing
    pad = np.zeros(16)
    nonpromo_indicator = 1 - np.isnan(vol)
    vol_exp = np.hstack((nonpromo_indicator, pad))
    weight1 = pd.rolling_sum(vol_exp, 34, 0)[8:-8]
    weight2 = pd.rolling_sum(vol_exp, 22, 0)[5:-11]
    weight3 = pd.rolling_sum(vol_exp, 14, 0)[3:-13]
    weight2 = weight2 * 0.75
    weight3 = weight3 * 0.75
    sums = sum1 + sum2 + sum3
    weights = weight1 + weight2 + weight3
    # result is zero where we have zero weeks to sum
    ma = np.zeros(len(weights))
    valid = weights > 0
    ma[valid] = sums[valid] / weights[valid]
    # weight1 also happens to be the numweeks metric that the SQL uses
    return ma, weight1


def no_promo_coefficients(variable_list, coefficients, reset_variable_list):
    """Sets coefficients relating to promotions to zeros
    """
    j = 0
    coefficients_wno_promo_temp = []
    for i in variable_list:
        variable = i.lower()
        if i in reset_variable_list:
            if i == 'non_promo_flag':
                coefficients_wno_promo_temp.append(1)
            else:
                coefficients_wno_promo_temp.append(0)
        else:
            coefficients_wno_promo_temp.append(coefficients[j])
        j += 1
    return coefficients_wno_promo_temp


def non_promo_inputs(df, reset_variable_list):
    """Sets input columns relating to promotions to zeros
    """
    df_temp = df.copy()
    for i in df_temp.columns.values:
        variable = i.lower()
        if i in reset_variable_list:
            if i == 'non_promo_flag':
                df_temp[i] = 1.0
            else:
                df_temp[i] = 0.0
    return df_temp


class input_sku_baseline:
    def __init__(self, raw_data, model_list, selected_columns, X_feature, rolling_columns, data_source='self'):
        self.data_source = data_source
        if data_source == '7fresh':
            # 7fresh sku schema
            self.DATASET_SCHEMA_SKU = ['store_id', 'sku_id', 'newyear', 'springfestival', 'tombsweepingfestival',
                                       'labourday',
                                       'dragonboatfestival', 'midautumnfestival', 'nationalday', 'week_of_year',
                                       'day_of_year',
                                       'day_of_week', 'sku_offer_flag', 'full_minus_offer_flag',
                                       'discount_code_offer_flag',
                                       'free_goods_flag', 'ghost_offer_flag', 'coupon_pay_flag', 'sale_qtty',
                                       'after_offer_amount',
                                       'before_prefr_amount', 'synthetic_before_prefr_amount', 'coupon_pay_amount',
                                       'major_offer_amount',
                                       'sku_offer_amount', 'full_minus_offer_amount', 'discount_code_offer_amount',
                                       'synthetic_total_offer_amount', 'free_goods_amount', 'sale_qtty_for_full_minus',
                                       'sale_qtty_for_coupon_pay', 'before_prefr_amount_for_free_gift',
                                       'before_prefr_amount_for_major_offer', 'before_prefr_amount_for_coupon_pay',
                                       'count_sale_ord_id',
                                       'non_promo_flag', 'participation_rate_full_minus',
                                       'participation_rate_coupon_pay',
                                       'sku_offer_discount_rate', 'full_minus_offer_discount_rate',
                                       'discount_code_offer_discount_rate',
                                       'ghost_offer_discount_rate', 'coupon_pay_discount_rate',
                                       'free_gift_discount_rate',
                                       'out_of_stock_flag', 'dt', 'log_sale_qtty', 'datediff', 'datediff_square',
                                       'datediff_square_root',
                                       'week_of_year_fourier_sin_1', 'week_of_year_fourier_cos_1',
                                       'week_of_year_fourier_sin_2',
                                       'week_of_year_fourier_cos_2', 'week_of_year_fourier_sin_3',
                                       'week_of_year_fourier_cos_3',
                                       'day_of_week_fourier_sin_1', 'day_of_week_fourier_cos_1',
                                       'day_of_week_fourier_sin_2',
                                       'day_of_week_fourier_cos_2', 'day_of_week_fourier_sin_3',
                                       'day_of_week_fourier_cos_3']
            self.X_SCHEMA_SKU = ['sku_offer_flag', 'full_minus_offer_flag', 'free_goods_flag', 'ghost_offer_flag',
                                 'discount_code_offer_flag', 'coupon_pay_flag', 'participation_rate_full_minus',
                                 'participation_rate_coupon_pay',
                                 'sku_offer_discount_rate', 'discount_code_offer_discount_rate',
                                 'full_minus_offer_discount_rate', 'ghost_offer_discount_rate',
                                 'coupon_pay_discount_rate',
                                 'free_gift_discount_rate', 'newyear',
                                 'springfestival', 'tombsweepingfestival', 'labourday', 'dragonboatfestival',
                                 'midautumnfestival',
                                 'nationalday', 'day_of_week', 'out_of_stock_flag',
                                 'week_of_year_fourier_sin_1', 'week_of_year_fourier_cos_1',
                                 'week_of_year_fourier_sin_2',
                                 'week_of_year_fourier_cos_2', 'week_of_year_fourier_sin_3',
                                 'week_of_year_fourier_cos_3',
                                 'day_of_week_fourier_sin_1', 'day_of_week_fourier_cos_1', 'day_of_week_fourier_sin_2',
                                 'day_of_week_fourier_cos_2', 'day_of_week_fourier_sin_3', 'day_of_week_fourier_cos_3']
            self.Y_SCHEMA_SKU = ['log_sale_qtty']
            self.FEATURE_BOUND = ['target_price']
            self.OUTPUT_KEYS_SKU = ['dt', 'sku_id', 'store_id']
            self.OUTPUT_VALUES_SKU = ['final_baseline', 'uplift', 'uplift_rate']
            self.OUTPUT_SCHEMA_SKU = ['dt', 'sku_id', 'store_id', 'final_baseline', 'uplift', 'uplift_rate']
            self.sku_promotion_features = ['sku_offer_flag', 'full_minus_offer_flag', 'free_goods_flag',
                                           'ghost_offer_flag', 'discount_code_offer_flag', 'non_promo_flag',
                                           'coupon_pay_flag', 'participation_rate_full_minus',
                                           'participation_rate_coupon_pay', 'sku_offer_discount_rate',
                                           'full_minus_offer_discount_rate', 'ghost_offer_discount_rate',
                                           'coupon_pay_discount_rate', 'free_gift_discount_rate',
                                           'discount_code_offer_discount_rate']
        else:
            # self sku schema
            self.DATASET_SCHEMA_SKU = selected_columns
            self.X_SCHEMA_SKU = X_feature
            self.X_SCHEMA_SKU_BASE = ['newyear', 'springfestival', 'tombsweepingfestival', 'labourday',
                                      'dragonboatfestival', 'midautumnfestival', 'nationalday', 'h1111mark', 'h618mark',
                                      'h1212mark', 'day_of_week', 'non_promo_flag',
                                      'week_of_year_fourier_sin_1', 'week_of_year_fourier_cos_1',
                                      'week_of_year_fourier_sin_2', 'week_of_year_fourier_cos_2',
                                      'week_of_year_fourier_sin_3', 'week_of_year_fourier_cos_3',
                                      'day_of_week_fourier_sin_1', 'day_of_week_fourier_cos_1',
                                      'day_of_week_fourier_sin_2', 'day_of_week_fourier_cos_2',
                                      'day_of_week_fourier_sin_3', 'day_of_week_fourier_cos_3', 'sku_status_cd',
                                      'decomposedtrend', 'rolling360mean', 'rolling180mean',
                                      'rolling90mean', 'rolling28mean', 'rolling14mean', 'rolling7mean',
                                      'rolling5mean', 'rolling3mean', 'rolling2mean', 'rolling1mean',
                                      'rolling14median', 'rolling7median', 'rolling360decaymean',
                                      'rolling180decaymean', 'rolling90decaymean', 'rolling28decaymean',
                                      'rolling14decaymean', 'rolling7decaymean', 'rolling3decaymean', 'baseprice']
            self.INCREASING_CONSTRAINT_SCHEMA = ['free_gift_flag', 'ghost_offer_flag', 'dq_and_jq_pay_flag',
                                                 'jq_pay_flag', 'dq_pay_flag', 'full_minus_offer_flag',
                                                 'suit_offer_flag', 'sku_offer_flag',
                                                 'participation_rate_full_minus_and_suit_offer',
                                                 'participation_rate_dq_and_jq_pay', 'sku_offer_discount_rate',
                                                 'full_minus_offer_discount_rate', 'suit_offer_discount_rate',
                                                 'ghost_offer_discount_rate', 'dq_and_jq_pay_discount_rate',
                                                 'jq_pay_discount_rate', 'dq_pay_discount_rate',
                                                 'free_gift_discount_rate', 'sku_status_cd']
            self.ROLLING_SCHEMA = rolling_columns
            self.DECREASING_CONSTRAINT_SCHEMA = ['non_promo_flag', 'out_of_stock_flag']
            self.Y_SCHEMA_SKU = ['target_qtty']
            self.FEATURE_BOUND = ['target_price']
            self.OUTPUT_KEYS_SKU = ['dt', 'item_sku_id']
            self.OUTPUT_VALUES_SKU = ['dt', 'item_sku_id', 'target_price', 'prediction', 'r2_predict', 'r2_test', 'mse', 'mape', 'model_type', 'model', 'feature_importance', 'valid_flag']
            self.OUTPUT_SCHEMA_SKU = ['dt', 'item_sku_id', 'prediction', 'uplift', 'uplift_rate', 'sample_num',
                                      'non_zero_sales_days', 'high_sales_days', 'status_3001_days', 'non_stock_days',
                                      'promotion_days', 'valid_sales_days', 'valid_promotion_days_ratio',
                                      'valid_non_zero_days_ratio', 'r2', 'mse', 'mape', 'model_type']
            self.sku_promotion_features = ['non_promo_flag', 'sku_offer_discount_rate',
                                           'full_minus_offer_discount_rate', 'suit_offer_discount_rate',
                                           'dq_pay_discount_rate', 'free_gift_discount_rate']

        self.dataset = pd.DataFrame(list(raw_data), columns=self.DATASET_SCHEMA_SKU).sort_values(self.DATASET_SCHEMA_SKU + ['dt']) \
            .drop_duplicates().reset_index(drop=True).fillna(0)


class input_cid_baseline:
    def __init__(self, raw_data, data_source='self'):
        self.data_source = data_source
        if data_source == '7fresh':
            # 7fresh cid4 schema
            self.DATASET_SCHEMA_CID3 = ['date', 'store_id', 'cate_id_4', 'newyear', 'springfestival',
                                        'tombsweepingfestival', 'labourday', 'dragonboatfestival', 'midautumnfestival',
                                        'nationalday', 'week_of_year', 'day_of_year', 'day_of_week', 'sale_qtty',
                                        'after_offer_amount', 'before_prefr_amount', 'synthetic_before_prefr_amount',
                                        'synthetic_sku_offer_amount', 'synthetic_discount_code_offer_amount',
                                        'synthetic_full_minus_offer_amount', 'synthetic_ghost_offer_amount',
                                        'free_gift_offer_amount', 'coupon_pay_amount', 'sku_offer_sale_qtty',
                                        'discount_code_offer_sale_qtty', 'full_minus_offer_sale_qtty',
                                        'ghost_offer_sale_qtty', 'free_gift_sale_qtty', 'coupon_pay_sale_qtty',
                                        'sku_offer_discount_rate', 'discount_code_offer_discount_rate',
                                        'full_minus_offer_discount_rate', 'ghost_offer_discount_rate',
                                        'free_gift_discount_rate', 'coupon_pay_discount_rate',
                                        'sku_offer_participation_rate', 'discount_code_offer_participation_rate',
                                        'full_minus_offer_participation_rate', 'ghost_offer_participation_rate',
                                        'free_gift_participation_rate', 'coupon_pay_participation_rate',
                                        'log_synthetic_before_prefr_amount', 'datediff', 'datediff_square',
                                        'datediff_square_root', 'week_of_year_fourier_sin_1',
                                        'week_of_year_fourier_cos_1', 'week_of_year_fourier_sin_2',
                                        'week_of_year_fourier_cos_2', 'week_of_year_fourier_sin_3',
                                        'week_of_year_fourier_cos_3', 'day_of_week_fourier_sin_1',
                                        'day_of_week_fourier_cos_1', 'day_of_week_fourier_sin_2',
                                        'day_of_week_fourier_cos_2', 'day_of_week_fourier_sin_3',
                                        'day_of_week_fourier_cos_3']
            self.X_SCHEMA_CID3 = ['newyear', 'springfestival', 'tombsweepingfestival', 'labourday',
                                  'dragonboatfestival', 'midautumnfestival', 'nationalday', 'day_of_week',
                                  'sku_offer_discount_rate', 'discount_code_offer_discount_rate',
                                  'full_minus_offer_discount_rate', 'ghost_offer_discount_rate',
                                  'free_gift_discount_rate', 'coupon_pay_discount_rate', 'sku_offer_participation_rate',
                                  'discount_code_offer_participation_rate', 'full_minus_offer_participation_rate',
                                  'ghost_offer_participation_rate', 'free_gift_participation_rate',
                                  'coupon_pay_participation_rate', 'datediff', 'datediff_square',
                                  'datediff_square_root', 'week_of_year_fourier_sin_1', 'week_of_year_fourier_cos_1',
                                  'week_of_year_fourier_sin_2', 'week_of_year_fourier_cos_2',
                                  'week_of_year_fourier_sin_3', 'week_of_year_fourier_cos_3',
                                  'day_of_week_fourier_sin_1', 'day_of_week_fourier_cos_1', 'day_of_week_fourier_sin_2',
                                  'day_of_week_fourier_cos_2', 'day_of_week_fourier_sin_3', 'day_of_week_fourier_cos_3']
            self.Y_SCHEMA_CID3 = ['log_synthetic_before_prefr_amount']
            self.OUTPUT_KEYS_CID3 = ['date', 'store_id', 'cate_id_4']
            self.OUTPUT_VALUES_CID3 = ['final_baseline']
            self.OUTPUT_SCHEMA_CID3 = ['date', 'store_id', 'cate_id_4', 'final_baseline']
            self.cid3_promotion_features = ['sku_offer_discount_rate', 'discount_code_offer_discount_rate',
                                            'full_minus_offer_discount_rate', 'ghost_offer_discount_rate',
                                            'free_gift_discount_rate', 'coupon_pay_discount_rate',
                                            'sku_offer_participation_rate', 'discount_code_offer_participation_rate',
                                            'full_minus_offer_participation_rate', 'ghost_offer_participation_rate',
                                            'free_gift_participation_rate', 'coupon_pay_participation_rate']
            self.LOCAL_SCHEMA_CID3 = ['date', 'store_id', 'cate_id_4', 'initial_base_line']
            self.SPLIT_FESTIVAL = []
        else:
            # self cid3 schema
            self.DATASET_SCHEMA_CID3 = ['date', 'item_third_cate_cd', 'newyear', 'springfestival',
                                        'tombsweepingfestival', 'labourday', 'dragonboatfestival', 'midautumnfestival',
                                        'nationalday', 'h1111mark', 'h618mark', 'h1212mark', 'week_of_year',
                                        'day_of_year', 'day_of_week', 'sale_qtty', 'after_prefr_amount',
                                        'before_prefr_amount', 'synthetic_before_prefr_amount',
                                        'sku_offer_discount_rate', 'suit_offer_discount_rate',
                                        'full_minus_offer_discount_rate', 'ghost_offer_discount_rate',
                                        'free_gift_discount_rate', 'dq_and_jq_pay_discount_rate',
                                        'jq_pay_discount_rate', 'dq_pay_discount_rate', 'sku_offer_participation_rate',
                                        'suit_offer_participation_rate', 'full_minus_offer_participation_rate',
                                        'ghost_offer_participation_rate', 'free_gift_participation_rate',
                                        'dq_and_jq_pay_participation_rate', 'jq_pay_participation_rate',
                                        'dq_pay_participation_rate', 'log_synthetic_before_prefr_amount', 'datediff',
                                        'datediff_square', 'datediff_square_root', 'week_of_year_fourier_sin_1',
                                        'week_of_year_fourier_cos_1', 'week_of_year_fourier_sin_2',
                                        'week_of_year_fourier_cos_2', 'week_of_year_fourier_sin_3',
                                        'week_of_year_fourier_cos_3', 'day_of_week_fourier_sin_1',
                                        'day_of_week_fourier_cos_1', 'day_of_week_fourier_sin_2',
                                        'day_of_week_fourier_cos_2', 'day_of_week_fourier_sin_3',
                                        'day_of_week_fourier_cos_3']
            self.X_SCHEMA_CID3 = ['sku_offer_discount_rate', 'suit_offer_discount_rate',
                                  'full_minus_offer_discount_rate', 'ghost_offer_discount_rate',
                                  'free_gift_discount_rate', 'dq_and_jq_pay_discount_rate',
                                  'sku_offer_participation_rate', 'suit_offer_participation_rate',
                                  'full_minus_offer_participation_rate', 'ghost_offer_participation_rate',
                                  'free_gift_participation_rate', 'dq_and_jq_pay_participation_rate', 'newyear',
                                  'springfestival', 'tombsweepingfestival', 'labourday', 'dragonboatfestival',
                                  'midautumnfestival', 'nationalday', 'h1111mark', 'h618mark', 'h1212mark',
                                  'week_of_year', 'week_of_year_fourier_sin_1', 'week_of_year_fourier_cos_1',
                                  'week_of_year_fourier_sin_2', 'week_of_year_fourier_cos_2',
                                  'week_of_year_fourier_sin_3', 'week_of_year_fourier_cos_3',
                                  'day_of_week_fourier_sin_1', 'day_of_week_fourier_cos_1', 'day_of_week_fourier_sin_2',
                                  'day_of_week_fourier_cos_2', 'day_of_week_fourier_sin_3', 'day_of_week_fourier_cos_3']
            self.Y_SCHEMA_CID3 = ['log_synthetic_before_prefr_amount']
            self.OUTPUT_KEYS_CID3 = ['date', 'item_third_cate_cd']
            self.OUTPUT_VALUES_CID3 = ['final_baseline']
            self.OUTPUT_SCHEMA_CID3 = ['date', 'item_third_cate_cd', 'final_baseline']
            self.cid3_promotion_features = ['sku_offer_discount_rate', 'suit_offer_discount_rate',
                                            'full_minus_offer_discount_rate', 'ghost_offer_discount_rate',
                                            'free_gift_discount_rate', 'dq_and_jq_pay_discount_rate',
                                            'sku_offer_participation_rate', 'suit_offer_participation_rate',
                                            'full_minus_offer_participation_rate', 'ghost_offer_participation_rate',
                                            'free_gift_participation_rate', 'dq_and_jq_pay_participation_rate']
            '''
            新增常量LOCAL_SCHEMA_CID3，用于将local_randomforest与local_elasticnet方法的结果统一
            '''
            self.LOCAL_SCHEMA_CID3 = ['date', 'item_third_cate_cd', 'initial_base_line']
            self.SPLIT_FESTIVAL = ['h1111mark', 'h618mark', 'h1212mark']
        self.dataset = pd.DataFrame(list(raw_data), columns=self.DATASET_SCHEMA_CID3).sort_values('date') \
            .drop_duplicates().reset_index(drop=True).fillna(0)


class baseline_model_sku:
    def baseline_smooth(self, initial_base_line):
        # 进行第一次平滑===结果中没有用到
        smoothed_initial_base_line = sql_ewma(initial_base_line)[0]
        # 计算周期性因子
        seasonality_factor = initial_base_line / initial_base_line.mean()
        # 计算剔除季节性因子的真实销量
        deseasonalized_unit_volume = self.y_df.values.ravel() / seasonality_factor
        # 促销日、高销量日以及缺货日销量替换为初始基线的均值
        psrb = initial_base_line / seasonality_factor
        day_to_replace = np.where(
            (self.dataset['sale_qtty'] > self.cutpoint) | (self.dataset['out_of_stock_flag'] == 1), 0,
            self.dataset['non_promo_flag'])
        # 去季节性因素、异常天因素的销量
        deaseasonalized_non_promo_volume = np.where(day_to_replace == 1, deseasonalized_unit_volume, psrb)
        smoothed_deseasonalized_non_promo_unit_volume = sql_ewma(deaseasonalized_non_promo_volume)[0]
        # 最终基线结果
        final_base_line = smoothed_deseasonalized_non_promo_unit_volume * seasonality_factor
        return final_base_line

    def model_elasticnetcv(self):
        lm = ElasticNetCV(random_state=1)
        lm.fit(self.x_np, self.y_np)
        coefficients = lm.coef_
        intercept = lm.intercept_
        initial_fitting = lm.predict(self.x_df_train)
        initial_fitting = np.exp(initial_fitting) - self.moderator_
        model = codecs.encode(pickle.dumps(lm), "base64").decode()
        # # 无促销天的基线预测
        test_fitting = lm.predict(self.x_df_test)
        test_fitting = np.exp(test_fitting) - self.moderator_
        if 'sku_status_cd' in list(self.dataset_change.columns):
            test_fitting = test_fitting * self.dataset_change['sku_status_cd']
        if 'stock_status' in list(self.dataset_change.columns):
            test_fitting = adjust_out_of_stock(self.dataset_origin, self.dataset_change, test_fitting)
        r2_predict = r2_score(np.exp(self.y_df_train) - self.moderator_, initial_fitting)
        mse = mean_squared_error(np.exp(self.y_df_test) - self.moderator_, test_fitting)
        real = (np.exp(self.y_df_test)['target_qtty'].values - self.moderator_)
        pred = test_fitting
        r2_test = r2_score(np.exp(self.y_df_test) - self.moderator_, test_fitting)
        mape = np.mean(abs(pred - real) * 1.0 / np.maximum(1, real))
        return_data = self.dataset[self.dataset.test_flag == 1].copy()
        return_data['prediction'] = test_fitting

        # 计算模型拟合指标
        return_data['model_type'] = 'lm'
        return_data['r2_predict'] = r2_predict
        return_data['r2_test'] = r2_test
        return_data['mse'] = mse
        return_data['mape'] = mape
        feature_importance = [round(i, 4) for i in lm.coef_]
        return_data['model'] = model
        return_data['feature_importance'] = str(sorted(zip(feature_importance, self.feature), reverse=True))

        # 返回平滑后的基线
        return return_data

    def model_huberregressor(self):
        hr = HuberRegressor()
        hr.fit(self.x_np, self.y_np)
        coefficients = hr.coef_
        intercept = hr.intercept_
        initial_fitting = hr.predict(self.x_df_train)
        initial_fitting = np.exp(initial_fitting) - self.moderator_
        model = codecs.encode(pickle.dumps(hr), "base64").decode()
        # # 无促销天的基线预测
        test_fitting = hr.predict(self.x_df_test)
        test_fitting = np.exp(test_fitting) - self.moderator_
        if 'sku_status_cd' in list(self.dataset_change.columns):
            test_fitting = test_fitting * self.dataset_change['sku_status_cd']
        if 'stock_status' in list(self.dataset_change.columns):
            test_fitting = adjust_out_of_stock(self.dataset_origin, self.dataset_change, test_fitting)
        r2_predict = r2_score(np.exp(self.y_df_train) - self.moderator_, initial_fitting)
        mse = mean_squared_error(np.exp(self.y_df_test) - self.moderator_, test_fitting)
        real = (np.exp(self.y_df_test)['target_qtty'].values - self.moderator_)
        pred = test_fitting
        r2_test = r2_score(np.exp(self.y_df_test) - self.moderator_, test_fitting)
        mape = np.mean(abs(pred - real) * 1.0 / np.maximum(1, real))
        return_data = self.dataset[self.dataset.test_flag == 1].copy()
        return_data['prediction'] = test_fitting

        # 计算模型拟合指标
        return_data['model_type'] = 'hr'
        return_data['r2_predict'] = r2_predict
        return_data['r2_test'] = r2_test
        return_data['mse'] = mse
        return_data['mape'] = mape
        feature_importance = [round(i, 4) for i in hr.coef_]
        return_data['model'] = model
        return_data['feature_importance'] = str(sorted(zip(feature_importance, self.feature), reverse=True))

        # 返回平滑后的基线
        return return_data

    def model_randomforest(self):
        rf_model = RandomForestRegressor(n_estimators=200, criterion='mse', max_depth=10, oob_score=True,
                                         random_state=1)
        model_fit = rf_model.fit(self.x_np, self.y_np)
        initial_fitting = np.exp(model_fit.predict(self.x_df_train)) - self.moderator_
        model = codecs.encode(pickle.dumps(model_fit), "base64").decode()
        # 预测测试天的销量
        test_fitting = np.exp(model_fit.predict(self.x_df_test)) - self.moderator_
        if 'sku_status_cd' in list(self.dataset_change.columns):
            test_fitting = test_fitting * self.dataset_change['sku_status_cd']
        if 'stock_status' in list(self.dataset_change.columns):
            test_fitting = adjust_out_of_stock(self.dataset_origin, self.dataset_change, test_fitting)
        # test_fitting = test_fitting * self.dataset_change['sku_status_cd']
        # test_fitting = adjust_out_of_stock(self.dataset_origin, self.dataset_change, test_fitting)
        r2_predict = r2_score(np.exp(self.y_df_train) - self.moderator_, initial_fitting)
        mse = mean_squared_error(np.exp(self.y_df_test) - self.moderator_, test_fitting)
        real = (np.exp(self.y_df_test)['target_qtty'].values - self.moderator_)
        pred = test_fitting
        r2_test = r2_score(np.exp(self.y_df_test) - self.moderator_, test_fitting)
        mape = np.mean(abs(pred - real) * 1.0 / np.maximum(1, real))
        return_data = self.dataset[self.dataset.test_flag == 1].copy()
        return_data['prediction'] = test_fitting
        return_data['model_type'] = 'rf'
        return_data['r2_predict'] = r2_predict
        return_data['r2_test'] = r2_test
        return_data['mse'] = mse
        return_data['mape'] = mape
        feature_importance = rf_model.feature_importances_.tolist()
        feature_importance = [round(i, 4) for i in feature_importance]
        return_data['model'] = model
        return_data['feature_importance'] = str(sorted(zip(feature_importance, self.feature), reverse=True))
        # 返回平滑后的基线
        # 返回平滑后的基线
        return return_data
    
    def model_prophet(self):
        data = self.dataset.copy()
        data['ds'] = pd.to_datetime(data['dt'])
        data_training = data.query('test_flag == 0').copy()
        data_test = data.query('test_flag == 1').copy()

        # abnormal value
        q25 = np.percentile(data[self.Y[0]], 25)
        q75 = np.percentile(data[self.Y[0]], 75)
        cut_off = (q75 - q25) * 1.8
        lower = q25 - cut_off
        upper = q75 + cut_off
        data_training['target_qtty_adj'] = data_training[self.Y[0]]
        data_training.loc[(data_training[self.Y[0]] > upper) | (data_training[self.Y[0]] < lower),'target_qtty_adj'] = None

        # prophet_model
#         model_data = data_training.loc[:,['ds'] + self.feature + ['target_qtty_adj']].rename(columns={'target_qtty_adj':'y'})
        model_data = data_training.loc[:,['ds'] + ['target_price', 'target_days_flag'] + ['target_qtty_adj']].rename(columns={'target_qtty_adj':'y'})
        if len(model_data[model_data['y'].notnull()]) <10:
            raise Exception("effective y lack!")

        m = Prophet()
#         for i in self.feature:
        for i in ['target_price', 'target_days_flag']:
            m.add_regressor(i)
        m.fit(model_data)
        model = codecs.encode(pickle.dumps(m), "base64").decode()
#         future = data[['ds'] + self.feature].reset_index(drop=True)
        future = data[['ds'] + ['target_price', 'target_days_flag']].reset_index(drop=True)
        forecast = m.predict(future)
        data['prediction'] = forecast['yhat']
        r2_predict = r2_score(data[data.test_flag == 0]['target_qtty'], data[data.test_flag == 0]['prediction'])
        r2_test = r2_score(data[data.test_flag == 1]['target_qtty'], data[data.test_flag == 1]['prediction'])
        real = data[data.test_flag == 1]['target_qtty']
        pred = data[data.test_flag == 1]['prediction']
        mse = mean_squared_error(real, pred)
        mape = np.mean(abs(pred - real) * 1.0 / np.maximum(1, real))
        return_data = data[data.test_flag == 1].copy()
        return_data['model_type'] = 'prophet'
        return_data['r2_predict'] = r2_predict
        return_data['r2_test'] = r2_test
        return_data['mse'] = mse
        return_data['mape'] = mape
        return_data['model'] = model
        return_data['feature_importance'] = 'prophet'
        return return_data

    def model_xgboost(self):
        train_matrix = xgboost.DMatrix(self.x_np, self.y_np)
        train_x = xgboost.DMatrix(self.x_np)
        test_x = xgboost.DMatrix(self.x_df_test.values)
        # 根据特征，确认单调性，1表示单调递增，-1表示单调递减，0表示无单调约束
        constraint_list = [-1 if i == 'target_price' else 0 for i in self.feature]
        
#         for i in self.x_df.columns.values.tolist():
#             if (i in self.increase_constraint):
#                 constraint_list.append(1)
#             elif (i in self.decrease_constraint):
#                 constraint_list.append(-1)
#             else:
#                 constraint_list.append(0)
        # 最大深度是10，学习速率是0.7，拟合结果会非常好，会有一些过拟合
        params = {'eta': 0.1, 'max_depth': 4, 'min_child_weight': 3, 'seed': 0,'silent': 0,
                  'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.6, 'alpha': 0.05, 'lambda': 1,
                  'monotone_constraints':str(tuple(constraint_list))}
#         params = {'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0, 'n_estimators':10,
#                   'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,
#                   'monotone_constraints':str(tuple(constraint_list))}
        # 迭代次数过多容易过拟合
        model_fit = xgboost.train(params, train_matrix, num_boost_round=50)
        initial_fitting = np.exp(model_fit.predict(train_x)) - self.moderator_
        model = codecs.encode(pickle.dumps(model_fit), "base64").decode()
        # 预测测试天的销量
        test_fitting = np.exp(model_fit.predict(test_x)) - self.moderator_
        if 'sku_status_cd' in list(self.dataset_change.columns):
            test_fitting = test_fitting * self.dataset_change['sku_status_cd']
        if 'stock_status' in list(self.dataset_change.columns):
            test_fitting = adjust_out_of_stock(self.dataset_origin, self.dataset_change, test_fitting)

        r2_predict = r2_score(np.exp(self.y_df_train) - self.moderator_, initial_fitting)
        mse = mean_squared_error(np.exp(self.y_df_test) - self.moderator_, test_fitting)
        real = (np.exp(self.y_df_test)['target_qtty'].values - self.moderator_)
        pred = test_fitting
        r2_test = r2_score(np.exp(self.y_df_test) - self.moderator_, test_fitting)
        mape = np.mean(abs(pred - real) * 1.0 / np.maximum(1, real))
        return_data = self.dataset[self.dataset.test_flag == 1].copy()
        return_data['prediction'] = test_fitting
        return_data['model_type'] = 'xgboost'
        return_data['r2_predict'] = r2_predict
        return_data['r2_test'] = r2_test
        return_data['mse'] = mse
        return_data['mape'] = mape
#         feature_importance = 'xgboost'
#         feature_importance = [round(i, 4) for i in feature_importance]
        return_data['model'] = model
        return_data['feature_importance'] = 'xgboost'
        return return_data

    def __init__(self, model_list, sku_input, moderator_):
        self.moderator_ = moderator_
        self.promotion_features = sku_input.sku_promotion_features
        self.output_columns = sku_input.OUTPUT_VALUES_SKU
        self.feature = sku_input.X_SCHEMA_SKU
        self.Y = sku_input.Y_SCHEMA_SKU
        # 计算模型评估所需数据
        if sku_input.dataset[sku_input.dataset.test_flag == 1].shape[0] > 0:
            self.dataset = sku_input.dataset
            df_pd_mean = self.dataset[self.dataset.test_flag == 0][sku_input.ROLLING_SCHEMA].iloc[
                         -45:].median().values
            if self.dataset[self.dataset.test_flag == 1].shape[0] > 0:
                self.dataset_origin = self.dataset[self.dataset.test_flag == 0].copy().sort_index()
                self.dataset_change = self.dataset[self.dataset.test_flag == 1].copy().sort_index()
                self.dataset_change[sku_input.ROLLING_SCHEMA] = df_pd_mean
                self.dataset = pd.concat([self.dataset_origin, self.dataset_change]).sort_index()
            self.x_df_train = self.dataset[sku_input.X_SCHEMA_SKU][self.dataset.test_flag == 0].sort_index()
            self.x_df_test = self.dataset[sku_input.X_SCHEMA_SKU][self.dataset.test_flag == 1].sort_index()
            self.y_df_train = np.log(self.dataset[sku_input.Y_SCHEMA_SKU] + self.moderator_)[
                self.dataset.test_flag == 0].sort_index()
            self.y_df_test = np.log(self.dataset[sku_input.Y_SCHEMA_SKU] + self.moderator_)[self.dataset.test_flag == 1].sort_index()
            self.x_np = self.x_df_train.values
            self.y_np = self.y_df_train.values.ravel()
            # 多模型结果
            try:
                lr_model = [] 
                tree_model = []
                prophet_model = []
                if 'lr' in model_list:
                    lr_model.append(self.model_elasticnetcv())
                if 'hr' in model_list:
                    lr_model.append(self.model_huberregressor())
                if 'rf' in model_list:
                    tree_model.append(self.model_randomforest())
                if 'prophet' in model_list:
                    prophet_model.append(self.model_prophet())
                if 'xgboost' in model_list:
                    tree_model.append(self.model_xgboost())
                    # rf可能会拟合出非常大，甚至inf的数据，但是spark中对inf定义的范围要比python小，所以用float('inf')会出现卡不住的情况。
                if judge_feature_bound_rf(self.dataset_origin, self.dataset_change, sku_input.FEATURE_BOUND) & judge_feature_bound_lr(self.dataset_origin, self.dataset_change, sku_input.FEATURE_BOUND):
                    if len(prophet_model) == 0:
                        output_df = pd.concat(tree_model)
                    else:
                        output_df = pd.concat(prophet_model)
                elif judge_feature_bound_rf(self.dataset_origin, self.dataset_change, sku_input.FEATURE_BOUND):
                    output_df = pd.concat(lr_model + prophet_model)
                elif judge_feature_bound_lr(self.dataset_origin, self.dataset_change, sku_input.FEATURE_BOUND):
                    output_df = pd.concat(tree_model + prophet_model)
                else:
                    output_df = pd.concat(lr_model + tree_model + prophet_model)
                output_df['prediction'] = np.where(output_df['prediction'] >= 0, output_df['prediction'], 0)
                output_df = output_df[sku_input.OUTPUT_VALUES_SKU + sku_input.Y_SCHEMA_SKU]
            except:
                output_df = self.dataset[sku_input.OUTPUT_KEYS_SKU]
                for col in sku_input.OUTPUT_VALUES_SKU + sku_input.Y_SCHEMA_SKU:
                    output_df[col] = np.nan

#             lr_model = []
#             tree_model = []
#             prophet_model = []
#             if 'lr' in model_list:
#                 lr_model.append(self.model_elasticnetcv())
#             if 'hr' in model_list:
#                 lr_model.append(self.model_huberregressor())
#             if 'rf' in model_list:
#                 tree_model.append(self.model_randomforest())
#             if 'prophet' in model_list:
#                 prophet_model.append(self.model_prophet())
# #             if 'xgboost' in model_list:
# #                 tree_model.append(self.model_xgboost())
#             if judge_feature_bound_rf(self.dataset_origin, self.dataset_change, sku_input.FEATURE_BOUND) & judge_feature_bound_lr(self.dataset_origin, self.dataset_change, sku_input.FEATURE_BOUND):
#                 if len(prophet_model) == 0:
#                     output_df = pd.concat(tree_model)
#                 else:
#                     output_df = pd.concat(prophet_model)
#             elif judge_feature_bound_rf(self.dataset_origin, self.dataset_change, sku_input.FEATURE_BOUND):
#                 output_df = pd.concat(lr_model + prophet_model)
#             elif judge_feature_bound_lr(self.dataset_origin, self.dataset_change, sku_input.FEATURE_BOUND):
#                 output_df = pd.concat(tree_model + prophet_model)
# #                 output_df = tree_model
#             else:
#                 output_df = pd.concat(lr_model + tree_model + prophet_model)
#             output_df['prediction'] = np.where(output_df['prediction'] >= 0, output_df['prediction'], 0)
#             output_df = output_df[sku_input.OUTPUT_VALUES_SKU + sku_input.Y_SCHEMA_SKU]

#                 多模型结果

            self.output_df = output_df
        else:
            self.dataset = sku_input.dataset
            self.x_df_train = self.dataset[sku_input.X_SCHEMA_SKU]
            self.x_df_test = self.dataset[sku_input.X_SCHEMA_SKU]
            self.y_df_train = np.log(self.dataset[sku_input.Y_SCHEMA_SKU] + self.moderator_)
            self.y_df_test = np.log(self.dataset[sku_input.Y_SCHEMA_SKU] + self.moderator_)
            self.x_np = self.x_df_train.values
            self.y_np = self.y_df_train.values.ravel()
            # 多模型结果
            try:
                # 多模型结果
                lm_df = self.model_elasticnetcv()
                rf_df = self.model_randomforest()
                # rf可能会拟合出非常大，甚至inf的数据，但是spark中对inf定义的范围要比python小，所以用float('inf')会出现卡不住的情况。
                if (np.average(rf_df['prediction'].fillna(0)) >= 100000000):
                    output_df = lm_df
                else:
                    output_df = pd.concat([lm_df, rf_df])
            except:
                output_df = self.dataset[sku_input.OUTPUT_KEYS_SKU]
                for col in sku_input.OUTPUT_VALUES_SKU + sku_input.Y_SCHEMA_SKU:
                    output_df[col] = np.nan

            output_df['prediction'] = np.where(output_df['prediction'] >= 0, output_df['prediction'], 0)
            output_df = output_df[sku_input.OUTPUT_VALUES_SKU + sku_input.Y_SCHEMA_SKU]
            self.output_df = output_df