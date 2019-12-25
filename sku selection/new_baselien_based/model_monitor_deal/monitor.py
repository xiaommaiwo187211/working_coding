import os
# os.environ['PYSPARK_PYTHON'] = 'python3.5'

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import datetime
from pyspark.sql import Window
from pyspark.sql.types import *
import sys

app_name = 'selection bu batch100'
spark = SparkSession.builder.appName(app_name).enableHiveSupport().getOrCreate()

spark.sql("set hive.exec.dynamic.partition=true")
spark.sql("set hive.exec.dynamic.partition.mode=nonstrict")

path_now = '/user/mart_rmb/user/changxin/selection/'

import datetime
import sys
import pandas as pd
import os
import numpy as np

# 1ST：监控筛选的promotion+sku和promotion，两次分区的数量差别；
df = spark.sql(''' select bu_id, type from dev.black_list_model_feature_self''')
df_dt = spark.sql('''select bu_id from dev.black_list_model_feature_self''').distinct().orderBy('bu_id', ascending = False)
df_dt_1 = df_dt.collect()[0][0]
df_dt_2 = df_dt.collect()[1][0]
depart = 'self'
self_type_list = ['manjian','meimanjian', 'manjianzhe','coupon', '2_stage_manjian','2_stage_manjianzhe']
#self_type_list = ['manjian']
record_feature = pd.DataFrame(columns=['detail_num_1','detail_num_2','detail_num_1_dis','promo_num_1','promo_num_2', 'detail_diff_rate_1_2', \
                             'detail_dis_diff_1', 'promo_diff_rate', 'type', 'dt_1', 'dt_2', 'status','department'])
i=0
for x in self_type_list:
    try:
        df_detail_1 = spark.sql(''' select * from dev.black_list_model_feature_self where bu_id = "%s" and type = "%s" '''%(df_dt_1, x))
        df_detail_2 = spark.sql(''' select * from dev.black_list_model_feature_self where bu_id = "%s" and type = "%s" '''%(df_dt_2, x))
    # 所有数据条数 and for distinct
        df_detail_1_num = df_detail_1.count()
        df_detail_1_num_dis = df_detail_1.distinct().count()
        df_detail_2_num = df_detail_2.count()
    # distinct promotion_id条数
        df_promotion_1_num = df_detail_1.select('promotion_id').distinct().count()
        df_promotion_2_num = df_detail_2.select('promotion_id').distinct().count()
        diff_1 = abs(df_detail_1_num - df_detail_2_num)/df_detail_1_num
        diff_2 = abs(df_detail_1_num - df_detail_1_num_dis)/df_detail_1_num
        diff_3 = abs(df_promotion_1_num - df_promotion_2_num)/df_promotion_1_num
        if (diff_1<0.05) & (diff_2<0.05) & (diff_3<0.05):
            warn = 'CLEAR'
        else:
            warn = 'WARN'
        record_feature.loc[i,] = [df_detail_1_num, df_detail_2_num, df_detail_1_num_dis, df_promotion_1_num, df_promotion_2_num,diff_1,diff_2,diff_3,x,df_dt_1,df_dt_2,warn,depart]
        i = i+1
    except:
        continue

feature_col = ['detail_num_1', 'detail_num_2', 'detail_num_1_dis', 'promo_num_1', 'promo_num_2', 'detail_diff_rate_1_2',\
 'detail_dis_diff_1',  'promo_diff_rate', 'dt_1', 'dt_2', 'status', 'type', 'department']

# 1ST：监控筛选的promotion+sku和promotion，两次分区的数量差别；
df = spark.sql(''' select bu_id, type from dev.black_list_model_feature_books''')
df_dt = spark.sql('''select bu_id from dev.black_list_model_feature_books''').distinct().orderBy('bu_id', ascending = False)
df_dt_1 = df_dt.collect()[0][0]
df_dt_2 = df_dt.collect()[1][0]
depart = 'book'
self_type_list = ['meimanjian','coupon']
#self_type_list = ['manjian']
record_books_feature = pd.DataFrame(columns=['detail_num_1','detail_num_2','detail_num_1_dis','promo_num_1','promo_num_2', 'detail_diff_rate_1_2', \
                             'detail_dis_diff_1', 'promo_diff_rate', 'type', 'dt_1', 'dt_2', 'status','department'])
i=0
for x in self_type_list:
    try:
        df_detail_1 = spark.sql(''' select * from dev.black_list_model_feature_books where bu_id = "%s" and type = "%s" '''%(df_dt_1, x))
        df_detail_2 = spark.sql(''' select * from dev.black_list_model_feature_books where bu_id = "%s" and type = "%s" '''%(df_dt_2, x))
    # 所有数据条数 and for distinct
        df_detail_1_num = df_detail_1.count()
        df_detail_1_num_dis = df_detail_1.distinct().count()
        df_detail_2_num = df_detail_2.count()
    # distinct promotion_id条数
        df_promotion_1_num = df_detail_1.select('promotion_id').distinct().count()
        df_promotion_2_num = df_detail_2.select('promotion_id').distinct().count()
        diff_1 = abs(df_detail_1_num - df_detail_2_num)/df_detail_1_num
        diff_2 = abs(df_detail_1_num - df_detail_1_num_dis)/df_detail_1_num
        diff_3 = abs(df_promotion_1_num - df_promotion_2_num)/df_promotion_1_num
        if (diff_1<0.05) & (diff_2<0.05) & (diff_3<0.05):
            warn = 'CLEAR'
        else:
            warn = 'WARN'
        record_books_feature.loc[i,] = [df_detail_1_num, df_detail_2_num, df_detail_1_num_dis, df_promotion_1_num, df_promotion_2_num,diff_1,diff_2,diff_3,x,df_dt_1,df_dt_2,warn,depart]
        i = i+1
    except:
        continue

# 模型准确性（自营非图书）：
self_type_list = ['manjian','meimanjian', 'manjianzhe','coupon', '2_stage_manjian','2_stage_manjianzhe']
model_list =['RF_6', 'RF_7']
#self_type_list = ['manjian']
depart = 'self'
record_model = pd.DataFrame(columns=['model','precision_1','recall_1','black_count_1','precision_2','recall_2','black_count_2',\
                               'precision_diff','recall_diff', 'num_diff','status','dt_1', 'dt_2','type','department'])

df = spark.sql(''' select bu_id, type from dev.dev_black_selection_model_record_self''')
df_dt = spark.sql('''select bu_id from dev.dev_black_selection_model_record_self''').distinct().orderBy('bu_id', ascending = False)
df_dt_1 = df_dt.collect()[0][0]
df_dt_2 = df_dt.collect()[1][0]


i=0
for x in self_type_list:
    for y in model_list:
        try:
            df_detail_1 = spark.sql(''' select * from dev.dev_black_selection_model_record_self where model = '%s' and bu_id = "%s" and type = "%s" '''%(y, df_dt_1, x))
            df_detail_2 = spark.sql(''' select * from dev.dev_black_selection_model_record_self where model = '%s' and bu_id = "%s" and type = "%s" '''%(y, df_dt_2, x))
    # 所有数据条数 and for distinct
            df_detail_model_1 = df_detail_1.select('model').collect()[0][0]
            df_detail_precision_1 = df_detail_1.select('precision').collect()[0][0]
            df_detail_recall_1 = df_detail_1.select('recall').collect()[0][0]
            df_detail_black_count_1 = df_detail_1.select('black_count').collect()[0][0]
            df_detail_dt_1 = df_detail_1.select('bu_id').collect()[0][0]
            df_detail_type_1 = df_detail_1.select('type').collect()[0][0]
            df_detail_model_2 = df_detail_2.select('model').collect()[0][0]
            df_detail_precision_2 = df_detail_2.select('precision').collect()[0][0]
            df_detail_recall_2 = df_detail_2.select('recall').collect()[0][0]
            df_detail_black_count_2 = df_detail_2.select('black_count').collect()[0][0]
            df_detail_dt_2 = df_detail_2.select('bu_id').collect()[0][0]
            df_detail_type_2 = df_detail_2.select('type').collect()[0][0]

            df_precision_diff = (df_detail_precision_1 - df_detail_precision_2)/df_detail_precision_2
            df_recall_diff = (df_detail_recall_1-df_detail_recall_2)/df_detail_recall_2
            df_black_count_diff = (df_detail_black_count_1-df_detail_black_count_2)/df_detail_black_count_2
            if (df_precision_diff<0.05) & (df_recall_diff<0.05) & (df_black_count_diff<0.05):
                warn = "CLEAR"
            else:
                warn = "WARN"
            record_model.loc[i,] = [df_detail_model_1, df_detail_precision_1, df_detail_recall_1, df_detail_black_count_1,\
                          df_detail_precision_2, df_detail_recall_2, df_detail_black_count_2,\
                         df_precision_diff, df_recall_diff, df_black_count_diff, warn, df_dt_1, df_dt_2, x,depart]
            i=i+1
        except:
            continue

# 模型准确性（自营图书）：
self_type_list = ['meimanjian','coupon']
model_list =['RF_6', 'RF_7']
#self_type_list = ['manjian']
depart = 'self'
record_books_model = pd.DataFrame(columns=['model','precision_1','recall_1','black_count_1','precision_2','recall_2','black_count_2',\
                               'precision_diff','recall_diff', 'num_diff','status','dt_1', 'dt_2','type','department'])

df = spark.sql(''' select bu_id, type from dev.dev_black_selection_model_record_book_self''')
df_dt = spark.sql('''select bu_id from dev.dev_black_selection_model_record_book_self''').distinct().orderBy('bu_id', ascending = False)
df_dt_1 = df_dt.collect()[0][0]
df_dt_2 = df_dt.collect()[1][0]


i=0
for x in self_type_list:
    for y in model_list:
        try:
            df_detail_1 = spark.sql(''' select * from dev.dev_black_selection_model_record_book_self where model = '%s' and bu_id = "%s" and type = "%s" '''%(y, df_dt_1, x))
            df_detail_2 = spark.sql(''' select * from dev.dev_black_selection_model_record_book_self where model = '%s' and bu_id = "%s" and type = "%s" '''%(y, df_dt_2, x))
    # 所有数据条数 and for distinct
            df_detail_model_1 = df_detail_1.select('model').collect()[0][0]
            df_detail_precision_1 = df_detail_1.select('precision').collect()[0][0]
            df_detail_recall_1 = df_detail_1.select('recall').collect()[0][0]
            df_detail_black_count_1 = df_detail_1.select('black_count').collect()[0][0]
            df_detail_dt_1 = df_detail_1.select('bu_id').collect()[0][0]
            df_detail_type_1 = df_detail_1.select('type').collect()[0][0]
            df_detail_model_2 = df_detail_2.select('model').collect()[0][0]
            df_detail_precision_2 = df_detail_2.select('precision').collect()[0][0]
            df_detail_recall_2 = df_detail_2.select('recall').collect()[0][0]
            df_detail_black_count_2 = df_detail_2.select('black_count').collect()[0][0]
            df_detail_dt_2 = df_detail_2.select('bu_id').collect()[0][0]
            df_detail_type_2 = df_detail_2.select('type').collect()[0][0]

            df_precision_diff = (df_detail_precision_1 - df_detail_precision_2)/df_detail_precision_2
            df_recall_diff = (df_detail_recall_1-df_detail_recall_2)/df_detail_recall_2
            df_black_count_diff = (df_detail_black_count_1-df_detail_black_count_2)/df_detail_black_count_2
            if (df_precision_diff<0.05) & (df_recall_diff<0.05) & (df_black_count_diff<0.05):
                warn = "CLEAR"
            else:
                warn = "WARN"
            record_books_model.loc[i,] = [df_detail_model_1, df_detail_precision_1, df_detail_recall_1, df_detail_black_count_1,\
                          df_detail_precision_2, df_detail_recall_2, df_detail_black_count_2,\
                         df_precision_diff, df_recall_diff, df_black_count_diff, warn, df_dt_1, df_dt_2, x,depart]
            i=i+1
        except:
            continue





model_col = ['model','precision_1','recall_1','black_count_1','precision_2','recall_2','black_count_2','precision_diff','recall_diff','num_diff','status','dt_1','dt_2','type','department']

spark.createDataFrame(record_books_model[model_col]).write.insertInto('dev.dev_black_list_model_monitor',overwrite = True)
spark.createDataFrame(record_model[model_col]).write.insertInto('dev.dev_black_list_model_monitor',overwrite = True)
spark.createDataFrame(record_books_feature[feature_col]).write.insertInto('dev.dev_black_list_features_monitor',overwrite = True)
spark.createDataFrame(record_feature[feature_col]).write.insertInto('dev.dev_black_list_features_monitor',overwrite = True)