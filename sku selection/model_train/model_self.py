import datetime
import sys
import pandas as pd
import os
import numpy as np
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


import datetime
import sys
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,f1_score,precision_score,recall_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.spatial.distance import cosine
name = locals()
from pyspark.sql.window import Window


# os.environ['PYSPARK_PYTHON'] = 'python3.5'

spark = (SparkSession\
         .builder\
         .appName("spark_test")\
         .enableHiveSupport()\
         .getOrCreate())

spark.conf.set("hive.exec.dynamic.partition","true")
spark.conf.set("hive.exec.dynamic.partition.mode","nonstrict")

latest_nosplit_dt =spark.sql('''select max(dt) from app.app_pa_performance_nosplit_self''').collect()[0][0]

# transactions_D_self
latest_trans_d_dt =spark.sql('''select max(dt) from app.app_pa_transactions_D_self''').collect()[0][0]


time_style = '%Y-%m-%d'
df_last = datetime.datetime.strptime(latest_trans_d_dt, time_style)
# Promotion decision shall be made 10 days before promotion start date, and we download data in all 365 days previous of decision date
df_last_365 = df_last - datetime.timedelta(days=365)
latest_trans_d_dt_365 = datetime.datetime.strftime(df_last_365, time_style)

bu_id = latest_nosplit_dt # (bu_id= last partition of nosplit)

stage = sys.argv[1] #[1_stage, 2_stage]
target_type = sys.argv[2] # (coupon, manjianzhe, manjian, meimanjian,2_stage_manjianzhe,2_stage_manjian)

# stage = '2_stage' #[1_stage, 2_stage]
# bu_id = '2019-03-22' # (bu_id= last partition of nosplit)
# target_type = '2_stage_manjianzhe' # (coupon, manjianzhe, manjian, meimanjian,2_stage_manjianzhe,2_stage_manjian)

# 目标：2个代码走天下，分为book和self；
if stage == "1_stage":
	# 读促销信息表-----by changxin
	df = spark.sql('''select * from app.self_promo_train_data where bu_id = "%s" and type ="%s"'''%(bu_id,target_type)).select('promotion_id')

if stage == "2_stage":
	# 读促销信息表-----by changxin
	df = spark.sql('''select * from app.self_promo_train_data_2_stages where bu_id = "%s" and type ="%s"'''%(bu_id,target_type)).select('promotion_id')


feature_col_1_feature = ['10per_less_sale_qtty_0_30',
'10per_less_sale_qtty_0_90',
'10per_less_sale_qtty_0_365',
'10per_less_sale_qtty_90_365',
'10per_20per_sale_qtty_0_30',
'10per_20per_sale_qtty_0_90',
'10per_20per_sale_qtty_0_365',
'10per_20per_sale_qtty_90_365',
'20per_30per_sale_qtty_0_30',
'20per_30per_sale_qtty_0_90',
'20per_30per_sale_qtty_0_365',
'20per_30per_sale_qtty_90_365',
'30per_40per_sale_qtty_0_30',
'30per_40per_sale_qtty_0_90',
'30per_40per_sale_qtty_0_365',
'30per_40per_sale_qtty_90_365',
'40per_50per_sale_qtty_0_30',
'40per_50per_sale_qtty_0_90',
'40per_50per_sale_qtty_0_365',
'40per_50per_sale_qtty_90_365',
'50per_more_sale_qtty_0_30',
'50per_more_sale_qtty_0_90',
'50per_more_sale_qtty_0_365',
'50per_more_sale_qtty_90_365',
'10per_less_len_0_30',
'10per_less_len_0_90',
'10per_less_len_0_365',
'10per_less_len_90_365',
'10per_20per_len_0_30',
'10per_20per_len_0_90',
'10per_20per_len_0_365',
'10per_20per_len_90_365',
'20per_30per_len_0_30',
'20per_30per_len_0_90',
'20per_30per_len_0_365',
'20per_30per_len_90_365',
'30per_40per_len_0_30',
'30per_40per_len_0_90',
'30per_40per_len_0_365',
'30per_40per_len_90_365',
'40per_50per_len_0_30',
'40per_50per_len_0_90',
'40per_50per_len_0_365',
'40per_50per_len_90_365',
'50per_more_len_0_30',
'50per_more_len_0_90',
'50per_more_len_0_365',
'50per_more_len_90_365',
'10per_less_black_rate_0_30',
'10per_less_black_rate_0_90',
'10per_less_black_rate_0_365',
'10per_less_black_rate_90_365',
'10per_20per_black_rate_0_30',
'10per_20per_black_rate_0_90',
'10per_20per_black_rate_0_365',
'10per_20per_black_rate_90_365',
'20per_30per_black_rate_0_30',
'20per_30per_black_rate_0_90',
'20per_30per_black_rate_0_365',
'20per_30per_black_rate_90_365',
'30per_40per_black_rate_0_30',
'30per_40per_black_rate_0_90',
'30per_40per_black_rate_0_365',
'30per_40per_black_rate_90_365',
'40per_50per_black_rate_0_30',
'40per_50per_black_rate_0_90',
'40per_50per_black_rate_0_365',
'40per_50per_black_rate_90_365',
'50per_more_black_rate_0_30',
'50per_more_black_rate_0_90',
'50per_more_black_rate_0_365',
'50per_more_black_rate_90_365',
'10per_less_roi_0_30',
'10per_less_roi_0_90',
'10per_less_roi_0_365',
'10per_less_roi_90_365',
'10per_20per_roi_0_30',
'10per_20per_roi_0_90',
'10per_20per_roi_0_365',
'10per_20per_roi_90_365',
'20per_30per_roi_0_30',
'20per_30per_roi_0_90',
'20per_30per_roi_0_365',
'20per_30per_roi_90_365',
'30per_40per_roi_0_30',
'30per_40per_roi_0_90',
'30per_40per_roi_0_365',
'30per_40per_roi_90_365',
'40per_50per_roi_0_30',
'40per_50per_roi_0_90',
'40per_50per_roi_0_365',
'40per_50per_roi_90_365',
'50per_more_roi_0_30',
'50per_more_roi_0_90',
'50per_more_roi_0_365',
'50per_more_roi_90_365',
'10per_less_incre_0_30',
'10per_less_incre_0_90',
'10per_less_incre_0_365',
'10per_less_incre_90_365',
'10per_20per_incre_0_30',
'10per_20per_incre_0_90',
'10per_20per_incre_0_365',
'10per_20per_incre_90_365',
'20per_30per_incre_0_30',
'20per_30per_incre_0_90',
'20per_30per_incre_0_365',
'20per_30per_incre_90_365',
'30per_40per_incre_0_30',
'30per_40per_incre_0_90',
'30per_40per_incre_0_365',
'30per_40per_incre_90_365',
'40per_50per_incre_0_30',
'40per_50per_incre_0_90',
'40per_50per_incre_0_365',
'40per_50per_incre_90_365',
'50per_more_incre_0_30',
'50per_more_incre_0_90',
'50per_more_incre_0_365',
'50per_more_incre_90_365',
'his_uv_0_30',
'his_uv_0_90',
'his_uv_0_365',
'his_uv_90_365',
'his_red_price_0_30',
'his_red_price_0_90',
'his_red_price_0_365',
'his_red_price_90_365',
'his_baseprice_0_30',
'his_baseprice_0_90',
'his_baseprice_0_365',
'his_baseprice_90_365',
'consume_lim',
'cps_face_value',
'discount_rate_cal',
'red_price',
'baseprice',
'uv',
'sale_qtty',
'promo_days',
'day_of_week',
'day_of_year',
'week_of_year',
'tombsweepingfestival',
'dragonboatfestival',
'labourday',
'h618mark',
'midautumnfestival',
'h1212mark',
'h1111mark',
'newyear',
'springfestival',
'nationalday']


feature_col_2_feature = ['10per_less_sale_qtty_0_30',
'10per_less_sale_qtty_0_90',
'10per_less_sale_qtty_0_365',
'10per_less_sale_qtty_90_365',
'10per_less_len_0_30',
'10per_less_len_0_90',
'10per_less_len_0_365',
'10per_less_len_90_365',
'10per_less_black_rate_0_30',
'10per_less_black_rate_0_90',
'10per_less_black_rate_0_365',
'10per_less_black_rate_90_365',
'10per_less_roi_0_30',
'10per_less_roi_0_90',
'10per_less_roi_0_365',
'10per_less_roi_90_365',
'10per_less_incre_0_30',
'10per_less_incre_0_90',
'10per_less_incre_0_365',
'10per_less_incre_90_365',
'10per_20per_sale_qtty_0_30',
'10per_20per_sale_qtty_0_90',
'10per_20per_sale_qtty_0_365',
'10per_20per_sale_qtty_90_365',
'10per_20per_len_0_30',
'10per_20per_len_0_90',
'10per_20per_len_0_365',
'10per_20per_len_90_365',
'10per_20per_black_rate_0_30',
'10per_20per_black_rate_0_90',
'10per_20per_black_rate_0_365',
'10per_20per_black_rate_90_365',
'10per_20per_roi_0_30',
'10per_20per_roi_0_90',
'10per_20per_roi_0_365',
'10per_20per_roi_90_365',
'10per_20per_incre_0_30',
'10per_20per_incre_0_90',
'10per_20per_incre_0_365',
'10per_20per_incre_90_365',
'20per_30per_sale_qtty_0_30',
'20per_30per_sale_qtty_0_90',
'20per_30per_sale_qtty_0_365',
'20per_30per_sale_qtty_90_365',
'20per_30per_len_0_30',
'20per_30per_len_0_90',
'20per_30per_len_0_365',
'20per_30per_len_90_365',
'20per_30per_black_rate_0_30',
'20per_30per_black_rate_0_90',
'20per_30per_black_rate_0_365',
'20per_30per_black_rate_90_365',
'20per_30per_roi_0_30',
'20per_30per_roi_0_90',
'20per_30per_roi_0_365',
'20per_30per_roi_90_365',
'20per_30per_incre_0_30',
'20per_30per_incre_0_90',
'20per_30per_incre_0_365',
'20per_30per_incre_90_365',
'30per_40per_sale_qtty_0_30',
'30per_40per_sale_qtty_0_90',
'30per_40per_sale_qtty_0_365',
'30per_40per_sale_qtty_90_365',
'30per_40per_len_0_30',
'30per_40per_len_0_90',
'30per_40per_len_0_365',
'30per_40per_len_90_365',
'30per_40per_black_rate_0_30',
'30per_40per_black_rate_0_90',
'30per_40per_black_rate_0_365',
'30per_40per_black_rate_90_365',
'30per_40per_roi_0_30',
'30per_40per_roi_0_90',
'30per_40per_roi_0_365',
'30per_40per_roi_90_365',
'30per_40per_incre_0_30',
'30per_40per_incre_0_90',
'30per_40per_incre_0_365',
'30per_40per_incre_90_365',
'40per_50per_sale_qtty_0_30',
'40per_50per_sale_qtty_0_90',
'40per_50per_sale_qtty_0_365',
'40per_50per_sale_qtty_90_365',
'40per_50per_len_0_30',
'40per_50per_len_0_90',
'40per_50per_len_0_365',
'40per_50per_len_90_365',
'40per_50per_black_rate_0_30',
'40per_50per_black_rate_0_90',
'40per_50per_black_rate_0_365',
'40per_50per_black_rate_90_365',
'40per_50per_roi_0_30',
'40per_50per_roi_0_90',
'40per_50per_roi_0_365',
'40per_50per_roi_90_365',
'40per_50per_incre_0_30',
'40per_50per_incre_0_90',
'40per_50per_incre_0_365',
'40per_50per_incre_90_365',
'50per_more_sale_qtty_0_30',
'50per_more_sale_qtty_0_90',
'50per_more_sale_qtty_0_365',
'50per_more_sale_qtty_90_365',
'50per_more_len_0_30',
'50per_more_len_0_90',
'50per_more_len_0_365',
'50per_more_len_90_365',
'50per_more_black_rate_0_30',
'50per_more_black_rate_0_90',
'50per_more_black_rate_0_365',
'50per_more_black_rate_90_365',
'50per_more_roi_0_30',
'50per_more_roi_0_90',
'50per_more_roi_0_365',
'50per_more_roi_90_365',
'50per_more_incre_0_30',
'50per_more_incre_0_90',
'50per_more_incre_0_365',
'50per_more_incre_90_365',
'his_uv_0_30',
'his_uv_0_90',
'his_uv_0_365',
'his_uv_90_365',
'his_red_price_0_30',
'his_red_price_0_90',
'his_red_price_0_365',
'his_red_price_90_365',
'his_baseprice_0_30',
'his_baseprice_0_90',
'his_baseprice_0_365',
'his_baseprice_90_365',
'red_price',
'baseprice',
'uv',
'sale_qtty',
'promo_days',
'h618mark',
'h1212mark',
'springfestival',
'labourday',
'nationalday',
'dragonboatfestival',
'newyear',
'tombsweepingfestival',
'midautumnfestival',
'h1111mark',
'day_of_year',
'day_of_week',
'week_of_year',
'first_stage_discount_rate',
'second_stage_discount_rate',
'threshold_num_first',
'discount_rate_first',
'threshold_num_second',
'discount_rate_second']

if stage == "1_stage":
    feature_col_feature = feature_col_1_feature
if stage == "2_stage":
    feature_col_feature = feature_col_2_feature

feature_col_1 = ['item_sku_id','10per_less_sale_qtty_0_30',
'10per_less_sale_qtty_0_90',
'10per_less_sale_qtty_0_365',
'10per_less_sale_qtty_90_365',
'10per_20per_sale_qtty_0_30',
'10per_20per_sale_qtty_0_90',
'10per_20per_sale_qtty_0_365',
'10per_20per_sale_qtty_90_365',
'20per_30per_sale_qtty_0_30',
'20per_30per_sale_qtty_0_90',
'20per_30per_sale_qtty_0_365',
'20per_30per_sale_qtty_90_365',
'30per_40per_sale_qtty_0_30',
'30per_40per_sale_qtty_0_90',
'30per_40per_sale_qtty_0_365',
'30per_40per_sale_qtty_90_365',
'40per_50per_sale_qtty_0_30',
'40per_50per_sale_qtty_0_90',
'40per_50per_sale_qtty_0_365',
'40per_50per_sale_qtty_90_365',
'50per_more_sale_qtty_0_30',
'50per_more_sale_qtty_0_90',
'50per_more_sale_qtty_0_365',
'50per_more_sale_qtty_90_365',
'10per_less_len_0_30',
'10per_less_len_0_90',
'10per_less_len_0_365',
'10per_less_len_90_365',
'10per_20per_len_0_30',
'10per_20per_len_0_90',
'10per_20per_len_0_365',
'10per_20per_len_90_365',
'20per_30per_len_0_30',
'20per_30per_len_0_90',
'20per_30per_len_0_365',
'20per_30per_len_90_365',
'30per_40per_len_0_30',
'30per_40per_len_0_90',
'30per_40per_len_0_365',
'30per_40per_len_90_365',
'40per_50per_len_0_30',
'40per_50per_len_0_90',
'40per_50per_len_0_365',
'40per_50per_len_90_365',
'50per_more_len_0_30',
'50per_more_len_0_90',
'50per_more_len_0_365',
'50per_more_len_90_365',
'10per_less_black_rate_0_30',
'10per_less_black_rate_0_90',
'10per_less_black_rate_0_365',
'10per_less_black_rate_90_365',
'10per_20per_black_rate_0_30',
'10per_20per_black_rate_0_90',
'10per_20per_black_rate_0_365',
'10per_20per_black_rate_90_365',
'20per_30per_black_rate_0_30',
'20per_30per_black_rate_0_90',
'20per_30per_black_rate_0_365',
'20per_30per_black_rate_90_365',
'30per_40per_black_rate_0_30',
'30per_40per_black_rate_0_90',
'30per_40per_black_rate_0_365',
'30per_40per_black_rate_90_365',
'40per_50per_black_rate_0_30',
'40per_50per_black_rate_0_90',
'40per_50per_black_rate_0_365',
'40per_50per_black_rate_90_365',
'50per_more_black_rate_0_30',
'50per_more_black_rate_0_90',
'50per_more_black_rate_0_365',
'50per_more_black_rate_90_365',
'10per_less_roi_0_30',
'10per_less_roi_0_90',
'10per_less_roi_0_365',
'10per_less_roi_90_365',
'10per_20per_roi_0_30',
'10per_20per_roi_0_90',
'10per_20per_roi_0_365',
'10per_20per_roi_90_365',
'20per_30per_roi_0_30',
'20per_30per_roi_0_90',
'20per_30per_roi_0_365',
'20per_30per_roi_90_365',
'30per_40per_roi_0_30',
'30per_40per_roi_0_90',
'30per_40per_roi_0_365',
'30per_40per_roi_90_365',
'40per_50per_roi_0_30',
'40per_50per_roi_0_90',
'40per_50per_roi_0_365',
'40per_50per_roi_90_365',
'50per_more_roi_0_30',
'50per_more_roi_0_90',
'50per_more_roi_0_365',
'50per_more_roi_90_365',
'10per_less_incre_0_30',
'10per_less_incre_0_90',
'10per_less_incre_0_365',
'10per_less_incre_90_365',
'10per_20per_incre_0_30',
'10per_20per_incre_0_90',
'10per_20per_incre_0_365',
'10per_20per_incre_90_365',
'20per_30per_incre_0_30',
'20per_30per_incre_0_90',
'20per_30per_incre_0_365',
'20per_30per_incre_90_365',
'30per_40per_incre_0_30',
'30per_40per_incre_0_90',
'30per_40per_incre_0_365',
'30per_40per_incre_90_365',
'40per_50per_incre_0_30',
'40per_50per_incre_0_90',
'40per_50per_incre_0_365',
'40per_50per_incre_90_365',
'50per_more_incre_0_30',
'50per_more_incre_0_90',
'50per_more_incre_0_365',
'50per_more_incre_90_365',
'his_uv_0_30',
'his_uv_0_90',
'his_uv_0_365',
'his_uv_90_365',
'his_red_price_0_30',
'his_red_price_0_90',
'his_red_price_0_365',
'his_red_price_90_365',
'his_baseprice_0_30',
'his_baseprice_0_90',
'his_baseprice_0_365',
'his_baseprice_90_365',
'consume_lim',
'cps_face_value',
'discount_rate_cal',
'red_price',
'baseprice',
'uv',
'sale_qtty',
'promo_days',
'day_of_week',
'day_of_year',
'week_of_year',
'tombsweepingfestival',
'dragonboatfestival',
'labourday',
'h618mark',
'midautumnfestival',
'h1212mark',
'h1111mark',
'newyear',
'springfestival',
'nationalday',
'label']


feature_col_2 = ['item_sku_id','10per_less_sale_qtty_0_30',
'10per_less_sale_qtty_0_90',
'10per_less_sale_qtty_0_365',
'10per_less_sale_qtty_90_365',
'10per_less_len_0_30',
'10per_less_len_0_90',
'10per_less_len_0_365',
'10per_less_len_90_365',
'10per_less_black_rate_0_30',
'10per_less_black_rate_0_90',
'10per_less_black_rate_0_365',
'10per_less_black_rate_90_365',
'10per_less_roi_0_30',
'10per_less_roi_0_90',
'10per_less_roi_0_365',
'10per_less_roi_90_365',
'10per_less_incre_0_30',
'10per_less_incre_0_90',
'10per_less_incre_0_365',
'10per_less_incre_90_365',
'10per_20per_sale_qtty_0_30',
'10per_20per_sale_qtty_0_90',
'10per_20per_sale_qtty_0_365',
'10per_20per_sale_qtty_90_365',
'10per_20per_len_0_30',
'10per_20per_len_0_90',
'10per_20per_len_0_365',
'10per_20per_len_90_365',
'10per_20per_black_rate_0_30',
'10per_20per_black_rate_0_90',
'10per_20per_black_rate_0_365',
'10per_20per_black_rate_90_365',
'10per_20per_roi_0_30',
'10per_20per_roi_0_90',
'10per_20per_roi_0_365',
'10per_20per_roi_90_365',
'10per_20per_incre_0_30',
'10per_20per_incre_0_90',
'10per_20per_incre_0_365',
'10per_20per_incre_90_365',
'20per_30per_sale_qtty_0_30',
'20per_30per_sale_qtty_0_90',
'20per_30per_sale_qtty_0_365',
'20per_30per_sale_qtty_90_365',
'20per_30per_len_0_30',
'20per_30per_len_0_90',
'20per_30per_len_0_365',
'20per_30per_len_90_365',
'20per_30per_black_rate_0_30',
'20per_30per_black_rate_0_90',
'20per_30per_black_rate_0_365',
'20per_30per_black_rate_90_365',
'20per_30per_roi_0_30',
'20per_30per_roi_0_90',
'20per_30per_roi_0_365',
'20per_30per_roi_90_365',
'20per_30per_incre_0_30',
'20per_30per_incre_0_90',
'20per_30per_incre_0_365',
'20per_30per_incre_90_365',
'30per_40per_sale_qtty_0_30',
'30per_40per_sale_qtty_0_90',
'30per_40per_sale_qtty_0_365',
'30per_40per_sale_qtty_90_365',
'30per_40per_len_0_30',
'30per_40per_len_0_90',
'30per_40per_len_0_365',
'30per_40per_len_90_365',
'30per_40per_black_rate_0_30',
'30per_40per_black_rate_0_90',
'30per_40per_black_rate_0_365',
'30per_40per_black_rate_90_365',
'30per_40per_roi_0_30',
'30per_40per_roi_0_90',
'30per_40per_roi_0_365',
'30per_40per_roi_90_365',
'30per_40per_incre_0_30',
'30per_40per_incre_0_90',
'30per_40per_incre_0_365',
'30per_40per_incre_90_365',
'40per_50per_sale_qtty_0_30',
'40per_50per_sale_qtty_0_90',
'40per_50per_sale_qtty_0_365',
'40per_50per_sale_qtty_90_365',
'40per_50per_len_0_30',
'40per_50per_len_0_90',
'40per_50per_len_0_365',
'40per_50per_len_90_365',
'40per_50per_black_rate_0_30',
'40per_50per_black_rate_0_90',
'40per_50per_black_rate_0_365',
'40per_50per_black_rate_90_365',
'40per_50per_roi_0_30',
'40per_50per_roi_0_90',
'40per_50per_roi_0_365',
'40per_50per_roi_90_365',
'40per_50per_incre_0_30',
'40per_50per_incre_0_90',
'40per_50per_incre_0_365',
'40per_50per_incre_90_365',
'50per_more_sale_qtty_0_30',
'50per_more_sale_qtty_0_90',
'50per_more_sale_qtty_0_365',
'50per_more_sale_qtty_90_365',
'50per_more_len_0_30',
'50per_more_len_0_90',
'50per_more_len_0_365',
'50per_more_len_90_365',
'50per_more_black_rate_0_30',
'50per_more_black_rate_0_90',
'50per_more_black_rate_0_365',
'50per_more_black_rate_90_365',
'50per_more_roi_0_30',
'50per_more_roi_0_90',
'50per_more_roi_0_365',
'50per_more_roi_90_365',
'50per_more_incre_0_30',
'50per_more_incre_0_90',
'50per_more_incre_0_365',
'50per_more_incre_90_365',
'his_uv_0_30',
'his_uv_0_90',
'his_uv_0_365',
'his_uv_90_365',
'his_red_price_0_30',
'his_red_price_0_90',
'his_red_price_0_365',
'his_red_price_90_365',
'his_baseprice_0_30',
'his_baseprice_0_90',
'his_baseprice_0_365',
'his_baseprice_90_365',
'red_price',
'baseprice',
'uv',
'sale_qtty',
'promo_days',
'h618mark',
'h1212mark',
'springfestival',
'labourday',
'nationalday',
'dragonboatfestival',
'newyear',
'tombsweepingfestival',
'midautumnfestival',
'h1111mark',
'day_of_year',
'day_of_week',
'week_of_year',
'first_stage_discount_rate',
'second_stage_discount_rate',
'threshold_num_first',
'discount_rate_first',
'threshold_num_second',
'discount_rate_second',
'label']

if stage == "1_stage":
    feature_col = feature_col_1
if stage == "2_stage":
    feature_col = feature_col_2

# 思路：1阶和2阶分开写，if tage = 1_stage or 2_stage，再分阶梯满件折和阶梯满减；
# 分开满减，每满减，满件折 和 coupon；
if stage == '1_stage':
    df = spark.sql('''select * from dev.black_list_model_feature_self where bu_id = "%s" and type ="%s"'''%(bu_id,target_type))
    if target_type == 'manjian':
        df_current_info_finaly = df.fillna(0).drop('pricetime')
    if target_type == 'meimanjian':
        df_current_info_finaly = df.fillna(0).drop('pricetime')
    if target_type == 'coupon':
        df_current_info_finaly = df.fillna(0).drop('pricetime')
    if target_type == 'manjianzhe':
        df_current_info_finaly = df.fillna(0).drop('pricetime')
    df = df.withColumn('rank', F.col('item_sku_id').cast('int')%15)
    df_0 = df.filter(F.col('rank')==0).toPandas()
    df_1 = df.filter(F.col('rank')==1).toPandas()
    df_2 = df.filter(F.col('rank')==2).toPandas()
    df_3 = df.filter(F.col('rank')==3).toPandas()
    df_4 = df.filter(F.col('rank')==4).toPandas()
    df_5 = df.filter(F.col('rank')==5).toPandas()
    df_6 = df.filter(F.col('rank')==6).toPandas()
    df_7 = df.filter(F.col('rank')==7).toPandas()
    df_8 = df.filter(F.col('rank')==8).toPandas()
    df_9 = df.filter(F.col('rank')==9).toPandas()
    df_10 = df.filter(F.col('rank')==10).toPandas()
    df_11 = df.filter(F.col('rank')==11).toPandas()
    df_12 = df.filter(F.col('rank')==12).toPandas()
    df_13 = df.filter(F.col('rank')==13).toPandas()
    df_14 = df.filter(F.col('rank')==14).toPandas()
    df = pd.concat([df_0[feature_col],df_1[feature_col],df_2[feature_col],df_3[feature_col],\
                    df_4[feature_col],df_5[feature_col],df_6[feature_col],df_7[feature_col],\
                    df_8[feature_col],df_9[feature_col],df_10[feature_col],df_11[feature_col],\
                    df_12[feature_col],df_13[feature_col],df_14[feature_col]], axis= 0)

if stage == '2_stage':
    if target_type == '2_stage_manjian':
        df_promo = spark.sql('''select promo_id as promotion_id, threshold_money, discount_money, threshold_num, discount_num, discount_rate, add_money from app.self_total_promo_rule_detail
        where dt>'%s'
        and dt<='%s'
        and promo_subtype in (1,4)
        and threshold_money>0
        and discount_money>0
        and start_time>'%s'
        and end_time<='%s' '''%(latest_trans_d_dt_365, latest_trans_d_dt, latest_trans_d_dt_365, latest_trans_d_dt))
        df_promo = df_promo.distinct()
        df_final = df.join(df_promo, 'promotion_id', 'inner')
        df_final_1 = df_final.groupBy('promotion_id').agg(F.count('threshold_money').alias('promo_stage'))
        #筛选出阶梯满件折中：有两阶的促销
        df_final_2 = df_final_1.filter(F.col('promo_stage')==2)
        df_current_info = df_final_2.join(df_promo, 'promotion_id', 'left')
        w= Window.partitionBy(df_current_info.promotion_id).orderBy(df_current_info.threshold_money)
        # 默认升序
        rank = df_current_info.withColumn("threshold_money_rank", F.rank().over(w))
        # 此处的阶梯满减的first=一阶的门槛，second=二阶门槛
        rank_1 = rank.filter(F.col('threshold_money_rank') == 1).withColumnRenamed('threshold_money', 'threshold_num_first').withColumnRenamed('discount_money', 'discount_rate_first')
        rank_2 = rank.filter(F.col('threshold_money_rank') == 2).withColumnRenamed('threshold_money', 'threshold_num_second').withColumnRenamed('discount_money', 'discount_rate_second').drop('promo_stage')
        df_current_info_finaly_1 = rank_1.join(rank_2, 'promotion_id', 'inner')
        # 获取合适的promotion_id中的一阶和二阶促销信息；
        df_current_info_finaly_2 = df_current_info_finaly_1.select(['promotion_id', 'threshold_num_first', 'discount_rate_first', 'threshold_num_second', 'discount_rate_second', 'promo_stage'])
        df_history = spark.sql('''select * from dev.black_list_model_feature_self where bu_id = "%s" and type ="%s"'''%(bu_id,target_type))
        df_current_info_finaly = df_current_info_finaly_2.join(df_history, 'promotion_id', 'inner')
        df = df_current_info_finaly.fillna(0).drop('promo_stage', 'promotion_id', 'consume_lim', 'cps_face_value', 'discount_rate_cal','pricetime')
        df = df.withColumn('first_stage_discount_rate', F.col('discount_rate_first')/F.col('threshold_num_first'))\
        .withColumn('second_stage_discount_rate', F.col('discount_rate_second')/F.col('threshold_num_second'))
        df = df.withColumn('rank', F.col('item_sku_id').cast('int')%15)

    if target_type == '2_stage_manjianzhe':
        df_promo = spark.sql('''select promo_id as promotion_id, threshold_money, discount_money, threshold_num, discount_num, discount_rate, add_money from app.self_total_promo_rule_detail
        where dt>'%s'
        and dt<='%s'
        and promo_subtype=15
        and threshold_num>0
        and discount_rate>0
        and start_time>'%s'
        and end_time<='%s' '''%(latest_trans_d_dt_365, latest_trans_d_dt, latest_trans_d_dt_365, latest_trans_d_dt))
        df_promo = df_promo.distinct()
        df_final = df.join(df_promo, 'promotion_id', 'left')
        df_final_1 = df_final.groupBy('promotion_id').agg(F.count('threshold_num').alias('promo_stage'))
        #筛选出阶梯满件折中：有两阶的促销
        df_final_2 = df_final_1.filter(F.col('promo_stage')==2)
        df_current_info = df_final_2.join(df_promo, 'promotion_id', 'left')
        w= Window.partitionBy(df_current_info.promotion_id).orderBy(df_current_info.threshold_num)
#默认升序
        rank = df_current_info.withColumn("threshold_num_rank", F.rank().over(w))
# 此处first=一阶信息，second = 二阶信息；
        rank_1 = rank.filter(F.col('threshold_num_rank') == 1).withColumnRenamed('threshold_num', 'threshold_num_first').withColumnRenamed('discount_rate', 'discount_rate_first')
        rank_2 = rank.filter(F.col('threshold_num_rank') == 2).withColumnRenamed('threshold_num', 'threshold_num_second').withColumnRenamed('discount_rate', 'discount_rate_second').drop('promo_stage')
        df_current_info_finaly_1 = rank_1.join(rank_2, 'promotion_id', 'inner')
# 获取合适的promotion_id中的一阶和二阶促销信息；
        df_current_info_finaly_2 = df_current_info_finaly_1.select(['promotion_id', 'threshold_num_first', 'discount_rate_first', 'threshold_num_second', 'discount_rate_second', 'promo_stage'])
        df_history = spark.sql('''select * from dev.black_list_model_feature_self where bu_id = "%s" and type ="%s"'''%(bu_id,target_type))
        df_current_info_finaly = df_current_info_finaly_2.join(df_history, 'promotion_id', 'inner')
        df = df_current_info_finaly.fillna(0).drop('promo_stage', 'promotion_id', 'consume_lim', 'cps_face_value', 'discount_rate_cal','pricetime')
        df = df.withColumn('first_stage_discount_rate', F.col('discount_rate_first')).withColumn('second_stage_discount_rate', F.col('discount_rate_second'))
        df = df.withColumn('rank', F.col('item_sku_id').cast('int')%15)
    df_0 = df.filter(F.col('rank')==0).toPandas()
    df_1 = df.filter(F.col('rank')==1).toPandas()
    df_2 = df.filter(F.col('rank')==2).toPandas()
    df_3 = df.filter(F.col('rank')==3).toPandas()
    df_4 = df.filter(F.col('rank')==4).toPandas()
    df_5 = df.filter(F.col('rank')==5).toPandas()
    df_6 = df.filter(F.col('rank')==6).toPandas()
    df_7 = df.filter(F.col('rank')==7).toPandas()
    df_8 = df.filter(F.col('rank')==8).toPandas()
    df_9 = df.filter(F.col('rank')==9).toPandas()
    df_10 = df.filter(F.col('rank')==10).toPandas()
    df_11 = df.filter(F.col('rank')==11).toPandas()
    df_12 = df.filter(F.col('rank')==12).toPandas()
    df_13 = df.filter(F.col('rank')==13).toPandas()
    df_14 = df.filter(F.col('rank')==14).toPandas()
    df = pd.concat([df_0[feature_col],df_1[feature_col],df_2[feature_col],df_3[feature_col],\
                    df_4[feature_col],df_5[feature_col],df_6[feature_col],df_7[feature_col],\
                    df_8[feature_col],df_9[feature_col],df_10[feature_col],df_11[feature_col],\
                    df_12[feature_col],df_13[feature_col],df_14[feature_col]], axis= 0)
record = pd.DataFrame(columns=['model','precision','recall','black_count', 'black_rate'])

base_training,base_test,training_x_df,test_x_df,training_y_df,test_y_df = train_test_split(df,df[feature_col_feature],df['label'],test_size=0.2,random_state=1)
# 模型拟合
rf_model = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=10, oob_score=True,max_features='sqrt',random_state=1)
model_fit_noweight = rf_model.fit(training_x_df,training_y_df)

feature_importance = pd.DataFrame(list(zip(training_x_df.columns,model_fit_noweight.feature_importances_)),columns=['name','importance'])
feature_importance.sort_values(by='importance',ascending=False)

# 随机森林结果
threshold = 0.7

predicted_proba = model_fit_noweight.predict_proba(test_x_df)
predicted = (predicted_proba [:,1] >= threshold).astype('int')

fitting_test_df = pd.DataFrame(predicted).rename(columns={0:'fit'})
tt_y = pd.DataFrame(list(base_test['item_sku_id'])).rename(columns={0:'fact'})
tt = pd.concat([tt_y,fitting_test_df],axis=1)
black_rate = sum(df['label'])/len(df)
record.loc[0] = ['RF_7',precision_score(test_y_df, predicted),recall_score(test_y_df, predicted),sum(predicted),black_rate]

# 随机森林结果
threshold = 0.6

predicted_proba = model_fit_noweight.predict_proba(test_x_df)
predicted = (predicted_proba [:,1] >= threshold).astype('int')

fitting_test_df = pd.DataFrame(predicted).rename(columns={0:'fit'})
tt_y = pd.DataFrame(list(base_test['item_sku_id'])).rename(columns={0:'fact'})
tt = pd.concat([tt_y,fitting_test_df],axis=1)
black_rate = sum(df['label'])/len(df)
record.loc[1] = ['RF_6',precision_score(test_y_df, predicted),recall_score(test_y_df, predicted),sum(predicted),black_rate]

record = spark.createDataFrame(record)
record = record.withColumn('bu_id', F.lit(bu_id)).withColumn('type',F.lit(target_type))
record_col =['model', 'precision', 'recall', 'black_count', 'black_rate', 'bu_id', 'type']
table_name = 'dev.dev_black_selection_model_record_self'
record.select(record_col).write.insertInto(table_name, overwrite=True)


from sklearn.externals import joblib
joblib.dump(model_fit_noweight, "self_%s_rf_model.m"%(target_type))

import upload_to_oss
upload_to_oss.model_upload('self_%s_rf_model.m'%(target_type),latest_nosplit_dt)


# import os
# os.system('hadoop fs -put self_%s_rf_model.m /user/mart_rmb/user/xiaoxiao10/black_list_model/edition_%s'%(target_type,bu_id))
