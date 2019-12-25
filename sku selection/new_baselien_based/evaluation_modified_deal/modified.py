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
import os

# os.environ['PYSPARK_PYTHON'] = 'python3.5'
#
# spark = (SparkSession         .builder         .appName("spark_test")         .enableHiveSupport()         .config("spark.executor.instances", "100")         .config("spark.executor.memory","16g")         .config("spark.executor.cores","4")         .config("spark.driver.memory","16g")         .config("spark.sql.shuffle.partitions","800")         .config("spark.default.parallelism","800")         .config("spark.driver.maxResultSize", "8g")         .config("spark.pyspark.python", "python3.5")         .config("spark.yarn.appMasterEnv.yarn.nodemanager.container-executor.class","DockerLinuxContainer")         .config("spark.executorEnv.yarn.nodemanager.container-executor.class","DockerLinuxContainer")         .config("spark.yarn.appMasterEnv.yarn.nodemanager.docker-container-executor.image-name","bdp-docker.jd.com:5000/wise_mart_rmb:latest")         .config("spark.executorEnv.yarn.nodemanager.docker-container-executor.image-name","bdp-docker.jd.com:5000/wise_mart_rmb:latest")         .getOrCreate())

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
from datetime import timedelta
import time

model_edition = sys.argv[1]
# model_edition = "4d9bbfd1"

df = spark.sql('''select * from dev.dev_black_selection_model_record_self''')

dict = {'coupon':'优惠券','manjian':'满减','meimanjian':'每满减','manjianzhe':'满件折','2_stage_manjian':'阶梯满减','2_stage_manjianzhe':'阶梯满件折',\
       'RF_7' :'随机森林:黑名单概率>70%','RF_6':'随机森林:黑名单概率>60%'}

col_name = ['model', 'precision', 'recall', 'black_count', 'black_rate', 'test_num', 'bu_id_duration','type_1', 'git_edition', 'bu_id', 'type']

df_pandas = df.toPandas()

df_pandas['type_1'] = df_pandas['type'].map(lambda x:dict[x])
df_pandas['model'] = df_pandas['model'].map(lambda x:dict[x])
df_pandas['bu_id_duration'] = df_pandas['bu_id']\
.map(lambda x: datetime.datetime.strftime(datetime.datetime.strptime(x,"%Y-%m-%d")- datetime.timedelta(days=365),"%Y-%m-%d")+' to '+x)
df_pandas['test_num'] = df_pandas['precision']*df_pandas['black_count']/df_pandas['recall']/df_pandas['black_rate']/0.2
df_pandas['git_edition'] = model_edition
df_spark = spark.createDataFrame(df_pandas)[col_name]
df_spark.write.insertInto('dev.dev_black_selection_model_record_self_modified',overwrite = True)


df = spark.sql('''select * from dev.dev_black_selection_model_record_book_self''')
dict = {'coupon':'优惠券','manjian':'满减','meimanjian':'每满减','manjianzhe':'满件折','2_stage_manjian':'阶梯满减','2_stage_manjianzhe':'阶梯满件折',\
       'RF_7' :'随机森林:黑名单概率>70%','RF_6':'随机森林:黑名单概率>60%'}
df_pandas = df.toPandas()
df_pandas['type_1'] = df_pandas['type'].map(lambda x:dict[x])
df_pandas['model'] = df_pandas['model'].map(lambda x:dict[x])
df_pandas['bu_id_duration'] = df_pandas['bu_id']\
.map(lambda x: datetime.datetime.strftime(datetime.datetime.strptime(x,"%Y-%m-%d")- datetime.timedelta(days=365),"%Y-%m-%d")+' to '+x)
df_spark = spark.createDataFrame(df_pandas).drop('type_1')
df_pandas['test_num'] = df_pandas['precision']*df_pandas['black_count']/df_pandas['recall']/df_pandas['black_rate']/0.2
df_pandas['git_edition'] = model_edition
df_spark = spark.createDataFrame(df_pandas)[col_name]
df_spark.write.insertInto('dev.dev_black_selection_model_record_book_self_modified',overwrite = True)