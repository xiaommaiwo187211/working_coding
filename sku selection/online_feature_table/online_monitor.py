
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
# coding:utf-8
import sys
import os
import pandas as pd
import numpy as np
import datetime
import smtplib
import upload_to_oss
from sklearn.externals import joblib
from pyspark.sql import SparkSession
from email.mime.text import MIMEText
from email.header import Header
import pyspark.sql.functions as F
from pyspark import SparkFiles
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
import pyspark.sql.types as sql_type
from pyspark.sql.window import Window
from pyspark.sql.types import DecimalType
spark = SparkSession         .builder         .appName("spark_test")         .enableHiveSupport()         .getOrCreate()


spark.conf.set("hive.exec.dynamic.partition","true")
spark.conf.set("hive.exec.dynamic.partition.mode","nonstrict")
#app_name='yxd_task'
spark.sql("set hive.exec.orc.split.strategy=ETL")
name = locals()


#=======================下游数据表========================
sku_type_list = ['self','book']
lib_name = 'dev'
location = ['hdfs://ns7/user/mart_scr/%s.db/'%lib_name,'hdfs://ns11/user/mart_rmb/%s.db/'%lib_name]
location_name = list(map(lambda x:'dev_empty_bottle_black_selection_'+x+'_self' if 'book' in x else 'dev_empty_bottle_black_selection_'+x,sku_type_list))
table_name = list(map(lambda x:lib_name+'.'+x,location_name))
promo_type = 'meimanjian'
model_type = 'rf_model'
result = pd.DataFrame(columns=['description','sku_type','old','new','diff','result','dt'])


# ### 二、新更新分区不能有重复记录

# In[ ]:


# 有的表中有重复数据
for i in range(len(location_name)):
    yesterday = spark.sql('''select max(dt) from %s '''%table_name[i]).collect()[0][0]
    a = spark.sql('''select * from %s where dt = '%s' '''%(table_name[i],yesterday)).select('item_sku_id')
    table_count = a.count()
    table_distinct_count = a.distinct().count()
    name['duplicate_data_%s'%i] = ['重复数据',sku_type_list[i],float(table_count),float(table_distinct_count),abs(table_count-table_distinct_count)/table_count,
                                   '正常' if table_count-table_distinct_count==0 else '异常',yesterday]    
    result.loc[len(result)] = eval('duplicate_data_%s'%i)


# ### 三、新更新分区不能与上一分区的表大小差距过大

# In[ ]:


for i in range(len(location_name)):
    a = os.popen('hadoop fs -du '+ location[i] + location_name[i]).read()
    b = a.split('\n')[:-1]
    c = pd.DataFrame(b)
    d = c[0].str.split('hdfs', expand=True)
    if 'SUCCESS' in d.iloc[0,1]:
        d = d.tail(len(d)-1).reset_index(drop=True)
    d[0] = d[0].str.strip().map(int)  
    if len(d)<=1:
        name['table_size_%s'%i] = ['分区大小',sku_type_list[i],-9999,float(d[0].iloc[-1]),-9999,'正常', yesterday]
    else:
        name['table_size_%s'%i] = ['分区大小',sku_type_list[i],float(d[0].iloc[-2]),float(d[0].iloc[-1]),abs(d[0].iloc[-2]-d[0].iloc[-1])/d[0].iloc[-2],
                                    '正常' if abs(d[0].iloc[-2]-d[0].iloc[-1])/d[0].iloc[-2]<=0.1 else '异常', yesterday]
    result.loc[len(result)] = eval('table_size_%s'%i)


# ### 四、新更新分区与上一分区使用同一模型结果差距不会很大
# 前后两个分区使用同样的本次促销特征，同样的上周模型，实际最近的上周模型中的20个promotion_id对应的本次促销信息  
# 1、总黑名单率相差不大 10%以内  
# 2、产生状态变化的sku占总sku的10%以内

# In[ ]:


total_feature_col_1 = ['10per_less_sale_qtty_0_30','10per_less_sale_qtty_0_90','10per_less_sale_qtty_0_365','10per_less_sale_qtty_90_365',
'10per_20per_sale_qtty_0_30','10per_20per_sale_qtty_0_90','10per_20per_sale_qtty_0_365','10per_20per_sale_qtty_90_365','20per_30per_sale_qtty_0_30',
'20per_30per_sale_qtty_0_90','20per_30per_sale_qtty_0_365','20per_30per_sale_qtty_90_365','30per_40per_sale_qtty_0_30','30per_40per_sale_qtty_0_90',
'30per_40per_sale_qtty_0_365','30per_40per_sale_qtty_90_365','40per_50per_sale_qtty_0_30','40per_50per_sale_qtty_0_90',
'40per_50per_sale_qtty_0_365','40per_50per_sale_qtty_90_365','50per_more_sale_qtty_0_30','50per_more_sale_qtty_0_90',
'50per_more_sale_qtty_0_365','50per_more_sale_qtty_90_365','10per_less_len_0_30','10per_less_len_0_90',
'10per_less_len_0_365','10per_less_len_90_365','10per_20per_len_0_30','10per_20per_len_0_90',
'10per_20per_len_0_365','10per_20per_len_90_365','20per_30per_len_0_30','20per_30per_len_0_90',
'20per_30per_len_0_365','20per_30per_len_90_365','30per_40per_len_0_30','30per_40per_len_0_90',
'30per_40per_len_0_365','30per_40per_len_90_365','40per_50per_len_0_30','40per_50per_len_0_90',
'40per_50per_len_0_365','40per_50per_len_90_365','50per_more_len_0_30','50per_more_len_0_90',
'50per_more_len_0_365','50per_more_len_90_365','10per_less_black_rate_0_30','10per_less_black_rate_0_90',
'10per_less_black_rate_0_365','10per_less_black_rate_90_365','10per_20per_black_rate_0_30','10per_20per_black_rate_0_90',
'10per_20per_black_rate_0_365','10per_20per_black_rate_90_365','20per_30per_black_rate_0_30','20per_30per_black_rate_0_90',
'20per_30per_black_rate_0_365','20per_30per_black_rate_90_365','30per_40per_black_rate_0_30','30per_40per_black_rate_0_90',
'30per_40per_black_rate_0_365','30per_40per_black_rate_90_365','40per_50per_black_rate_0_30','40per_50per_black_rate_0_90',
'40per_50per_black_rate_0_365','40per_50per_black_rate_90_365','50per_more_black_rate_0_30','50per_more_black_rate_0_90',
'50per_more_black_rate_0_365','50per_more_black_rate_90_365','10per_less_roi_0_30','10per_less_roi_0_90',
'10per_less_roi_0_365','10per_less_roi_90_365','10per_20per_roi_0_30','10per_20per_roi_0_90',
'10per_20per_roi_0_365','10per_20per_roi_90_365','20per_30per_roi_0_30','20per_30per_roi_0_90',
'20per_30per_roi_0_365','20per_30per_roi_90_365','30per_40per_roi_0_30','30per_40per_roi_0_90',
'30per_40per_roi_0_365','30per_40per_roi_90_365','40per_50per_roi_0_30','40per_50per_roi_0_90',
'40per_50per_roi_0_365','40per_50per_roi_90_365','50per_more_roi_0_30','50per_more_roi_0_90',
'50per_more_roi_0_365','50per_more_roi_90_365','10per_less_incre_0_30','10per_less_incre_0_90',
'10per_less_incre_0_365','10per_less_incre_90_365','10per_20per_incre_0_30','10per_20per_incre_0_90',
'10per_20per_incre_0_365','10per_20per_incre_90_365','20per_30per_incre_0_30','20per_30per_incre_0_90',
'20per_30per_incre_0_365','20per_30per_incre_90_365','30per_40per_incre_0_30','30per_40per_incre_0_90',
'30per_40per_incre_0_365','30per_40per_incre_90_365','40per_50per_incre_0_30','40per_50per_incre_0_90',
'40per_50per_incre_0_365','40per_50per_incre_90_365','50per_more_incre_0_30','50per_more_incre_0_90',
'50per_more_incre_0_365','50per_more_incre_90_365','his_uv_0_30','his_uv_0_90',
'his_uv_0_365','his_uv_90_365','his_red_price_0_30','his_red_price_0_90',
'his_red_price_0_365','his_red_price_90_365','his_baseprice_0_30','his_baseprice_0_90',
'his_baseprice_0_365','his_baseprice_90_365','consume_lim','cps_face_value',
'discount_rate_cal','red_price','baseprice','uv',
'sale_qtty','promo_days','day_of_week','day_of_year',
'week_of_year','tombsweepingfestival','dragonboatfestival','labourday',
'h618mark','midautumnfestival','h1212mark','h1111mark',
'newyear','springfestival','nationalday']

cur_promo_feature = ['item_sku_id','promotion_id','consume_lim','cps_face_value',
'discount_rate_cal','red_price','baseprice',
'promo_days','day_of_week','day_of_year',
'week_of_year','tombsweepingfestival','dragonboatfestival','labourday',
'h618mark','midautumnfestival','h1212mark','h1111mark',
'newyear','springfestival','nationalday']


# In[ ]:


promo_num = 10
threshold = 0.65
for i in range(len(location_name)):
    # 读取本次促销信息
    model_table = 'dev.black_list_model_feature_%s'%('books' if 'book' in sku_type_list[i] else sku_type_list[i])
    model_day = spark.sql('''show partitions %s '''%model_table).collect()
    model_day = sorted(list(set(list(map(lambda x:x[0].split('/')[0].split('=')[1],model_day)))))[-2]
    model_feature = spark.sql('''select * from %s where bu_id ='%s' and type='%s' '''%(model_table,model_day,promo_type))
    cur_feature = model_feature.select(cur_promo_feature)
    # 取出覆盖300个sku以上的前10个promotion_id
    cur_promo_list = cur_feature.groupby('promotion_id').agg(F.countDistinct('item_sku_id').alias('sku_num')).toPandas()
    cur_promo_list = cur_promo_list.loc[cur_promo_list['sku_num']>=300].head(promo_num)['promotion_id']
    # 下载模型 文件
    upload_to_oss.model_download(sku_type_list[i],promo_type,model_type,model_day+'deal_price')
    model_rf = joblib.load(sku_type_list[i]+'_'+promo_type+'_'+model_type+'.m')
    yesterday_new = spark.sql('''select max(dt) from %s '''%table_name[i]).collect()[0][0]
    online_new = spark.sql('''select * from %s where dt = '%s' '''%(table_name[i],yesterday_new)).drop('promotion_id')
    yesterday_old =  spark.sql('''show partitions %s '''%table_name[i]).collect()
    if len(yesterday_old) <=1:
        name['sku_status_diff_%s'%i] = ['sku差异',sku_type_list[i],float(-9999),float(-9999),-9999,
                                       '正常' ,yesterday_new]
        result.loc[len(result)] = eval('sku_status_diff_%s'%i)            
    else:
        yesterday_old = sorted(list(set(list(map(lambda x:x[0].split('=')[1],yesterday_old)))))[-2]
        online_old = spark.sql('''select * from %s where dt = '%s' '''%(table_name[i],yesterday_old)).drop('promotion_id')
        online_sku = online_new.select('item_sku_id').join(online_old.select('item_sku_id'),'item_sku_id','inner')   
        new = online_new.join(online_sku,'item_sku_id','inner')
        old = online_old.join(online_sku,'item_sku_id','inner')    
        new_test = new.join(cur_feature.filter(F.col('promotion_id').isin(list(cur_promo_list))),'item_sku_id','inner').toPandas()
        old_test = old.join(cur_feature.filter(F.col('promotion_id').isin(list(cur_promo_list))),'item_sku_id','inner').toPandas()        
        new_predicted = (model_rf.predict_proba(new_test[total_feature_col_1])[:,1]>=threshold).astype('int')
        old_predicted = (model_rf.predict_proba(old_test[total_feature_col_1])[:,1]>=threshold).astype('int')
        new_result = pd.concat([new_test[['item_sku_id','promotion_id']],pd.DataFrame({'new_black_proba':new_predicted})],axis=1)
        old_result = pd.concat([old_test[['item_sku_id','promotion_id']],pd.DataFrame({'old_black_proba':old_predicted})],axis=1)
        black_result = pd.merge(new_result,old_result,on=['item_sku_id','promotion_id'],how='inner')
        # 新旧黑名单率
        new_black_rate = 1.0*len(black_result.loc[black_result['new_black_proba']==1])/len(black_result)
        old_black_rate = 1.0*len(black_result.loc[black_result['old_black_proba']==1])/len(black_result)
        # sku状态变化
        sku_status_diff = 1.0*len(black_result.loc[black_result['new_black_proba']!=black_result['old_black_proba']])/len(black_result)
        name['model_check_%s'%i] = ['黑名单率',sku_type_list[i],old_black_rate,new_black_rate,abs(new_black_rate-old_black_rate),
                                   '正常' if abs(new_black_rate-old_black_rate)<=0.1 else '异常', yesterday_new]
        result.loc[len(result)] = eval('model_check_%s'%i)
        name['sku_status_diff_%s'%i] = ['sku差异',sku_type_list[i],float(-9999),float(-9999),sku_status_diff,
                                       '正常' if sku_status_diff<=0.1 else '异常',yesterday_new]
        result.loc[len(result)] = eval('sku_status_diff_%s'%i)


# ### 四、结果保存

# In[ ]:


spark.createDataFrame(result).write.insertInto('dev.dev_empty_bottle_black_selection_online_monitor',overwrite=True)

