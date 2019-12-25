#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pyspark.sql import SparkSession
from pyspark import SparkFiles
from pyspark.sql.types import *
from pyspark.storagelevel import StorageLevel
from pyspark.sql import functions as F
import os
import datetime
name = locals()

spark = SparkSession.builder.appName("spark_test").enableHiveSupport().getOrCreate()

spark.conf.set("hive.exec.dynamic.partition","true")
spark.conf.set("hive.exec.dynamic.partition.mode","nonstrict")


# In[2]:


# 找最新分区，找最新分区的最大date
yesterday = spark.sql('''select max(dt) from app.app_pa_performance_nosplit_book_self ''').collect()[0][0]
# yesterday='2019-03-22'
end_dt = spark.sql('''select max(date) from app.app_pa_performance_nosplit_book_self where dt = '%s' '''%yesterday).collect()[0][0]
start_dt = (datetime.datetime.strptime(end_dt, '%Y-%m-%d') - datetime.timedelta(days=365)).strftime('%Y-%m-%d')

# 历史nosplit表信息
performance = spark.sql('''
select 
    item_sku_id,
    batch_id as his_batch_id,
    promotion_id as his_promotion_id,
    sale_qtty,
    uplift_sales,
    baseline_sales,
    halo,
    incremental_halo,
    promo_giveaway,
    coupon_giveaway,
    cann_sales,
    date as dt
from app.app_pa_performance_nosplit_book_self
where dt='%s'
and date >= '%s'
''' %(yesterday,start_dt))

# sku池子
df_sql_skus = performance.select('item_sku_id').distinct()                         .withColumn('promotion_id',F.lit('-9999'))                         .withColumn('min_dt',F.lit(end_dt))

# 读取SKU的最新上下柜状态信息
df_sql_sku_status = spark.sql('''
select 
    sku_id as item_sku_id,
    cast(sku_status_cd as int) as sku_status_cd,
    dt
from dev.self_sku_det_da
where sku_type in (2,3)
and dt>='%s'
'''%start_dt)
df_sql_sku_status = df_sql_sku_status.groupby('item_sku_id','dt').agg(F.max('sku_status_cd').alias('sku_status_cd'))

# 读取SKU的红价数据
df_sql_sku_price_scraped = spark.sql('''
select 
    sku_id as item_sku_id,
    last_price,
    dt
from dev.self_sku_redprice_group
where dt >= '%s'
'''%start_dt)
df_sql_sku_price_scraped = df_sql_sku_price_scraped.groupby('item_sku_id','dt').agg(F.mean('last_price').alias('red_price'))

# 读取SKU的基线价格
df_sql_sku_price_baseline = spark.sql('''
select 
    item_sku_id,
    baseprice,
    dt
from app.app_pa_price_baseprice_self
where dt >= '%s'
'''%start_dt)
df_sql_sku_price_baseline = df_sql_sku_price_baseline.groupby('item_sku_id','dt').agg(F.mean('baseprice').alias('baseprice'))

# 基线信息
df_baseline_history = spark.sql('''
select
    date as dt, 
    item_sku_id, 
    final_baseline
from app.app_pa_baseline_sku_book_self
where dt = '%s'
and date >='%s'
''' % (yesterday,start_dt))
df_baseline_history = df_baseline_history.groupby('item_sku_id','dt').agg(F.mean('final_baseline').alias('baseqtty'))

# 促销基本信息
df_promo_info_all = spark.sql("""
select 
    promo_id as his_promotion_id
from app.self_total_promo_rule_detail
group by promo_id
""").distinct()

# 流量数据
uv_info = spark.sql('''
select 
    sku_id as item_sku_id,
    dt,
    uv,
    pv
from dev.all_sku_traffic
where dt>='%s'
'''%start_dt).distinct()

# 日期信息
time_info = spark.sql('''
select
*
from app.app_pa_time
where dt >='%s'
'''%start_dt)


# In[3]:


# 历史基线部分
df_baseline = df_sql_skus.select('item_sku_id','promotion_id')                         .join(df_baseline_history, 'item_sku_id','inner')                         .withColumn('baseline_success',F.lit(1))                         .select('dt','item_sku_id','promotion_id','baseline_success')


# In[4]:


performance_1 = performance.filter((F.col('his_batch_id').isNotNull()) & (F.col('his_batch_id') != 0) & (F.col('his_batch_id') != '0'))
performance_2 = performance.filter((F.col('his_batch_id').isNull()) | (F.col('his_batch_id') == 0) | (F.col('his_batch_id') == '0'))                           .join(df_promo_info_all, 'his_promotion_id', 'inner')

sql_day_col = performance_1.columns
performance = performance_1.select(sql_day_col)                            .union(performance_2.select(sql_day_col))

# 历史活动表现（历史促销分析数据）
df_sql_sku_day_mess_xx = df_sql_skus.select('item_sku_id','promotion_id','min_dt')                                    .join(performance,'item_sku_id', "inner")


# sku dt维度
df_sql_sku_day_mess_xx = df_sql_sku_day_mess_xx     .groupby(['item_sku_id','promotion_id','min_dt' ,'his_promotion_id', 'his_batch_id', 'dt'])     .agg(
    F.sum('sale_qtty').alias('sale_qtty'),
    F.sum('baseline_sales').alias('baseline_sales'),
    F.sum('uplift_sales').alias('uplift_sales'),
    F.sum(F.col('promo_giveaway') + F.col('coupon_giveaway')).alias('total_giveaway'),
    F.sum(F.col('promo_giveaway')).alias('promo_giveaway'),
    F.sum(F.col('coupon_giveaway')).alias('coupon_giveaway'),
    F.sum('cann_sales').alias('cann_sales'),
    F.sum('halo').alias('halo'),
    F.sum('incremental_halo').alias('incremental_halo')
)


# In[5]:


# 历史活动信息
# 距离本次活动开始的 0-30 0-90 0-365 90-365 四个区间的 uv 红价 基线价
df_his_promo = df_sql_skus.select('item_sku_id','promotion_id','min_dt')                          .withColumn('start_str_30',F.date_sub(F.col('min_dt'),30))                          .withColumn('start_str_90',F.date_sub(F.col('min_dt'),90))                          .withColumn('start_str_365',F.date_sub(F.col('min_dt'),365))

# =====================================历史uv================================================
df_his_promo_uv_1 = df_his_promo.join(uv_info,'item_sku_id','left')

df_his_promo_uv_0_30 = df_his_promo_uv_1.filter((F.col('dt')>=F.col('start_str_30'))&(F.col('dt')<=F.col('min_dt')))                                        .groupby('item_sku_id','promotion_id')                                        .agg(F.mean('uv').alias('his_uv_0_30'))

df_his_promo_uv_0_90 = df_his_promo_uv_1.filter((F.col('dt')>=F.col('start_str_90'))&(F.col('dt')<=F.col('min_dt')))                                        .groupby('item_sku_id','promotion_id')                                        .agg(F.mean('uv').alias('his_uv_0_90'))

df_his_promo_uv_0_365 = df_his_promo_uv_1.filter((F.col('dt')>=F.col('start_str_365'))&(F.col('dt')<=F.col('min_dt')))                                         .groupby('item_sku_id','promotion_id')                                         .agg(F.mean('uv').alias('his_uv_0_365'))

df_his_promo_uv_90_365 = df_his_promo_uv_1.filter((F.col('dt')>=F.col('start_str_365'))&(F.col('dt')<=F.col('start_str_90')))                                          .groupby('item_sku_id','promotion_id')                                          .agg(F.mean('uv').alias('his_uv_90_365'))

df_his_promo_uv = df_sql_skus.select('item_sku_id','promotion_id')                             .join(df_his_promo_uv_0_30,['item_sku_id','promotion_id'],'left')                             .join(df_his_promo_uv_0_90,['item_sku_id','promotion_id'],'left')                             .join(df_his_promo_uv_0_365,['item_sku_id','promotion_id'],'left')                             .join(df_his_promo_uv_90_365,['item_sku_id','promotion_id'],'left')                             .fillna(0)

# =====================================历史红价================================================
df_his_promo_red_1 = df_his_promo.join(df_sql_sku_price_scraped,'item_sku_id','left')

df_his_promo_red_0_30 = df_his_promo_red_1.filter((F.col('dt')>=F.col('start_str_30'))&(F.col('dt')<=F.col('min_dt')))                                        .groupby('item_sku_id','promotion_id')                                        .agg(F.mean('red_price').alias('his_red_price_0_30'))

df_his_promo_red_0_90 = df_his_promo_red_1.filter((F.col('dt')>=F.col('start_str_90'))&(F.col('dt')<=F.col('min_dt')))                                        .groupby('item_sku_id','promotion_id')                                        .agg(F.mean('red_price').alias('his_red_price_0_90'))

df_his_promo_red_0_365 = df_his_promo_red_1.filter((F.col('dt')>=F.col('start_str_365'))&(F.col('dt')<=F.col('min_dt')))                                        .groupby('item_sku_id','promotion_id')                                        .agg(F.mean('red_price').alias('his_red_price_0_365'))

df_his_promo_red_90_365 = df_his_promo_red_1.filter((F.col('dt')>=F.col('start_str_365'))&(F.col('dt')<=F.col('start_str_90')))                                        .groupby('item_sku_id','promotion_id')                                        .agg(F.mean('red_price').alias('his_red_price_90_365'))

df_his_promo_red = df_sql_skus.select('item_sku_id','promotion_id')                             .join(df_his_promo_red_0_30,['item_sku_id','promotion_id'],'left')                             .join(df_his_promo_red_0_90,['item_sku_id','promotion_id'],'left')                             .join(df_his_promo_red_0_365,['item_sku_id','promotion_id'],'left')                             .join(df_his_promo_red_90_365,['item_sku_id','promotion_id'],'left')                             .fillna(0)

# =====================================历史基线价================================================
df_his_promo_base_1 = df_his_promo.join(df_sql_sku_price_baseline,'item_sku_id','left')

df_his_promo_base_0_30 = df_his_promo_base_1.filter((F.col('dt')>=F.col('start_str_30'))&(F.col('dt')<=F.col('min_dt')))                                        .groupby('item_sku_id','promotion_id')                                        .agg(F.mean('baseprice').alias('his_baseprice_0_30'))

df_his_promo_base_0_90 = df_his_promo_base_1.filter((F.col('dt')>=F.col('start_str_90'))&(F.col('dt')<=F.col('min_dt')))                                        .groupby('item_sku_id','promotion_id')                                        .agg(F.mean('baseprice').alias('his_baseprice_0_90'))

df_his_promo_base_0_365 = df_his_promo_base_1.filter((F.col('dt')>=F.col('start_str_365'))&(F.col('dt')<=F.col('min_dt')))                                        .groupby('item_sku_id','promotion_id')                                        .agg(F.mean('baseprice').alias('his_baseprice_0_365'))

df_his_promo_base_90_365 = df_his_promo_base_1.filter((F.col('dt')>=F.col('start_str_365'))&(F.col('dt')<=F.col('start_str_90')))                                        .groupby('item_sku_id','promotion_id')                                        .agg(F.mean('baseprice').alias('his_baseprice_90_365'))

df_his_promo_base = df_sql_skus.select('item_sku_id','promotion_id')                             .join(df_his_promo_base_0_30,['item_sku_id','promotion_id'],'left')                             .join(df_his_promo_base_0_90,['item_sku_id','promotion_id'],'left')                             .join(df_his_promo_base_0_365,['item_sku_id','promotion_id'],'left')                             .join(df_his_promo_base_90_365,['item_sku_id','promotion_id'],'left')                             .fillna(0)

df_his_promo = df_his_promo_uv.join(df_his_promo_red,['item_sku_id','promotion_id'],'inner')                              .join(df_his_promo_base,['item_sku_id','promotion_id'],'inner')


# In[6]:


# 特征加工
feature_1 = df_sql_sku_day_mess_xx.join(df_baseline,['item_sku_id','promotion_id','dt'],'inner')                                  .filter(F.col('baseline_success')==1)                                  .drop(df_baseline['baseline_success'])                                  .withColumn('discount_rate_history',F.col('total_giveaway')/(F.col('baseline_sales')+F.col('uplift_sales')))
feature_1 = feature_1.withColumn('discount_level_history',F.when(F.col('discount_rate_history')<0.1,F.lit('10per_less'))
                                                           .when((F.col('discount_rate_history')<0.2)&(F.col('discount_rate_history')>=0.1),F.lit('10per_20per'))
                                                           .when((F.col('discount_rate_history')<0.3)&(F.col('discount_rate_history')>=0.2),F.lit('20per_30per'))
                                                           .when((F.col('discount_rate_history')<0.4)&(F.col('discount_rate_history')>=0.3),F.lit('30per_40per'))
                                                           .when((F.col('discount_rate_history')<0.5)&(F.col('discount_rate_history')>=0.4),F.lit('40per_50per'))
                                                           .otherwise(F.lit('50per_more')))\
                     .withColumn('dt_minus',F.datediff(F.col('min_dt'),F.col('dt')))\
                     .withColumn('flag_0_30',F.when(F.col('dt_minus')<=30,F.lit(1)).otherwise(F.lit(0)))\
                     .withColumn('flag_0_90',F.when(F.col('dt_minus')<=90,F.lit(1)).otherwise(F.lit(0)))\
                     .withColumn('flag_0_365',F.when(F.col('dt_minus')<=365,F.lit(1)).otherwise(F.lit(0)))\
                     .withColumn('flag_90_365',F.when((F.col('dt_minus')<=365)&(F.col('dt_minus')>90),F.lit(1)).otherwise(F.lit(0)))\
                     .withColumn('black_flag',F.when(F.col('uplift_sales')<F.col('total_giveaway'),F.lit(1)).otherwise(F.lit(0)))

status_info = df_sql_skus.select('item_sku_id','promotion_id','min_dt')                         .join(df_sql_sku_status.filter(F.col('sku_status_cd')==3001),(df_sql_skus.item_sku_id==df_sql_sku_status.item_sku_id)&
                                                 (df_sql_skus.min_dt>=df_sql_sku_status.dt),'inner')\
                         .drop(df_sql_sku_status['item_sku_id'])\
                         .withColumn('dt_minus',F.datediff(F.col('min_dt'),F.col('dt')))\
                         .withColumn('flag_0_30',F.when(F.col('dt_minus')<=30,F.lit(1)).otherwise(F.lit(0)))\
                         .withColumn('flag_0_90',F.when(F.col('dt_minus')<=90,F.lit(1)).otherwise(F.lit(0)))\
                         .withColumn('flag_0_365',F.when(F.col('dt_minus')<=365,F.lit(1)).otherwise(F.lit(0)))\
                         .withColumn('flag_90_365',F.when((F.col('dt_minus')<=365)&(F.col('dt_minus')>90),F.lit(1)).otherwise(F.lit(0)))

# 历史活动按groupby字段算日均销量，若当天没卖出去，则按0计算
for x,y in zip([0,0,0,90],[30,90,365,365]):
    name['status_%s_%s'%(x,y)] = status_info.filter(F.col('flag_%s_%s'%(x,y))==1)                                            .groupby('item_sku_id','promotion_id')                                            .agg(F.count('sku_status_cd').alias('status_%s_%s'%(x,y)))
    
    name['feature_%s_%s'%(x,y)] = eval('status_%s_%s'%(x,y))                            .join(feature_1.filter(F.col('flag_%s_%s'%(x,y))==1),['item_sku_id','promotion_id'],'left')                            .groupby('item_sku_id','promotion_id','discount_level_history')                            .agg((F.sum('sale_qtty')/F.max('status_%s_%s'%(x,y))).alias('sale_qtty_%s_%s'%(x,y)),
                                 F.count('sale_qtty').alias('len_%s_%s'%(x,y)),
                                 (F.sum('black_flag')/F.count('sale_qtty')).alias('black_rate_%s_%s'%(x,y)),
                                 ((F.sum('uplift_sales')-F.sum('total_giveaway'))/F.sum('total_giveaway')).alias('roi_%s_%s'%(x,y)),
                                 ((F.sum('uplift_sales')-F.sum('total_giveaway'))/F.sum('baseline_sales')).alias('incre_%s_%s'%(x,y)))


feature_2 = feature_1.select('item_sku_id','promotion_id','discount_level_history').distinct()                     .join(feature_0_30,['item_sku_id','promotion_id','discount_level_history'],'left')                     .join(feature_0_90,['item_sku_id','promotion_id','discount_level_history'],'left')                     .join(feature_0_365,['item_sku_id','promotion_id','discount_level_history'],'left')                     .join(feature_90_365,['item_sku_id','promotion_id','discount_level_history'],'left')                     .fillna(0)

feature_3 = feature_2.groupby('item_sku_id','promotion_id').pivot('discount_level_history', ['10per_less','10per_20per','20per_30per','30per_40per','40per_50per','50per_more'])\
                     .agg(*[F.sum('sale_qtty_%s_%s'%(x,y)).alias('sale_qtty_%s_%s'%(x,y))  for x,y in zip([0,0,0,90],[30,90,365,365])],
                     *[F.sum('len_%s_%s'%(x,y)).alias('len_%s_%s'%(x,y))  for x,y in zip([0,0,0,90],[30,90,365,365])],
                     *[F.sum('black_rate_%s_%s'%(x,y)).alias('black_rate_%s_%s'%(x,y))  for x,y in zip([0,0,0,90],[30,90,365,365])],
                     *[F.sum('roi_%s_%s'%(x,y)).alias('roi_%s_%s'%(x,y))  for x,y in zip([0,0,0,90],[30,90,365,365])],
                     *[F.sum('incre_%s_%s'%(x,y)).alias('incre_%s_%s'%(x,y))  for x,y in zip([0,0,0,90],[30,90,365,365])])\
                .fillna(0)


# In[7]:


# 本次活动 最新分区前一周日均流量、销量
days_para = 7
df_cur_promo_tmp = df_sql_skus.select('item_sku_id','promotion_id','min_dt')                              .withColumn('pricetime',F.col('min_dt'))                              .withColumn('pricetime_7',F.date_sub(F.col('pricetime'),days_para))

df_cur_promo_3_uv = df_cur_promo_tmp.join(uv_info,(df_cur_promo_tmp.item_sku_id == uv_info.item_sku_id)&
                                                  (df_cur_promo_tmp.pricetime_7 <= uv_info.dt)&
                                                  (df_cur_promo_tmp.pricetime > uv_info.dt),'left')\
                                    .drop(uv_info['item_sku_id'])\
                                    .drop(uv_info['dt'])\
                                    .groupby('item_sku_id','promotion_id')\
                                    .agg(F.mean('uv').alias('uv'))\
                                    .fillna(0)

cur_performance = performance.groupby('item_sku_id','dt').agg(F.sum('sale_qtty').alias('sale_qtty'))
df_cur_promo_3_qtty = df_cur_promo_tmp.join(df_sql_sku_status.filter(F.col('sku_status_cd')==3001),(df_cur_promo_tmp.item_sku_id == df_sql_sku_status.item_sku_id)&
                                                            (df_cur_promo_tmp.pricetime_7 <= df_sql_sku_status.dt)&
                                                            (df_cur_promo_tmp.pricetime > df_sql_sku_status.dt),'inner')\
                                      .drop(df_sql_sku_status['item_sku_id'])
df_cur_promo_3_qtty_2 = df_cur_promo_3_qtty.join(cur_performance,['item_sku_id','dt'],'left')                                           .fillna(0)                                           .groupby('item_sku_id','promotion_id')                                           .agg((F.sum('sale_qtty')/F.count('sku_status_cd')).alias('sale_qtty'))


df_cur_promo = df_sql_skus.select('item_sku_id','promotion_id')                          .join(df_cur_promo_3_uv,['item_sku_id','promotion_id'],'inner')                          .join(df_cur_promo_3_qtty_2,['item_sku_id','promotion_id'],'inner')


# In[8]:


model_data = feature_3.join(df_his_promo,['item_sku_id','promotion_id'],'inner')                      .join(df_cur_promo,['item_sku_id','promotion_id'],'inner')                      .withColumn('dt',F.lit(yesterday))


# In[9]:


def save_result(df, table_name, partitioning_columns=[], write_mode='insert'):
    if isinstance(partitioning_columns, str):
        partitioning_columns = [partitioning_columns]
    save_mode =  'overwrite'
    if write_mode == 'save':
        spark.sql('''drop table if exists %s '''%table_name)
        if len(partitioning_columns) > 0:
            df.repartition(*partitioning_columns).write.mode(save_mode).partitionBy(partitioning_columns).format('orc').saveAsTable(table_name)
        else:
            df.write.mode(save_mode).format('orc').saveAsTable(table_name)
        spark.sql('''ALTER TABLE %s SET TBLPROPERTIES ('author' = '%s')''' % (table_name, 'zoushuhan'))
    elif write_mode == 'insert':
        if len(partitioning_columns) > 0:
            rows = df.select(partitioning_columns).distinct().collect()
            querys = []
            for r in rows:
                p_str = ','.join(["%s='%s'" % (k, r[k]) for k in partitioning_columns])
                querys.append("alter table %s drop if exists partition(%s)" %
                              (table_name, p_str))
            for q in querys:
                spark.sql(q)
            df.repartition(*partitioning_columns).write.insertInto(table_name, overwrite=False)
        else:
            df.write.insertInto(table_name, overwrite=False)
    else:
        raise ValueError('mode "%s" not supported ' % write_mode)


# In[10]:


save_result(model_data,'dev.dev_empty_bottle_black_selection_book_self',partitioning_columns=['dt'], write_mode='insert')
# save_result(model_data,'dev.dev_empty_bottle_black_selection_book_self',partitioning_columns=['dt'], write_mode='save')

