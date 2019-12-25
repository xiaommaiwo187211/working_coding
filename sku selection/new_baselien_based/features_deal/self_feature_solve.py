# coding: utf-8
# In[1]:
# subtype =15
import datetime
import sys

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark import SparkFiles
import pyspark.sql.types as sql_type
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
import pyspark.sql.types as sql_type
app_name = 'black_list'
spark = SparkSession.builder.appName(app_name).enableHiveSupport().getOrCreate()
spark.conf.set("hive.exec.dynamic.partition", "true")
spark.conf.set("hive.exec.dynamic.partition.mode", "nonstrict")

# In[2]:
name = locals()

# In[3]:
latest_nosplit_dt = spark.sql('''select max(dt) from app.app_pa_performance_nosplit_self_deal_price''').collect()[0][0]
bu_id = latest_nosplit_dt # (bu_id= last partition of nosplit)

stage = sys.argv[1] #[1_stage, 2_stage]
target_type = sys.argv[2] # (coupon, manjianzhe, manjian, meimanjian,2_stage_manjianzhe,2_stage_manjian)


# stage = "2_stage" #[1_stage, 2_stage]
# bu_id = "2019-03-22"  # [1726,1420]:
# target_type = '2_stage_manjianzhe'  # ['coupon','meimanjian','manjian']:
test = 'all'

# In[4]:

# 目标：2个代码走天下，分为book和self；
if stage == "1_stage":
	# 读促销信息表-----by changxin
	chang_info = spark.sql('''select * from app.self_promo_train_data''')
	# 读本次促销sku表现数据
	label = spark.sql('''select * from dev.self_train_dataset''')


if stage == "2_stage":
	# 读促销信息表-----by changxin
	chang_info = spark.sql('''select * from app.self_promo_train_data_2_stages''')
	# 读本次促销sku表现数据
	label = spark.sql('''select * from dev.self_train_dataset_2_stages''')


label = label.withColumn('label', F.when(F.col('uplift_minus_discount') < 0, F.lit(1)).otherwise(F.lit(0)))
base_info = chang_info.join(label, ['bu_id', 'type', 'promotion_id'], 'inner')

if test == 'test':
	base_info = base_info.filter(
		(F.col('bu_id') == bu_id) & (F.col('type') == target_type) & (F.col('promotion_id') == 83828678))
else:
	base_info = base_info.filter((F.col('bu_id') == bu_id) & (F.col('type') == target_type))

# In[ ]:


# 日期处理
base_info = base_info.withColumn('start_str_chanpin', F.date_sub(F.col('min_dt'), 375)).withColumn('end_str_chanpin',
                                                                                                   F.date_sub(
	                                                                                                   F.col('min_dt'),
	                                                                                                   10)).withColumn(
	'pricetime', F.date_sub(F.col('min_dt'), 7))

# 起始日期，减少数据量
origin_dt = '2017-01-01'

# sku池子与促销信息
df_sql_skus = base_info.select('item_sku_id', 'promotion_id', 'min_dt', 'max_dt', 'start_str_chanpin',
                               'end_str_chanpin', 'pricetime', 'consume_lim', 'cps_face_value').distinct()

# 历史nosplit表信息
yesterday = spark.sql('''select max(dt) from app.app_pa_performance_nosplit_self_deal_price ''').collect()[0][0]
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
from app.app_pa_performance_nosplit_self_deal_price
where dt='%s'
and date >= '%s'
''' % (yesterday, origin_dt))

# 读取SKU的最新上下柜状态信息
df_sql_sku_status = spark.sql('''
select 
    sku_id as item_sku_id,
    cast(sku_status_cd as int) as sku_status_cd,
    dt
from dev.self_sku_det_da
where sku_type = 1
and dt>='%s'
''' % origin_dt)
df_sql_sku_status = df_sql_sku_status.groupby('item_sku_id', 'dt').agg(F.max('sku_status_cd').alias('sku_status_cd'))

# 读取SKU的最新红价数据
df_sql_sku_price_scraped = spark.sql('''
select 
    sku_id as item_sku_id,
    last_price,
    dt
from dev.self_sku_redprice_group
where dt >= '%s'
''' % origin_dt)
df_sql_sku_price_scraped = df_sql_sku_price_scraped.groupby('item_sku_id', 'dt').agg(
	F.mean('last_price').alias('red_price'))

# 读取SKU的最新基线价格
df_sql_sku_price_baseline = spark.sql('''
select 
    item_sku_id,
    baseprice,
    dt
from app.app_pa_baseline_xiaoxiao_final''')
df_sql_sku_price_baseline = df_sql_sku_price_baseline.groupby('item_sku_id', 'dt').agg(
	F.mean('baseprice').alias('baseprice'))


# 基线信息
df_baseline_history = spark.sql('''
select
    dt, 
    item_sku_id, 
    final_baseline
from app.app_pa_baseline_xiaoxiao_final
where dt >='%s'
''' % origin_dt)
df_baseline_history = df_baseline_history.groupby('item_sku_id', 'dt').agg(F.mean('final_baseline').alias('baseqtty'))

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
from dev.self_sku_pv
where dt>='%s'
''' % origin_dt).distinct()

# 日期信息
time_info = spark.sql('''
select
*
from app.app_pa_time
where dt >='%s'
''' % origin_dt)

# In[ ]:


# 历史基线部分
df_baseline = df_sql_skus.select('item_sku_id', 'promotion_id', 'start_str_chanpin', 'end_str_chanpin').join(
	df_baseline_history, (df_sql_skus.item_sku_id == df_baseline_history.item_sku_id) &
	(df_sql_skus.start_str_chanpin <= df_baseline_history.dt) &
	(df_sql_skus.end_str_chanpin >= df_baseline_history.dt), 'inner') \
	.select(df_sql_skus.item_sku_id, df_sql_skus.promotion_id, df_baseline_history.dt, df_baseline_history.baseqtty) \
	.withColumn('baseline_success', F.lit(1)) \
	.select('dt', 'item_sku_id', 'promotion_id', 'baseline_success')

# In[ ]:


# 历史活动表现（历史促销分析数据）

performance_1 = performance.filter(
	(F.col('his_batch_id').isNotNull()) & (F.col('his_batch_id') != 0) & (F.col('his_batch_id') != '0'))
performance_2 = performance.filter(
	(F.col('his_batch_id').isNull()) | (F.col('his_batch_id') == 0) | (F.col('his_batch_id') == '0')).join(
	df_promo_info_all, 'his_promotion_id', 'inner')

sql_day_col = performance_1.columns
performance = performance_1.select(sql_day_col).union(performance_2.select(sql_day_col))

df_sql_sku_day_mess_xx = df_sql_skus.select('item_sku_id', 'promotion_id', 'start_str_chanpin', 'end_str_chanpin').join(
	performance, (df_sql_skus.item_sku_id == performance.item_sku_id) &
	(df_sql_skus.start_str_chanpin <= performance.dt) &
	(df_sql_skus.end_str_chanpin >= performance.dt), "inner") \
	.drop(df_sql_skus['start_str_chanpin']) \
	.drop(performance['item_sku_id'])

# sku dt维度
df_sql_sku_day_mess_xx = df_sql_sku_day_mess_xx.groupby(
	['item_sku_id', 'promotion_id', 'end_str_chanpin', 'his_promotion_id', 'his_batch_id', 'dt']).agg(
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

# In[ ]:


# 历史活动信息
# 距离本次活动开始的 0-30 0-90 0-365 90-365 四个区间的 uv 红价 基线价
df_his_promo = df_sql_skus.select('item_sku_id', 'promotion_id', 'end_str_chanpin').withColumn('start_str_30',
                                                                                               F.date_sub(F.col(
	                                                                                               'end_str_chanpin'),
                                                                                                          30)).withColumn(
	'start_str_90', F.date_sub(F.col('end_str_chanpin'), 90)).withColumn('start_str_365',
                                                                         F.date_sub(F.col('end_str_chanpin'), 365))

# =====================================历史uv================================================
df_his_promo_uv_1 = df_his_promo.join(uv_info, 'item_sku_id', 'left')

df_his_promo_uv_0_30 = df_his_promo_uv_1.filter(
	(F.col('dt') >= F.col('start_str_30')) & (F.col('dt') <= F.col('end_str_chanpin'))).groupby('item_sku_id',
                                                                                                'promotion_id').agg(
	F.mean('uv').alias('his_uv_0_30'))

df_his_promo_uv_0_90 = df_his_promo_uv_1.filter(
	(F.col('dt') >= F.col('start_str_90')) & (F.col('dt') <= F.col('end_str_chanpin'))).groupby('item_sku_id',
                                                                                                'promotion_id').agg(
	F.mean('uv').alias('his_uv_0_90'))

df_his_promo_uv_0_365 = df_his_promo_uv_1.filter(
	(F.col('dt') >= F.col('start_str_365')) & (F.col('dt') <= F.col('end_str_chanpin'))).groupby('item_sku_id',
                                                                                                 'promotion_id').agg(
	F.mean('uv').alias('his_uv_0_365'))

df_his_promo_uv_90_365 = df_his_promo_uv_1.filter(
	(F.col('dt') >= F.col('start_str_365')) & (F.col('dt') <= F.col('start_str_90'))).groupby('item_sku_id',
                                                                                              'promotion_id').agg(
	F.mean('uv').alias('his_uv_90_365'))

df_his_promo_uv = df_sql_skus.select('item_sku_id', 'promotion_id').join(df_his_promo_uv_0_30,
                                                                         ['item_sku_id', 'promotion_id'], 'left').join(
	df_his_promo_uv_0_90, ['item_sku_id', 'promotion_id'], 'left').join(df_his_promo_uv_0_365,
                                                                        ['item_sku_id', 'promotion_id'], 'left').join(
	df_his_promo_uv_90_365, ['item_sku_id', 'promotion_id'], 'left').fillna(0)

# =====================================历史红价================================================
df_his_promo_red_1 = df_his_promo.join(df_sql_sku_price_scraped, 'item_sku_id', 'left')

df_his_promo_red_0_30 = df_his_promo_red_1.filter(
	(F.col('dt') >= F.col('start_str_30')) & (F.col('dt') <= F.col('end_str_chanpin'))).groupby('item_sku_id',
                                                                                                'promotion_id').agg(
	F.mean('red_price').alias('his_red_price_0_30'))

df_his_promo_red_0_90 = df_his_promo_red_1.filter(
	(F.col('dt') >= F.col('start_str_90')) & (F.col('dt') <= F.col('end_str_chanpin'))).groupby('item_sku_id',
                                                                                                'promotion_id').agg(
	F.mean('red_price').alias('his_red_price_0_90'))

df_his_promo_red_0_365 = df_his_promo_red_1.filter(
	(F.col('dt') >= F.col('start_str_365')) & (F.col('dt') <= F.col('end_str_chanpin'))).groupby('item_sku_id',
                                                                                                 'promotion_id').agg(
	F.mean('red_price').alias('his_red_price_0_365'))

df_his_promo_red_90_365 = df_his_promo_red_1.filter(
	(F.col('dt') >= F.col('start_str_365')) & (F.col('dt') <= F.col('start_str_90'))).groupby('item_sku_id',
                                                                                              'promotion_id').agg(
	F.mean('red_price').alias('his_red_price_90_365'))

df_his_promo_red = df_sql_skus.select('item_sku_id', 'promotion_id').join(df_his_promo_red_0_30,
                                                                          ['item_sku_id', 'promotion_id'], 'left').join(
	df_his_promo_red_0_90, ['item_sku_id', 'promotion_id'], 'left').join(df_his_promo_red_0_365,
                                                                         ['item_sku_id', 'promotion_id'], 'left').join(
	df_his_promo_red_90_365, ['item_sku_id', 'promotion_id'], 'left').fillna(0)

# =====================================历史基线价================================================
df_his_promo_base_1 = df_his_promo.join(df_sql_sku_price_baseline, 'item_sku_id', 'left')

df_his_promo_base_0_30 = df_his_promo_base_1.filter(
	(F.col('dt') >= F.col('start_str_30')) & (F.col('dt') <= F.col('end_str_chanpin'))).groupby('item_sku_id',
                                                                                                'promotion_id').agg(
	F.mean('baseprice').alias('his_baseprice_0_30'))

df_his_promo_base_0_90 = df_his_promo_base_1.filter(
	(F.col('dt') >= F.col('start_str_90')) & (F.col('dt') <= F.col('end_str_chanpin'))).groupby('item_sku_id',
                                                                                                'promotion_id').agg(
	F.mean('baseprice').alias('his_baseprice_0_90'))

df_his_promo_base_0_365 = df_his_promo_base_1.filter(
	(F.col('dt') >= F.col('start_str_365')) & (F.col('dt') <= F.col('end_str_chanpin'))).groupby('item_sku_id',
                                                                                                 'promotion_id').agg(
	F.mean('baseprice').alias('his_baseprice_0_365'))

df_his_promo_base_90_365 = df_his_promo_base_1.filter(
	(F.col('dt') >= F.col('start_str_365')) & (F.col('dt') <= F.col('start_str_90'))).groupby('item_sku_id',
                                                                                              'promotion_id').agg(
	F.mean('baseprice').alias('his_baseprice_90_365'))

df_his_promo_base = df_sql_skus.select('item_sku_id', 'promotion_id').join(df_his_promo_base_0_30,
                                                                           ['item_sku_id', 'promotion_id'],
                                                                           'left').join(df_his_promo_base_0_90,
                                                                                        ['item_sku_id', 'promotion_id'],
                                                                                        'left').join(
	df_his_promo_base_0_365, ['item_sku_id', 'promotion_id'], 'left').join(df_his_promo_base_90_365,
                                                                           ['item_sku_id', 'promotion_id'],
                                                                           'left').fillna(0)

df_his_promo = df_his_promo_uv.join(df_his_promo_red, ['item_sku_id', 'promotion_id'], 'inner').join(df_his_promo_base,
                                                                                                     ['item_sku_id',
                                                                                                      'promotion_id'],
                                                                                                     'inner')

# In[ ]:


# 本次活动基本信息
# （1）本次活动开始7天前的红价 基线价（2）本次活动名义折扣率 门槛 面额
# （3）本次活动前7天至前14天的日均流量 日均销量
# （4）本次活动是否覆盖节假日，以及活动开始日期是周几，第几天，第几周
days_para = 7 # 本次活动前7天到前（7+days_para）天，共一周
if target_type == 'manjianzhe':
	df_cur_promo_1 = df_sql_skus.select('item_sku_id', 'promotion_id', 'consume_lim', 'cps_face_value') \
		.withColumn('discount_rate_cal', F.col('cps_face_value'))
else:
	df_cur_promo_1 = df_sql_skus.select('item_sku_id','promotion_id','consume_lim','cps_face_value')\
		.withColumn('discount_rate_cal',F.col('cps_face_value')/F.col('consume_lim'))


df_cur_promo_2 = df_sql_skus.select('item_sku_id','promotion_id','pricetime')\
                            .join(df_sql_sku_price_scraped, (df_sql_skus.item_sku_id==df_sql_sku_price_scraped.item_sku_id)&
                                                     (df_sql_skus.pricetime==df_sql_sku_price_scraped.dt), "inner")\
                            .drop(df_sql_sku_price_scraped['item_sku_id'])\
                            .drop(df_sql_sku_price_scraped['dt'])\
                            .join(df_sql_sku_price_baseline,(df_sql_skus.item_sku_id==df_sql_sku_price_baseline.item_sku_id)&
                                                     (df_sql_skus.pricetime==df_sql_sku_price_baseline.dt), "inner")\
                            .drop(df_sql_sku_price_baseline['item_sku_id'])\
                            .drop(df_sql_sku_price_baseline['dt'])

df_cur_promo_tmp = df_sql_skus.select('item_sku_id','promotion_id','pricetime')\
                            .withColumn('pricetime_7',F.date_sub(F.col('pricetime'),days_para))
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
df_cur_promo_3_qtty_2 = df_cur_promo_3_qtty.join(cur_performance,['item_sku_id','dt'],'left')\
                                           .fillna(0)\
                                           .groupby('item_sku_id','promotion_id')\
                                           .agg((F.sum('sale_qtty')/F.count('sku_status_cd')).alias('sale_qtty'))

df_cur_promo_4 = df_sql_skus.select('item_sku_id','promotion_id','min_dt','max_dt')\
                            .withColumn('promo_days',F.datediff(F.col('max_dt'),F.col('min_dt'))+1)\
                            .join(time_info,(df_sql_skus.min_dt<=time_info.dt)&
                                            (df_sql_skus.max_dt>=time_info.dt),'left')\
                            .sort(['item_sku_id','promotion_id','dt'])\
                            .groupby('item_sku_id','promotion_id')\
                            .agg(F.max('promo_days').alias('promo_days'),
                                 *[F.first(x).alias(x) if x in ['day_of_week','day_of_year','week_of_year'] else F.max(x).alias(x) for x in list(set(time_info.columns)-set(['dt']))])

df_cur_promo = df_cur_promo_1.join(df_cur_promo_2,['item_sku_id','promotion_id'],'inner')\
                             .join(df_cur_promo_3_uv,['item_sku_id','promotion_id'],'inner')\
                             .join(df_cur_promo_3_qtty_2,['item_sku_id','promotion_id'],'inner')\
                             .join(df_cur_promo_4,['item_sku_id','promotion_id'],'inner')
# In[ ]:


# 特征加工
feature_1 = df_sql_sku_day_mess_xx.join(df_baseline, ['item_sku_id', 'promotion_id', 'dt'], 'inner').filter(
	F.col('baseline_success') == 1).drop(df_baseline['baseline_success']).withColumn('discount_rate_history',
                                                                                     F.col('total_giveaway') / (F.col(
	                                                                                     'baseline_sales') + F.col(
	                                                                                     'uplift_sales')))
feature_1 = feature_1.withColumn('discount_level_history',
                                 F.when(F.col('discount_rate_history') < 0.1, F.lit('10%_less'))
                                 .when((F.col('discount_rate_history') < 0.2) & (F.col('discount_rate_history') >= 0.1),
                                       F.lit('10%_20%'))
                                 .when((F.col('discount_rate_history') < 0.3) & (F.col('discount_rate_history') >= 0.2),
                                       F.lit('20%_30%'))
                                 .when((F.col('discount_rate_history') < 0.4) & (F.col('discount_rate_history') >= 0.3),
                                       F.lit('30%_40%'))
                                 .when((F.col('discount_rate_history') < 0.5) & (F.col('discount_rate_history') >= 0.4),
                                       F.lit('40%_50%'))
                                 .otherwise(F.lit('50%_more'))) \
	.withColumn('dt_minus', F.datediff(F.col('end_str_chanpin'), F.col('dt'))) \
	.withColumn('flag_0_30', F.when(F.col('dt_minus') <= 30, F.lit(1)).otherwise(F.lit(0))) \
	.withColumn('flag_0_90', F.when(F.col('dt_minus') <= 90, F.lit(1)).otherwise(F.lit(0))) \
	.withColumn('flag_0_365', F.when(F.col('dt_minus') <= 365, F.lit(1)).otherwise(F.lit(0))) \
	.withColumn('flag_90_365',
                F.when((F.col('dt_minus') <= 365) & (F.col('dt_minus') > 90), F.lit(1)).otherwise(F.lit(0))) \
	.withColumn('black_flag', F.when(F.col('uplift_sales') < F.col('total_giveaway'), F.lit(1)).otherwise(F.lit(0)))

status_info = df_sql_skus.select('item_sku_id', 'promotion_id', 'start_str_chanpin', 'end_str_chanpin').join(
	df_sql_sku_status.filter(F.col('sku_status_cd') == 3001),
	(df_sql_skus.item_sku_id == df_sql_sku_status.item_sku_id) &
	(df_sql_skus.end_str_chanpin >= df_sql_sku_status.dt) &
	(df_sql_skus.start_str_chanpin <= df_sql_sku_status.dt), 'inner') \
	.drop(df_sql_sku_status['item_sku_id']) \
	.withColumn('dt_minus', F.datediff(F.col('end_str_chanpin'), F.col('dt'))) \
	.withColumn('flag_0_30', F.when(F.col('dt_minus') <= 30, F.lit(1)).otherwise(F.lit(0))) \
	.withColumn('flag_0_90', F.when(F.col('dt_minus') <= 90, F.lit(1)).otherwise(F.lit(0))) \
	.withColumn('flag_0_365', F.when(F.col('dt_minus') <= 365, F.lit(1)).otherwise(F.lit(0))) \
	.withColumn('flag_90_365',
                F.when((F.col('dt_minus') <= 365) & (F.col('dt_minus') > 90), F.lit(1)).otherwise(F.lit(0)))

# 历史活动按groupby字段算日均销量，若当天没卖出去，则按0计算
for x, y in zip([0, 0, 0, 90], [30, 90, 365, 365]):
	name['status_%s_%s' % (x, y)] = status_info.filter(F.col('flag_%s_%s' % (x, y)) == 1).groupby('item_sku_id','promotion_id').\
		agg(F.count('sku_status_cd').alias('status_%s_%s' % (x, y)))
	name['feature_%s_%s' % (x, y)] = eval('status_%s_%s' % (x, y)).join(
		feature_1.filter(F.col('flag_%s_%s' % (x, y)) == 1), ['item_sku_id', 'promotion_id'], 'left').groupby(
		'item_sku_id', 'promotion_id', 'discount_level_history').agg(
		(F.sum('sale_qtty') / F.max('status_%s_%s' % (x, y))).alias('sale_qtty_%s_%s' % (x, y)),
		F.count('sale_qtty').alias('len_%s_%s' % (x, y)),
		(F.sum('black_flag') / F.count('sale_qtty')).alias('black_rate_%s_%s' % (x, y)),
		((F.sum('uplift_sales') - F.sum('total_giveaway')) / F.sum('total_giveaway')).alias('roi_%s_%s' % (x, y)),
		((F.sum('uplift_sales') - F.sum('total_giveaway')) / F.sum('baseline_sales')).alias('incre_%s_%s' % (x, y)))

feature_2 = feature_1.select('item_sku_id', 'promotion_id', 'discount_level_history').distinct() \
	.join(feature_0_30, ['item_sku_id', 'promotion_id', 'discount_level_history'], 'left') \
	.join(feature_0_90, ['item_sku_id', 'promotion_id', 'discount_level_history'], 'left') \
	.join(feature_0_365, ['item_sku_id', 'promotion_id', 'discount_level_history'], 'left') \
	.join(feature_90_365, ['item_sku_id', 'promotion_id', 'discount_level_history'], 'left') \
	.fillna(0)

#
# feature_3 = feature_2.groupby('item_sku_id', 'promotion_id') \
# 	.pivot('discount_level_history', ['10%_less', '10%_20%', '20%_30%', '30%_40%', '40%_50%', '50%_more']) \
# 	.agg(*[F.sum('sale_qtty_%s_%s' % (x, y)).alias('sale_qtty_%s_%s' % (x, y)) for x, y in
#            zip([0, 0, 0, 90], [30, 90, 365, 365])],
#          *[F.sum('len_%s_%s' % (x, y)).alias('len_%s_%s' % (x, y)) for x, y in zip([0, 0, 0, 90], [30, 90, 365, 365])],
#          *[F.sum('black_rate_%s_%s' % (x, y)).alias('black_rate_%s_%s' % (x, y)) for x, y in
#            zip([0, 0, 0, 90], [30, 90, 365, 365])],
#          *[F.sum('roi_%s_%s' % (x, y)).alias('roi_%s_%s' % (x, y)) for x, y in zip([0, 0, 0, 90], [30, 90, 365, 365])],
#          *[F.sum('incre_%s_%s' % (x, y)).alias('incre_%s_%s' % (x, y)) for x, y in
#            zip([0, 0, 0, 90], [30, 90, 365, 365])]) \
# 	.fillna(0)
cols = [F.sum('sale_qtty_%s_%s' % (x, y)).alias('sale_qtty_%s_%s' % (x, y)) for x, y in zip([0, 0, 0, 90], [30, 90, 365, 365])] \
+ [F.sum('len_%s_%s' % (x, y)).alias('len_%s_%s' % (x, y)) for x, y in zip([0, 0, 0, 90], [30, 90, 365, 365])] \
+[F.sum('black_rate_%s_%s' % (x, y)).alias('black_rate_%s_%s' % (x, y)) for x, y in zip([0, 0, 0, 90], [30, 90, 365, 365])]\
 +[F.sum('roi_%s_%s' % (x, y)).alias('roi_%s_%s' % (x, y)) for x, y in zip([0, 0, 0, 90], [30, 90, 365, 365])]\
 +[F.sum('incre_%s_%s' % (x, y)).alias('incre_%s_%s' % (x, y)) for x, y in zip([0, 0, 0, 90], [30, 90, 365, 365])]

feature_3 = feature_2.groupby('item_sku_id', 'promotion_id') \
    .pivot('discount_level_history', ['10%_less', '10%_20%', '20%_30%', '30%_40%', '40%_50%', '50%_more']) \
    .agg(*cols).fillna(0)


df_sql_sku_day_mess_xx.unpersist()

model_data = feature_3.join(df_his_promo, ['item_sku_id', 'promotion_id'], 'inner') \
	.join(df_cur_promo, ['item_sku_id', 'promotion_id'], 'inner') \
	.join(base_info.select('item_sku_id', 'promotion_id', 'label'), ['item_sku_id', 'promotion_id'], 'inner')

model_data = model_data.withColumn('bu_id', F.lit(bu_id)).withColumn('type', F.lit(target_type))

for x in model_data.columns:
	model_data = model_data.withColumnRenamed(x, x.replace('%','per'))

col_name = ['item_sku_id',
 'promotion_id',
 '10per_less_sale_qtty_0_30',
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
 'consume_lim',
 'cps_face_value',
 'discount_rate_cal',
 'pricetime',
 'red_price',
 'baseprice',
 'uv',
 'sale_qtty',
 'promo_days',
 'labourday',
 'springfestival',
 'h1111mark',
 'tombsweepingfestival',
 'midautumnfestival',
 'nationalday',
 'day_of_year',
 'day_of_week',
 'h618mark',
 'week_of_year',
 'h1212mark',
 'dragonboatfestival',
 'newyear',
 'label',
 'bu_id',
 'type']

model_data = model_data.select(col_name)

# # 建表函数(save)
# def save_result(df, table_name, partitioning_columns=[], write_mode='save'):
#     if isinstance(partitioning_columns, str):
#         partitioning_columns = [partitioning_columns]
#     save_mode =  'overwrite'
#     if write_mode == 'save':
#         if len(partitioning_columns) > 0:
#             df.repartition(*partitioning_columns).write.mode(save_mode).partitionBy(partitioning_columns).format('orc').saveAsTable(table_name)
#         else:
#             df.write.mode(save_mode).format('orc').saveAsTable(table_name)
#         spark.sql('''ALTER TABLE %s SET TBLPROPERTIES ('author' = '%s')''' % (table_name, 'xiaoxiao10'))
#     elif write_mode == 'insert':
#         if len(partitioning_columns) > 0:
#             rows = df.select(partitioning_columns).distinct().collect()
#             querys = []
#             for r in rows:
#                 p_str = ','.join(["%s='%s'" % (k, r[k]) for k in partitioning_columns])
#                 querys.append("alter table %s drop if exists partition(%s)" %
#                               (table_name, p_str))
#             for q in querys:
#                 spark.sql(q)
#             df.repartition(*partitioning_columns).write.insertInto(table_name, overwrite=False)
#         else:
#             df.write.insertInto(table_name, overwrite=False)
#     else:
#         raise ValueError('mode "%s" not supported ' % write_mode)

table_name = 'dev.black_list_model_feature_self'  # 表名需要更改，先建表，再存数据
# partitioning_columns = ['bu_id', 'type']
# # 第一次建表用save，建好后用insert
# save_result(model_data,table_name,partitioning_columns=partitioning_columns,write_mode='insert')
model_data.write.insertInto(table_name, overwrite=True)
