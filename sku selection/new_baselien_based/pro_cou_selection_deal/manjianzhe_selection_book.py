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

latest_nosplit_dt =spark.sql('''select max(dt) from app.app_pa_performance_nosplit_book_self''').collect()[0][0]

# transactions_D_self
latest_trans_d_dt =spark.sql('''select max(dt) from app.app_pa_transactions_D_self''').collect()[0][0]

time_style = '%Y-%m-%d'
df_last = datetime.datetime.strptime(latest_trans_d_dt, time_style)
# Promotion decision shall be made 10 days before promotion start date, and we download data in all 365 days previous of decision date
df_last_365 = df_last - datetime.timedelta(days=365)
latest_trans_d_dt_365 = datetime.datetime.strftime(df_last_365, time_style)

# refresh_dt
refresh_dt = latest_nosplit_dt
# target_type
# target_type = sys.argv[1]
target_type = 'manjianzhe'

# important notes: bu_id 是之前表分区的内容，由于前后关联太重不更改，bu_id  现在相当于用dt分区，分区字段为 refresh_dt

# bu_id = 1420

# gdm = spark.sql('''
#    select sku_id as item_sku_id
#    from dev.self_sku_det_da
#    where sku_type=1
#    and dt='2019-03-01'
#    and bu_id = '%s'
# '''%(bu_id) )

distinct_sku = spark.sql('''
    select item_sku_id from 
    app.app_pa_performance_nosplit_book_self
    where dt= '%s'
	''' % latest_nosplit_dt).distinct()

distinct_sku.cache()

sql_sku_full = spark.sql('''
    select dt,
    full_minus_promotion_id as promotion_id,
    item_sku_id 
    from app.app_pa_transactions_D_self
    where dt>= '%s'
    and full_minus_offer_amount>0
    group by dt,full_minus_promotion_id,item_sku_id
''' % latest_trans_d_dt_365)

# 6808

sql_sku_full = distinct_sku.join(sql_sku_full, 'item_sku_id', 'inner')
# 筛选限定条件下的优惠券
sql_promo = spark.sql(
	'''
		select promotion_id,
		max(threshold_num) as threshold_num,
		max(discount_rate) as discount_rate,
		max(first_ratio) as first_ratio,
		max(days) as days
		from
			(select promo_id as promotion_id,
			threshold_num,
			discount_rate,
			discount_rate as first_ratio,
			datediff(end_time,start_time) as days
			from app.self_total_promo_rule_detail
			where dt>'%s'
			and dt<='%s'
			and promo_subtype = 15
			and threshold_num>0
			and discount_rate>0
			and start_time>'%s'
			and end_time<='%s') a
		group by promotion_id
	''' % (latest_trans_d_dt_365, latest_trans_d_dt, latest_trans_d_dt_365, latest_trans_d_dt)
)

sql_promo = sql_promo.withColumn('discount_face', F.col('discount_rate'))
sql_promo = sql_promo.filter('discount_face > 0.2 and discount_face<=0.95')
sql_promo = sql_promo.filter('days<40 and days>1')
# 1554


nonsplit = spark.sql('''
    select * from 
    app.app_pa_performance_nosplit_book_self
    where dt='%s'
	''' % latest_nosplit_dt)
sql_promo.cache()
sql_sku_full.cache()

# 筛选使用sku个数大于1000小于30000,活动天数小于20大于3的优惠券
df_promo = sql_promo.join(sql_sku_full, 'promotion_id', 'inner').groupby('promotion_id').agg(
	F.countDistinct('item_sku_id').alias('sku_num'))
df_promo = df_promo.filter('sku_num>100 and sku_num<=30000')
# df_promo.select('promotion_id').distinct().count()

# 420

sql_promo2 = nonsplit.join(df_promo, 'promotion_id', 'inner')

# 62438434
# 限制叠加的优惠券优惠金额占比小于30%的满减促销
promo_two = sql_promo2.groupby(['promotion_id', 'item_sku_id']).agg(
	F.sum('promo_giveaway').alias('promo_giveaway'),
	F.sum('coupon_giveaway').alias('coupon_giveaway'))

promo_two = promo_two.withColumn('ratio', F.col('coupon_giveaway') / F.col('promo_giveaway'))
promo_two = promo_two.join(distinct_sku, 'item_sku_id', 'inner')
promo_two = promo_two.filter('ratio<0.3')
promo_sku = promo_two.groupby('promotion_id').agg(
	F.countDistinct('item_sku_id').alias('sku_num')).filter('sku_num>100')

result = promo_two.join(promo_sku, 'promotion_id', 'inner').select(['item_sku_id', 'promotion_id'])

result = result.withColumn('type', F.lit(target_type))

result2 = result.select('item_sku_id', 'promotion_id', 'type')

result4 = result2.join(nonsplit, ['promotion_id', 'item_sku_id'], 'inner')

result3 = result.select(['promotion_id', 'type']).distinct()

promo100_info = result3.join(sql_sku_full, 'promotion_id', 'left').groupby(['promotion_id', 'type']).agg(
	F.min('dt').alias('min_dt'),
	F.max('dt').alias('max_dt')).join(sql_promo, 'promotion_id', 'left')

df_promo_all = result4.groupBy(['item_sku_id', 'promotion_id', 'type']).agg(
	F.sum('baseline_sales').alias('baseline_sales'),
	F.sum('uplift_sales').alias('uplift_sales'),
	F.sum('promo_giveaway').alias('promo_giveaway'),
	F.sum('coupon_giveaway').alias('coupon_giveaway'),
	F.sum('cann_sales').alias('cann_sales'),
	F.sum('halo').alias('halo'))

df_promo_all = df_promo_all.withColumn('total_giveaway', F.col('promo_giveaway') + F.col('coupon_giveaway'))

df_promo_all = df_promo_all.withColumn('uplift_minus_discount', F.col('uplift_sales') - F.col('total_giveaway')) \
	.withColumn('baseline_uplift_discount', F.col('baseline_sales') + F.col('uplift_sales') - F.col('total_giveaway')) \
	.withColumn('incremental_halo',
                F.col('uplift_sales') - F.col('total_giveaway') + F.col('halo') - F.col('cann_sales'))

df_promo_all = df_promo_all.withColumn('bu_id', F.lit('%s' % refresh_dt))

df = df_promo_all.select('item_sku_id', 'baseline_sales', 'uplift_sales', 'uplift_minus_discount',
                         'baseline_uplift_discount', 'incremental_halo', 'promotion_id', 'bu_id', 'type')


# 满件折筛选(df_final_promo)
df_find_promotion = df.select('promotion_id').distinct() 
df_promo = spark.sql('''select promo_id as promotion_id, threshold_money, discount_money, threshold_num,\
                discount_num, discount_rate, add_money from app.self_total_promo_rule_detail
                where dt>'%s'
                and dt<='%s'
                and promo_subtype=15
                and start_time>'%s'
                and end_time<='%s' '''%(latest_trans_d_dt_365, latest_trans_d_dt, latest_trans_d_dt_365,  latest_trans_d_dt))
df_promo = df_promo.distinct()
df_final = df_find_promotion.join(df_promo, 'promotion_id', 'left')
df_final_1 = df_final.groupBy('promotion_id').agg(F.count('threshold_num').alias('promo_stage'))
df_final_2 = df_final_1.filter(F.col('promo_stage')==1)
df_final_promo = df_final_2.select('promotion_id').distinct()
df = df_final_promo.join(df, 'promotion_id', 'inner')
df = df.select('item_sku_id', 'baseline_sales', 'uplift_sales', 'uplift_minus_discount',
                         'baseline_uplift_discount', 'incremental_halo', 'promotion_id', 'bu_id', 'type')

# 建表函数(save)
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

table_name = 'dev.books_train_dataset'  # 表名需要更改，先建表，再存数据
partitioning_columns = ['bu_id', 'type']
df.write.insertInto(table_name, overwrite=True)


promo100_info_1 = promo100_info.withColumn('consume_lim', F.col('threshold_num'))
promo100_info_1 = promo100_info_1.withColumn('cps_face_value', F.col('discount_rate'))
promo100_info_1 = promo100_info_1.withColumn('bu_id', F.lit(refresh_dt))
promo100_info_1 = promo100_info_1.select(
	["promotion_id", "min_dt", "max_dt", "cps_face_value", "consume_lim", 'bu_id', 'type'])
promo100_info_1 = promo100_info_1.distinct()

# 筛选满减促销中的满减
promo100_info_1 = df_final_promo.join(promo100_info_1, 'promotion_id', 'inner')
promo100_info_1 = promo100_info_1.select(['promotion_id','min_dt','max_dt','cps_face_value','consume_lim','bu_id','type'])

table_name = 'app.books_promo_train_data'  # 表名需要更改，先建表，再存数据
partitioning_columns = ['bu_id', 'type']
promo100_info_1.write.insertInto(table_name, overwrite=True)
# batch100_info.repartition(1).write.mode('overwrite').format('orc').saveAsTextFile(path_now+'batch100', header=True, mode='overwrite', sep='\t')