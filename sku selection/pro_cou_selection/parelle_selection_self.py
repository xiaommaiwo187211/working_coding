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

# nosplit取最新分区
latest_nosplit_dt =spark.sql('''select max(dt) from app.app_pa_performance_nosplit_self''').collect()[0][0]
# transactions_D_self
latest_trans_d_dt =spark.sql('''select max(dt) from app.app_pa_transactions_D_self''').collect()[0][0]

time_style = '%Y-%m-%d'
df_last = datetime.datetime.strptime(latest_trans_d_dt, time_style)

# 由于平行优惠于2019年2月15日开始实施，因此，开始日期设为2月15日（与之前的取数逻辑365不同）
latest_trans_d_dt_365 = "2019-02-15"
# refresh_dt
refresh_dt = latest_nosplit_dt
# target_type
# target_type = sys.argv[1]
target_type = 'parellel'

distinct_sku = spark.sql('''
    select * from 
    app.app_pa_performance_nosplit_self
    where dt= "%s" and date between "%s" and "%s" and promo_giveaway>0 and coupon_giveaway>0''' \
                         % (latest_nosplit_dt, latest_trans_d_dt_365, latest_trans_d_dt)).distinct()

distinct_sku.cache()

# meimanjian
total_promo_detail = spark.sql(
	'''
		select promo_id as promotion_id, promo_subtype, max(threshold_money) as threshold_money,\
        max(discount_money) as discount_money, max(threshold_num) as threshold_num, max(discount_num) as discount_num,\
        max(discount_rate) as discount_rate, max(add_money) as add_money
		from app.self_total_promo_rule_detail where promo_subtype in (1,2,4,15) \
        and start_time>='%s' and end_time<='%s' group by promo_id, promo_subtype
    ''' % (latest_trans_d_dt_365, latest_trans_d_dt)
).distinct()

df_sku_promo = distinct_sku.join(total_promo_detail, 'promotion_id', 'inner')

sql_promo_coupon_first = spark.sql(
	'''
	select batch_id
	FROM
	gdm.gdm_m07_cps_batch_da
	where dt='%s' and coupon_style=0
	''' % (latest_trans_d_dt)
).distinct()

df_sku_promo = df_sku_promo.join(sql_promo_coupon_first, 'batch_id', 'inner')

df_base_price = spark.sql('''
select 
    item_sku_id,
    baseprice,
    dt as date
from app.app_pa_price_baseprice_self
where dt between "%s" and "%s"
''' % (latest_trans_d_dt_365, latest_trans_d_dt))

df_baese_price = df_base_price.groupby('item_sku_id', 'date').agg(
	F.mean('baseprice').alias('baseprice'))


# 读取SKU的最新红价数据
df_red_price = spark.sql('''
select 
    sku_id as item_sku_id,
    last_price,
    dt as date
from dev.self_sku_redprice_group
where dt between "%s" and "%s"
''' % (latest_trans_d_dt_365, latest_trans_d_dt))

df_red_price = df_red_price.groupby('item_sku_id', 'date').agg(
	F.mean('last_price').alias('red_price'))

df_price = df_sku_promo.join(df_base_price, ["item_sku_id","date"], "inner").join(df_red_price, ["item_sku_id", "date"], "inner")

df_data_scope = df_price.filter((F.abs(F.col('baseprice')-F.col('red_price'))/F.col('red_price')<0.05))

df_data_scope = df_data_scope.select(['item_sku_id', 'date', 'promotion_id', 'batch_id', 'uplift_sales', 'baseline_sales', 'promo_giveaway',\
                     'coupon_giveaway', 'cann_sales', 'halo', 'promo_subtype', 'threshold_money', 'discount_money', 'threshold_num', 'discount_num',\
                     'discount_rate', 'add_money', 'baseprice', 'red_price'])

# In condition of promo_subtype, compute不同promo和batch的countDistinct(sku_id)
# df_promo_sku_num = df_data_scope.groupBy(['promotion_id', 'batch_id', 'promo_subtype']).agg(F.countDistinct('item_sku_id').alias('sku_num'))
# df_subpromo_sku_num = df_promo_sku_num.groupBy('promo_subtype').agg(F.sum('sku_num').alias('subtype_sku_num'))

# 每满减（没有任何阶梯）
df_meimanjian_all = df_data_scope.filter(F.col('promo_subtype')==2)

# 满减+阶梯满减
df_manjian_all = df_data_scope.filter(F.col('promo_subtype')==4)

# 满件折+阶梯满件折
df_manjianzhe_all = df_data_scope.filter(F.col('promo_subtype')==15)

#每满减
df_meimanjian = df_meimanjian_all.groupby(['promotion_id', 'item_sku_id', 'batch_id', 'promo_subtype']).agg(
	F.sum('baseline_sales').alias('baseline_sales'),
	F.sum('uplift_sales').alias('uplift_sales'),
    (F.sum('uplift_sales')-F.sum('promo_giveaway')-F.sum('coupon_giveaway')).alias('uplift_minus_discount'),
    (F.sum('baseline_sales') + F.sum('uplift_sales') -F.sum('promo_giveaway')-F.sum('coupon_giveaway')).alias('baseline_uplift_discount'),
    (F.sum('uplift_sales') -F.sum('promo_giveaway')-F.sum('coupon_giveaway') + F.sum('halo') - F.sum('cann_sales')).alias('incremental_halo')
).withColumn('bu_id',F.lit(latest_nosplit_dt)).withColumn('type', F.lit('meimanjian'))
df_meimanjian = df_meimanjian.select('item_sku_id', 'baseline_sales', 'uplift_sales', 'uplift_minus_discount',
                         'baseline_uplift_discount', 'incremental_halo', 'promotion_id','batch_id', 'bu_id', 'type')

# 筛选满减促销中的满减(df_final_promo)
df_find_promotion = df_manjian_all.select('promotion_id').distinct()
df_promo = spark.sql('''select promo_id as promotion_id, threshold_money, discount_money, threshold_num,\
                discount_num, discount_rate, add_money from app.self_total_promo_rule_detail
                where dt>'%s'
                and dt<='%s'
                and promo_subtype in (1,4)
                and start_time>'%s'
                and end_time<='%s' '''%(latest_trans_d_dt_365, latest_trans_d_dt, latest_trans_d_dt_365,  latest_trans_d_dt))
df_promo = df_promo.distinct()
df_final = df_find_promotion.join(df_promo, 'promotion_id', 'left')
df_final_1 = df_final.groupBy('promotion_id').agg(F.count('threshold_money').alias('promo_stage'))
df_final_2 = df_final_1.filter(F.col('promo_stage')==1)
df_final_promo = df_final_2.select('promotion_id').distinct()
df = df_final_promo.join(df_manjian_all, 'promotion_id', 'inner')

df = df.groupby(['promotion_id', 'item_sku_id', 'batch_id', 'promo_subtype']).agg(
	F.sum('baseline_sales').alias('baseline_sales'),
	F.sum('uplift_sales').alias('uplift_sales'),
    (F.sum('uplift_sales')-F.sum('promo_giveaway')-F.sum('coupon_giveaway')).alias('uplift_minus_discount'),
    (F.sum('baseline_sales') + F.sum('uplift_sales') -F.sum('promo_giveaway')-F.sum('coupon_giveaway')).alias('baseline_uplift_discount'),
    (F.sum('uplift_sales') -F.sum('promo_giveaway')-F.sum('coupon_giveaway') + F.sum('halo') - F.sum('cann_sales')).alias('incremental_halo')
).withColumn('bu_id',F.lit(latest_nosplit_dt)).withColumn('type', F.lit('manjian'))

df_manjian = df.select('item_sku_id', 'baseline_sales', 'uplift_sales', 'uplift_minus_discount',
                         'baseline_uplift_discount', 'incremental_halo', 'promotion_id','batch_id', 'bu_id', 'type')

# 筛选满减促销中的满减(df_final_promo)
df_find_promotion = df_manjian_all.select('promotion_id').distinct()
df_promo = spark.sql('''select promo_id as promotion_id,  threshold_money, discount_money, threshold_num,\
                discount_num, discount_rate, add_money from app.self_total_promo_rule_detail
                where dt>'%s'
                and dt<='%s'
                and promo_subtype in (1,4)
                and start_time>'%s'
                and end_time<='%s' '''%(latest_trans_d_dt_365, latest_trans_d_dt, latest_trans_d_dt_365,  latest_trans_d_dt))
df_promo = df_promo.distinct()
df_final = df_find_promotion.join(df_promo, 'promotion_id', 'left')
df_final_1 = df_final.groupBy('promotion_id').agg(F.count('threshold_money').alias('promo_stage'))
df_final_2 = df_final_1.filter(F.col('promo_stage')==2)
df_final_promo = df_final_2.select('promotion_id').distinct()
df = df_final_promo.join(df_manjian_all, 'promotion_id', 'inner')

df = df.groupby(['promotion_id', 'item_sku_id', 'batch_id', 'promo_subtype']).agg(
	F.sum('baseline_sales').alias('baseline_sales'),
	F.sum('uplift_sales').alias('uplift_sales'),
    (F.sum('uplift_sales')-F.sum('promo_giveaway')-F.sum('coupon_giveaway')).alias('uplift_minus_discount'),
    (F.sum('baseline_sales') + F.sum('uplift_sales') -F.sum('promo_giveaway')-F.sum('coupon_giveaway')).alias('baseline_uplift_discount'),
    (F.sum('uplift_sales') -F.sum('promo_giveaway')-F.sum('coupon_giveaway') + F.sum('halo') - F.sum('cann_sales')).alias('incremental_halo')
).withColumn('bu_id',F.lit(latest_nosplit_dt)).withColumn('type', F.lit('2_stage_manjian'))

df_2_stage_manjian = df.select('item_sku_id', 'baseline_sales', 'uplift_sales', 'uplift_minus_discount',
                         'baseline_uplift_discount', 'incremental_halo', 'promotion_id','batch_id', 'bu_id', 'type')

# 满件折筛选(df_final_promo)
df_find_promotion = df_manjianzhe_all.select('promotion_id').distinct()
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
df = df_final_promo.join(df_manjianzhe_all, 'promotion_id', 'inner')

df = df.groupby(['promotion_id', 'item_sku_id', 'batch_id', 'promo_subtype']).agg(
	F.sum('baseline_sales').alias('baseline_sales'),
	F.sum('uplift_sales').alias('uplift_sales'),
    (F.sum('uplift_sales')-F.sum('promo_giveaway')-F.sum('coupon_giveaway')).alias('uplift_minus_discount'),
    (F.sum('baseline_sales') + F.sum('uplift_sales') -F.sum('promo_giveaway')-F.sum('coupon_giveaway')).alias('baseline_uplift_discount'),
    (F.sum('uplift_sales') -F.sum('promo_giveaway')-F.sum('coupon_giveaway') + F.sum('halo') - F.sum('cann_sales')).alias('incremental_halo')
).withColumn('bu_id',F.lit(latest_nosplit_dt)).withColumn('type', F.lit('manjianzhe'))

df_manjianzhe = df.select('item_sku_id', 'baseline_sales', 'uplift_sales', 'uplift_minus_discount',
                         'baseline_uplift_discount', 'incremental_halo', 'promotion_id','batch_id', 'bu_id', 'type')

# 满件折筛选（df_final_promo）
df_find_promotion = df_manjianzhe_all.select('promotion_id').distinct()
df_promo = spark.sql('''select promo_id as promotion_id,  threshold_money, discount_money, threshold_num,\
                discount_num, discount_rate, add_money from app.self_total_promo_rule_detail
        where dt>'%s'
                and dt<='%s'
                and promo_subtype=15
                and start_time>'%s'
                and end_time<='%s' '''%(latest_trans_d_dt_365, latest_trans_d_dt, latest_trans_d_dt_365,  latest_trans_d_dt))

df_promo = df_promo.distinct()
df_final = df_find_promotion.join(df_promo, 'promotion_id', 'left')
df_final_1 = df_final.groupBy('promotion_id').agg(F.count('threshold_num').alias('promo_stage'))
df_final_2 = df_final_1.filter(F.col('promo_stage')==2)
df_final_promo = df_final_2.select('promotion_id').distinct()
df = df_final_promo.join(df_manjianzhe_all, 'promotion_id', 'inner')

df = df.groupby(['promotion_id', 'item_sku_id', 'batch_id', 'promo_subtype']).agg(
	F.sum('baseline_sales').alias('baseline_sales'),
	F.sum('uplift_sales').alias('uplift_sales'),
    (F.sum('uplift_sales')-F.sum('promo_giveaway')-F.sum('coupon_giveaway')).alias('uplift_minus_discount'),
    (F.sum('baseline_sales') + F.sum('uplift_sales') -F.sum('promo_giveaway')-F.sum('coupon_giveaway')).alias('baseline_uplift_discount'),
    (F.sum('uplift_sales') -F.sum('promo_giveaway')-F.sum('coupon_giveaway') + F.sum('halo') - F.sum('cann_sales')).alias('incremental_halo')
).withColumn('bu_id',F.lit(latest_nosplit_dt)).withColumn('type', F.lit('2_stage_manjianzhe'))

df_2_stage_manjianzhe = df.select('item_sku_id', 'baseline_sales', 'uplift_sales', 'uplift_minus_discount',
                         'baseline_uplift_discount', 'incremental_halo', 'promotion_id','batch_id', 'bu_id', 'type')

df_promo_meimanjian = df_meimanjian_all.groupBy(['promotion_id', 'batch_id', 'promo_subtype']).agg(
	F.min('date').alias('min_dt'),
	F.max('date').alias('max_dt'))
df_promo_manjian = df_manjian_all.groupBy(['promotion_id', 'batch_id', 'promo_subtype']).agg(
	F.min('date').alias('min_dt'),
	F.max('date').alias('max_dt'))
df_promo_manjianzhe = df_manjianzhe_all.groupBy(['promotion_id', 'batch_id', 'promo_subtype']).agg(
	F.min('date').alias('min_dt'),
	F.max('date').alias('max_dt'))
# meimanjian
sql_promo_meimanjian = spark.sql(
	'''
		select promotion_id,
		max(threshold_money) as threshold_money,
		max(discount_money) as discount_money,
		max(first_ratio) as first_ratio,
		max(days) as days
		from
			(select promo_id as promotion_id,
			threshold_money,
			discount_money,
			discount_money/threshold_money as first_ratio,
			datediff(end_time,start_time) as days
			from app.self_total_promo_rule_detail
			where promo_subtype = 2
			and threshold_money>0
			and discount_money>0
			and start_time>='%s'
			and end_time<='%s') a
		group by promotion_id
	''' % (latest_trans_d_dt_365, latest_trans_d_dt)
)

# manjian
sql_promo_manjian = spark.sql(
	'''
		select promotion_id,
		max(threshold_money) as threshold_money,
		max(discount_money) as discount_money,
		max(first_ratio) as first_ratio,
		max(days) as days
		from
			(select promo_id as promotion_id,
			threshold_money,
			discount_money,
			discount_money/threshold_money as first_ratio,
			datediff(end_time,start_time) as days
			from app.self_total_promo_rule_detail
			where promo_subtype in (1,4)
			and threshold_money>0
			and discount_money>0
			and start_time>='%s'
			and end_time<='%s') a
		group by promotion_id
	''' % ( latest_trans_d_dt_365, latest_trans_d_dt)
)

# manjianzhe
sql_promo_manjianzhe = spark.sql(
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
			where promo_subtype = 15
			and threshold_num>0
			and discount_rate>0
			and start_time>='%s'
			and end_time<='%s') a
		group by promotion_id
	''' % (latest_trans_d_dt_365, latest_trans_d_dt)
)

# coupon
sql_promo_coupon = spark.sql(
	'''
	select batch_id, cps_face_value, consume_lim,first_ratio as first_ratio_coupon,
	datediff(valid_end_tm,valid_start_tm) as days
	FROM
	gdm.gdm_m07_cps_batch_da
	where dt='%s' and coupon_style = 0
	''' % (latest_trans_d_dt)
).distinct()

# and first_biz_type_id=3

df_promo_meimanjian_all = df_promo_meimanjian.join(sql_promo_meimanjian,'promotion_id', 'inner').join(sql_promo_coupon, 'batch_id', 'inner').\
withColumnRenamed('threshold_money', 'threshold_money_promo').withColumnRenamed('discount_money', 'discount_money_promo')\
.withColumnRenamed('first_ratio', 'first_ratio_promo').withColumnRenamed('consume_lim', 'threshold_money_coupon')\
.withColumnRenamed('cps_face_value', 'discount_money_coupon').withColumnRenamed('first_ratio_coupon', 'first_ratio_coupon')

df_promo_manjian_all = df_promo_manjian.join(sql_promo_manjian,'promotion_id', 'inner').join(sql_promo_coupon, 'batch_id', 'inner').\
withColumnRenamed('threshold_money', 'threshold_money_promo').withColumnRenamed('discount_money', 'discount_money_promo')\
.withColumnRenamed('first_ratio', 'first_ratio_promo').withColumnRenamed('consume_lim', 'threshold_money_coupon')\
.withColumnRenamed('cps_face_value', 'discount_money_coupon').withColumnRenamed('first_ratio_coupon', 'first_ratio_coupon')


df_promo_manjianzhe_all = df_promo_manjianzhe.join(sql_promo_manjianzhe,'promotion_id', 'inner').join(sql_promo_coupon, 'batch_id', 'inner').\
withColumnRenamed('threshold_num', 'threshold_money_promo').withColumnRenamed('discount_rate', 'discount_money_promo')\
.withColumnRenamed('first_ratio', 'first_ratio_promo').withColumnRenamed('consume_lim', 'threshold_money_coupon')\
.withColumnRenamed('cps_face_value', 'discount_money_coupon').withColumnRenamed('first_ratio_coupon', 'first_ratio_coupon')

df_meimanjian_simple = df_meimanjian.select(['promotion_id','bu_id', 'type']).distinct()
df_manjian_simple = df_manjian.select(['promotion_id','bu_id', 'type']).distinct()
df_2_stage_manjian_simple = df_2_stage_manjian.select(['promotion_id','bu_id', 'type']).distinct()
df_manjianzhe_simple = df_manjianzhe.select(['promotion_id','bu_id', 'type']).distinct()
df_2_stage_manjianzhe_simple = df_2_stage_manjianzhe.select(['promotion_id','bu_id', 'type']).distinct()

all_100_col = ['promotion_id', 'batch_id', 'min_dt', 'max_dt', 'threshold_money_promo', 'discount_money_promo', 'first_ratio_promo',\
              'threshold_money_coupon', 'discount_money_coupon', 'first_ratio_coupon', 'bu_id', 'type']

df_promo_meimanjian_all_100 = df_promo_meimanjian_all.join(df_meimanjian_simple, 'promotion_id', 'inner').select(all_100_col)
df_promo_manjian_all_100 = df_promo_manjian_all.join(df_manjian_simple, 'promotion_id', 'inner').select(all_100_col)
df_promo_2_stage_manjian_all_100 = df_promo_manjian_all.join(df_2_stage_manjian_simple, 'promotion_id', 'inner').select(all_100_col)
df_promo_manjianzhe_all_100 = df_promo_manjianzhe_all.join(df_manjianzhe_simple, 'promotion_id', 'inner').select(all_100_col)
df_promo_2_stage_manjianzhe_all_100 = df_promo_manjianzhe_all.join(df_2_stage_manjianzhe_simple, 'promotion_id', 'inner').select(all_100_col)

# detail table write
table_name = 'dev.dev_parellel_pormo_detail'
df_meimanjian.write.insertInto(table_name, overwrite=True)
df_manjian.write.insertInto(table_name, overwrite=True)
df_2_stage_manjian.write.insertInto(table_name, overwrite=True)
df_manjianzhe.write.insertInto(table_name, overwrite=True)
df_2_stage_manjianzhe.write.insertInto(table_name, overwrite=True)

# basic table write
table_name = 'dev.dev_parellel_pormo_basic'
df_promo_meimanjian_all_100.write.insertInto(table_name, overwrite=True)
df_promo_manjian_all_100.write.insertInto(table_name, overwrite=True)
df_promo_2_stage_manjian_all_100.write.insertInto(table_name, overwrite=True)
df_promo_manjianzhe_all_100.write.insertInto(table_name, overwrite=True)
df_promo_2_stage_manjianzhe_all_100.write.insertInto(table_name, overwrite=True)


