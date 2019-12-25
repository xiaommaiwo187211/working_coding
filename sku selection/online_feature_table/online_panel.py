
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession
from pyspark import SparkFiles
from pyspark.sql.types import *
from pyspark.storagelevel import StorageLevel
from pyspark.sql import functions as F
import os
import datetime
import pandas as pd
name = locals()

spark = SparkSession.builder\
					.appName("spark_test")\
					.enableHiveSupport()\
					.getOrCreate()

spark.conf.set("hive.exec.dynamic.partition","true")
spark.conf.set("hive.exec.dynamic.partition.mode","nonstrict")


# ## 一、覆盖sku范围，部门品类分布
# ## 二、前一个月promotion_id的黑名单率

# ### 定义函数统一解决问题，定义写表函数

# In[ ]:


# 做函数统一解决不同粒度的统计问题
def calculate(base_info,sku_perform,sku_type,groupby_words):

    # groupby的字段列表
    all_group_words = ['bu_id','bu_name','cid1','cid1_name','cid2','cid2_name','cid3','cid3_name','sale_amt_band']
    # 写表规整字段顺序
    result_words = all_group_words + ['sku_shelf','all_shelf','shelf_rate','sku_sale','all_sale','sale_rate','black_rate','duration','update_dt','dt','type']

    sku_perform = sku_perform.join(base_info.filter(F.col('sale_flag')==1).select(['item_sku_id']+all_group_words),'item_sku_id','inner')
    # 若是all代表全部
    if groupby_words == ['all']:
        a_all = base_info.filter(F.col('shelf_flag')==1).count()
        a = base_info.filter((F.col('shelf_flag')==1)&(F.col('table_flag')==1)).count()
        
        b_all = base_info.filter(F.col('sale_flag')==1).count()
        b = base_info.filter((F.col('sale_flag')==1)&(F.col('table_flag')==1)).count()
        
        c_all = sku_perform.count()
        c = sku_perform.filter(F.col('black_flag')==1).count()
        result = spark.createDataFrame(pd.DataFrame({'bu_id':[-9999],'bu_name':['-9999'],'cid1':[-9999],'cid1_name':['-9999'],
              'cid2':[-9999],'cid2_name':['-9999'],'cid3':[-9999],'cid3_name':['-9999'],'sale_amt_band':['-9999'],'sku_shelf':[a],
              'all_shelf':[a_all],'sku_sale':[b],'all_sale':[b_all],'black_rate':[1.0*c/c_all]})) 
    # 其余情况全部为按照所给字段进行groupby
    else:
        # 输入bu_id或cid，给他们加上name
        def group_names(words):
            if 'bu' in words:
                return 'bu_name'
            elif 'band' in words:
                return 'sale_amt_band'
            else :
                return words+'_name'
            
        groupby_names = list(map(group_names,groupby_words))
        groupby_words = groupby_words+groupby_names
        # 计算对应粒度的四个值+黑名单率
        a_all = base_info.filter(F.col('shelf_flag')==1)\
                         .groupby(groupby_words)\
                         .agg(F.count('item_sku_id').alias('all_shelf'))
        
        a = base_info.filter((F.col('shelf_flag')==1)&(F.col('table_flag')==1))\
                     .groupby(groupby_words)\
                     .agg(F.count('item_sku_id').alias('sku_shelf'))
        
        b_all = base_info.filter(F.col('sale_flag')==1)\
                         .groupby(groupby_words)\
                         .agg(F.count('item_sku_id').alias('all_sale'))
        
        b = base_info.filter((F.col('sale_flag')==1)&(F.col('table_flag')==1))\
                     .groupby(groupby_words)\
                     .agg(F.count('item_sku_id').alias('sku_sale'))
        
        c = sku_perform.groupby(groupby_words)\
                       .agg(F.sum('black_flag').alias('black_sum'),
                            F.count('black_flag').alias('total'))\
                       .withColumn('black_rate',F.col('black_sum')/F.col('total'))\
                       .drop('total','black_sum')
        
        result = a_all.join(b_all,groupby_words,'left')\
                      .join(a,groupby_words,'left')\
                      .join(b,groupby_words,'left')\
                      .join(c,groupby_words,'left')
        
        for name in groupby_words:
            result = result.filter(F.col(name).isNotNull())
        
        result = result.fillna(0)
        
        # 为这次没有groupby的字段加上-9999，区分name还是id
        other_words = list(set(all_group_words)-set(groupby_words))
        for word in other_words:
            if 'name' in other_words:
                result = result.withColumn(word,F.lit('-9999'))
            elif 'band' in other_words:
                result = result.withColumn(word,F.lit('-9999'))
            else :
                result = result.withColumn(word,F.lit(-9999))
    # 计算rate，加上分区字段，0/0的补充为0
    result = result.withColumn('shelf_rate',F.col('sku_shelf')/F.col('all_shelf'))\
               .withColumn('sale_rate',F.col('sku_sale')/F.col('all_sale'))\
               .fillna(0)\
               .withColumn('duration',F.lit(shelf_start_dt+'~'+shelf_end_dt))\
               .withColumn('update_dt',F.lit(now_day))\
               .withColumn('dt',F.lit(last_dt))\
               .withColumn('type',F.lit(sku_type))\
               .fillna(0)\
               .select(result_words)
    return result

# ## 指标看板

# In[ ]:


for sku_type in ['self','book_self']:
    last_dt = spark.sql('''select max(dt) from dev.dev_empty_bottle_black_selection_%s '''%sku_type).collect()[0][0]
    nosplit_dt = last_dt
    sku_table = spark.sql('''select item_sku_id from dev.dev_empty_bottle_black_selection_%s where dt='%s'  '''%(sku_type,last_dt))
    sku_table = sku_table.withColumn('table_flag',F.lit(1))
    now_day = datetime.datetime.now().strftime('%Y-%m-%d')
    # 读取SKU的最新上下柜状态信息
    oneday = datetime.timedelta(days=1)
    for num in range(1,8):
        if (datetime.datetime.strptime(last_dt,'%Y-%m-%d') - oneday*num).strftime('%w') == '0':
            shelf_end_dt = (datetime.datetime.strptime(last_dt,'%Y-%m-%d') - oneday*num).strftime('%Y-%m-%d')
            break
    shelf_start_dt = (datetime.datetime.strptime(shelf_end_dt,'%Y-%m-%d') - oneday*6).strftime('%Y-%m-%d')

    # 前一周上过柜的sku
    if sku_type == 'self':
        sku_shelf = spark.sql('''
        select 
            sku_id as item_sku_id
        from dev.self_sku_det_da
        where sku_type = 1
        and sku_status_cd = 3001
        and dt>='%s'
        and dt<='%s'
        '''%(shelf_start_dt,shelf_end_dt)).distinct()
    else:
        sku_shelf = spark.sql('''
        select 
            sku_id as item_sku_id
        from dev.self_sku_det_da
        where sku_type in (2,3)
        and sku_status_cd = 3001
        and dt>='%s'
        and dt<='%s'
        '''%(shelf_start_dt,shelf_end_dt)).distinct()    
    sku_shelf = sku_shelf.withColumn('shelf_flag',F.lit(1))

    if sku_type == 'self':       
        # 前一周有销量的sku
        sku_sale = spark.sql('''
        select 
        item_sku_id
        from app.app_pa_performance_nosplit_self_deal_price
        where dt = '%s'
        and date>='%s'
        and date<='%s'
        '''%(nosplit_dt,shelf_start_dt,shelf_end_dt)).distinct()
        sku_sale = sku_sale.withColumn('sale_flag',F.lit(1))

        # 前一周有的sku促销表现
        perform = spark.sql('''
        select 
        item_sku_id,
        promotion_id,
        batch_id,
        uplift_sales - promo_giveaway - coupon_giveaway as uplift_minus
        from app.app_pa_performance_nosplit_self_deal_price
        where dt = '%s'
        and date>='%s'
        and date<='%s'
        '''%(nosplit_dt,shelf_start_dt,shelf_end_dt))
    else :
        # 前一周有销量的sku
        sku_sale = spark.sql('''
        select 
        item_sku_id
        from app.app_pa_performance_nosplit_%s
        where dt = '%s'
        and date>='%s'
        and date<='%s'
        '''%(sku_type,nosplit_dt,shelf_start_dt,shelf_end_dt)).distinct()
        sku_sale = sku_sale.withColumn('sale_flag',F.lit(1))

        # 前一周有的sku促销表现
        perform = spark.sql('''
        select 
        item_sku_id,
        promotion_id,
        batch_id,
        uplift_sales - promo_giveaway - coupon_giveaway as uplift_minus
        from app.app_pa_performance_nosplit_%s
        where dt = '%s'
        and date>='%s'
        and date<='%s'
        '''%(sku_type,nosplit_dt,shelf_start_dt,shelf_end_dt))
    sku_perform_1 = perform.groupby('item_sku_id','promotion_id').agg(F.sum('uplift_minus').alias('uplift_minus'))\
                           .withColumn('black_flag',F.when(F.col('uplift_minus')<0,1).otherwise(F.lit(0))).drop('uplift_minus')
    sku_perform_2 = perform.groupby('item_sku_id','batch_id').agg(F.sum('uplift_minus').alias('uplift_minus'))\
                           .withColumn('black_flag',F.when(F.col('uplift_minus')<0,1).otherwise(F.lit(0))).drop('uplift_minus')

    sku_perform = sku_perform_1.select('item_sku_id','promotion_id','black_flag').union(sku_perform_2.select('item_sku_id','batch_id','black_flag'))

    # sku所属部门品类信息
    if sku_type == 'self':
        sku_info = spark.sql('''
        select 
            sku_id as item_sku_id,
            bu_id,
            bu_name,
            cid1,
            cid1_name,
            cid2,
            cid2_name,
            cid3,
            cid3_name
        from dev.self_sku_det_da
        where sku_type = 1
        and dt='%s'
        '''%shelf_end_dt)
        
        band_info = spark.sql('''
        select 
            sku_id as item_sku_id,
            sale_amt_band
        from app.self_sku_width_det
        where sku_type = 1
        and dt='%s'
        '''%shelf_end_dt).distinct()        
    else:
        sku_info = spark.sql('''
        select 
            sku_id as item_sku_id,
            bu_id,
            bu_name,
            cid1,
            cid1_name,
            cid2,
            cid2_name,
            cid3,
            cid3_name
        from dev.self_sku_det_da
        where sku_type in (2,3)
        and dt='%s'
        '''%shelf_end_dt)
        # 销售band表没有sku_type=3的分区，为了和上面保持一致加入3
        band_info = spark.sql('''
        select 
            sku_id as item_sku_id,
            sale_amt_band
        from app.self_sku_width_det
        where sku_type in (2,3)
        and dt='%s'
        '''%shelf_end_dt).distinct()
    
    sku_info = sku_info.join(band_info,'item_sku_id','left').fillna('other',subset=['sale_amt_band'])
    
    base_info = sku_info.join(sku_table,'item_sku_id','left')\
                        .join(sku_shelf,'item_sku_id','left')\
                        .join(sku_sale,'item_sku_id','left')\
                        .fillna(0)\
                        .filter((F.col('table_flag')!=0)|(F.col('shelf_flag')!=0)|(F.col('sale_flag')!=0))


    # 按照不同粒度进行union
    panel = calculate(base_info,sku_perform,sku_type,['all']).union(calculate(base_info,sku_perform,sku_type,['bu_id']))\
                                                             .union(calculate(base_info,sku_perform,sku_type,['bu_id','cid1']))\
                                                             .union(calculate(base_info,sku_perform,sku_type,['bu_id','cid1','cid2']))\
                                                             .union(calculate(base_info,sku_perform,sku_type,['bu_id','cid1','cid2','cid3']))\
                                                             .union(calculate(base_info,sku_perform,sku_type,['bu_id','cid1','cid2','cid3','sale_amt_band']))
    panel.write.insertInto('dev.dev_empty_bottle_black_selection_panel',overwrite=True)
