#!/usr/bin/env python3
# coding:utf-8

import os
import subprocess
import re
import pyspark.sql.functions as F
from pyspark import SparkFiles


def save_result(df, table_name, partitioning_columns=[], write_mode='insert',
                spark=None, params=None):
    if params is None:
        params = dict()
    table_name = rename(table_name, params)
    if isinstance(partitioning_columns, str):
        partitioning_columns = [partitioning_columns]
    save_mode =  'overwrite' if ('overwrite' in params.keys()) and (params['overwrite'] == 1) else 'error'
    if write_mode == 'save':
        if len(partitioning_columns) > 0:
            df.repartition(*partitioning_columns).write.mode(save_mode).partitionBy(partitioning_columns).format('orc').saveAsTable(table_name)
        else:
            df.write.mode(save_mode).format('orc').saveAsTable(table_name)
        spark.sql('''ALTER TABLE %s SET TBLPROPERTIES ('author' = '%s')''' % (table_name, params['author']))
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


def save_hive_result(df, table_name, partitioning_columns=[], write_mode='insert',
                     spark=None, params=None):
    """
    This function use a fast way to save table,
    and only overwrite partition(s) in your dataframe, other partitions are safe
    :param df:
    :param table_name:
    :param partitioning_columns:
    :param write_mode:
    :param spark:
    :param params:
    :return: None
    """
    if params is None:
        params = {'author': 'pa'}
    try:
        # rename function, designed for protecting data in testing mode
        table_name = rename(table_name, params)
    except NameError:
        # rename function not found. it's ok, forget it
        pass
    if isinstance(partitioning_columns, str):
        partitioning_columns = [partitioning_columns]
    if write_mode == 'save':
        drop_sql = "DROP TABLE IF EXISTS %s " % table_name
        create_sql = "CREATE TABLE %s " % table_name
        columns_sql = "(%s)" % ','.join(["%s %s" % (x, y) for x,y in df.dtypes if x not in partitioning_columns])
        partition_sql = ""
        if len(partitioning_columns) > 0:
            partition_sql = "PARTITIONED BY (%s)" % ','.join(["%s string" % x for x,y in df.dtypes if x in partitioning_columns])
        orc_sql = """
            ROW FORMAT SERDE 'org.apache.hadoop.hive.ql.io.orc.OrcSerde'
            STORED AS
            INPUTFORMAT 'org.apache.hadoop.hive.ql.io.orc.OrcInputFormat'
            OUTPUTFORMAT 'org.apache.hadoop.hive.ql.io.orc.OrcOutputFormat'
        """
        author_sql = "TBLPROPERTIES ('author'='%s')" % params['author']
        final_sql = ' '.join([create_sql, columns_sql, partition_sql, orc_sql, author_sql])
        spark.sql(drop_sql)
        spark.sql(final_sql)
        write_mode = 'insert'
    if write_mode == 'insert':
        if len(partitioning_columns) > 0:
            spark.sql("set hive.exec.dynamic.partition=true")
            spark.sql("set hive.exec.dynamic.partition.mode=nonstrict")
            df.repartition(*partitioning_columns).write.insertInto(table_name, overwrite=True)
        else:
            df.write.insertInto(table_name, overwrite=False)
    else:
        raise ValueError('mode "%s" not supported ' % write_mode)


def convert_timestamp_to_date(df, cols):
    if type(cols) is not list:
        cols = [cols]
    for col in cols:
        df = df.withColumn(col, F.to_date(col))
    return df


def read_csv_by_dt(table, dt=None, start=None, end=None, spark=None, params=None, **kwargs):
    table_path = os.path.join(params['input_path'], table)
    paths = []
    existed = get_partition(table_path)
    if (dt is None) and (start is None) and (end is None):
        paths.extend(existed)
    elif dt is not None:
        path = os.path.join(table_path, dt)
        paths.append(path)
    elif (start is not None) or (end is not None):
        if start is None:
            start = '0000-00-00'
        if end is None:
            end = 'latest'
        start_path = os.path.join(table_path, start)
        end_path = os.path.join(table_path, end)
        between_path = [x for x in existed if start_path <= x <= end_path ]
        paths.extend(between_path)
    df = spark.read.csv(paths, **kwargs)
    return df


def get_partition(table):
    cmd = "hadoop fs -ls '%s'" % table
    out_bytes = subprocess.check_output(cmd, shell=True)
    out_string = out_bytes.decode('utf-8')
    if len(out_string) == 0:
        return None
    else:
        out_records = out_string.split('\n')
        out_files = [x.split(' ')[-1] for x in out_records[1:-1]]
        return out_files


def read_hive_by_dt(table, dt=None, start=None, end=None, spark=None, params=None):
    if (dt is None) and (start is None) and (end is None):
        start = '0000-00-00'
        end = '9999-99-99'
    elif dt == 'latest':
        start = params['update_end']
        end = '9999-99-99'
    elif dt is not None:
        start = dt
        end = dt
    elif (start is not None) or (end is not None):
        if start is None:
            start = '0000-00-00'
        if end is None:
            end = '9999-99-99'
    tail = params['update_end']
    df = spark.sql(params['querys'][table].format(start=start, end=end, tail=tail))
    return df


def read_table(table, dt=None, start=None, end=None, spark=None, params=None, **kwargs):
    if params['input_path'] == 'hive':
        df = read_hive_by_dt(table, dt=dt, start=start, end=end, spark=spark, params=params)
    elif params['input_path'].startswith('hdfs://'):
        df = read_csv_by_dt(table, dt=dt, start=start, end=end, spark=spark, params=params, **kwargs)
    else:
        raise ValueError('input_path "%s" not supported ' % params['input_path'])
    return df


def load_params(file_name, auto_date=False, spark=None):
    """
    file_name 必需，从yaml文件读取配置
    auto_date 可选，是否从hive表读取配置
    spark     可选，如果从hive表读取配置，需要传进来执行任务的spark对象
    """
    try:
        import yaml
        params = yaml.load(open(SparkFiles.get(file_name)))
        if auto_date:
            # 如果添加setup表中的日期配置，则会覆盖params.yaml文件中的设置
            setup_table = rename(params['setup_table'], params)
            row = spark.sql('select * from %s' % setup_table).collect()[0]
            params['update_start'] = row.update_start
            params['update_end'] = row.update_end
            params['update_origin'] = row.update_origin
    except ImportError:
        params = {}
    return params


def rename(old, params):
    """
    测试模式下，通过添加后缀，重命名表
    正常模式下，不改变表名
    """
    if params and 'test_suffix' in params:
        suffix = re.sub('\s', '', params['test_suffix'])
        if 0 < len(suffix) <= 5:
            # 加_000_是为了防止如下情况：表名AAA添加BBB后缀，变成AAA_BBB，而AAA_BBB刚好是另外一张表
            return old + '_000_' + suffix
        else:
            raise ValueError('test suffix "%s" not valid ' % params['test_suffix'])
    else:
        return old
