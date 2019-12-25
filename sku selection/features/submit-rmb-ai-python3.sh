#!/bin/sh
spark-submit \
--master yarn-cluster \
--conf spark.executor.instances=200 \
--conf spark.executor.memory=16g \
--conf spark.executor.cores=4 \
--conf spark.driver.cores=8 \
--conf spark.driver.memory=16g \
--conf spark.yarn.am.cores=8 \
--conf spark.yarn.am.memory=16g \
--conf spark.default.parallelism=1000 \
--conf spark.sql.shuffle.partitions=1000 \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.network.timeout=600s \
--conf spark.rpc.askTimeout=1200s \
--conf spark.rpc.lookupTimeout=600s \
--conf spark.rpc.numRetries=10 \
--conf spark.shuffle.io.maxRetries=10 \
--conf spark.pyspark.python=python3.5 \
--conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
--conf spark.network.timeout=1500 \
--conf spark.sql.hive.mergeFiles=true \
$@