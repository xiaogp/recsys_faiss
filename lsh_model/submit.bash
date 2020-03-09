#!/bin/bash
cd /path
spark-submit --class com.mycom.recsys.content_similar \
--master local[*] \
--conf spark.sql.shuffle.partitions=150 \
--conf spark.default.parallelism=150 \
--executor-cores 3 \
--num-executors 3 \
--executor-memory 1g \
recsys-1.0-SNAPSHOT.jar "/path/config.Properities"
