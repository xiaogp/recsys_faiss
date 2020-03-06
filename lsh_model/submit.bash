#!/bin/bash
cd /Users/path
spark-submit --class com.mycom.recsys.content_similar --master local[*] --conf spark.sql.shuffle.partitions=200 --conf spark.default.parallelism=200 --executor-cores 2 --num-executors 2 --executor-memory 1g recsys-1.0-SNAPSHOT.jar "/Users/path/config.Properities"
