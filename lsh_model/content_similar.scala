package com.mycom.recsys

import scala.io.Source
import java.util.Properties
import java.io.FileInputStream

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{Word2Vec, BucketedRandomProjectionLSH, Normalizer}
import org.apache.spark.sql.Row
import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.ml.linalg.{Vector, Vectors}

import org.ansj.recognition.impl.StopRecognition
import org.ansj.splitWord.analysis.ToAnalysis


object content_similar {

  def getStopWord(stopPath: String): StopRecognition = {
    val stop = new StopRecognition()
    stop.insertStopNatures("w") // 过滤掉标点
    stop.insertStopNatures("null") // 过滤null词性
    stop.insertStopNatures("m") // 过滤量词
    stop.insertStopRegexes("^[a-zA-Z]{1,}") // 过滤英文字母
    val source = Source.fromFile(stopPath)
    for (line <- source.getLines()) {
      stop.insertStopWords(line)
    }
    stop
  }

  def getProPerties(proPath: String) = {
    val properties: Properties = new Properties()
    properties.load(new FileInputStream(proPath))
    properties
  }

  def main(args: Array[String]): Unit = {
    Logger.getRootLogger.setLevel(Level.WARN)

    val properties = getProPerties(args(0))
    val stopPath = properties.getProperty("stopPath")  // 停止用词路径
    val w2v_window_size = properties.getProperty("w2v_window_size").toInt
    val w2v_iter = properties.getProperty("w2v_iter").toInt
    val w2v_vector_size = properties.getProperty("w2v_vector_size").toInt
    val w2v_min_count = properties.getProperty("w2v_min_count").toInt
    val lsh_bucket_len =  properties.getProperty("lsh_bucket_len").toInt
    val lsh_num_hash =  properties.getProperty("lsh_num_hash").toInt
    val lsh_threshold = properties.getProperty("lsh_threshold").toDouble

    val spark = SparkSession.builder()
      .appName("content_similar")
      .master("local[*]")
      .getOrCreate()
    import spark.implicits._

    val df = spark.read.format("jdbc")
      .option("url", "jdbc:mysql://localhost:3306/recsys?useUnicode=true&characterEncoding=utf8&autoReconnect=true&useSSL=false")
      .option("user", "root")
      .option("password", "password")
      .option("dbtable", "(select ITEM_NUM_ID, ITEM_NAME, PTY1_NAME, PTY2_NAME, PTY3_NAME, BRAND_NAME from goods_data) as t1")
      .load()

    val filter = getStopWord(stopPath)
    val seg_udf = udf((text: String) => (ToAnalysis.parse(text).recognition(filter).toStringWithOutNature(" ")))

    // 分词
    val df2 = df.withColumn("seg", concat_ws(" ", seg_udf($"ITEM_NAME"), $"PTY1_NAME", $"PTY2_NAME", $"PTY3_NAME"))
      .select($"ITEM_NUM_ID", $"seg")
      .withColumn("seg", split($"seg", " "))
      .cache()

    // 训练word2vec
    val word2Vec = new Word2Vec()
      .setInputCol("seg")
      .setOutputCol("res")
      .setWindowSize(w2v_window_size)
      .setMaxIter(w2v_iter)
      .setVectorSize(w2v_vector_size)
      .setMinCount(w2v_min_count)

    val model = word2Vec.fit(df2)

    // 向量归一化
    val normalizer = new Normalizer()
      .setInputCol("vector")
      .setOutputCol("normal_vector")
      .setP(1.0)

    val vector_df = normalizer.transform(model.getVectors).drop($"vector")

    // explode所有商品的词
    val df3 = df2.withColumn("word", explode($"seg"))

    // join
    val df4 = df3.join(vector_df, Seq("word"), "inner").select($"ITEM_NUM_ID", $"normal_vector")

    // 词向量相加
    val vec_sum_df = df4.rdd
      .map { case Row(k: Long, v: Vector) => (k, BDV(v.toDense.values)) }
      .foldByKey(BDV.zeros[Double](w2v_vector_size))(_ += _)
      .mapValues(v => Vectors.dense(v.toArray))
      .toDF("id", "vec")

    // LSH
    val brp = new BucketedRandomProjectionLSH()
      .setBucketLength(lsh_bucket_len)
      .setNumHashTables(lsh_num_hash)
      .setInputCol("vec")
      .setOutputCol("hashes")
    val brpModel = brp.fit(vec_sum_df)

    val brpDf = brpModel.approxSimilarityJoin(vec_sum_df, vec_sum_df, lsh_threshold, "EuclideanDistance")

    // 结果整理
    val getid_df = brpDf
      .sort($"EuclideanDistance")
      .withColumn("datasetA", udf((input: Row) => {input(0).toString}).apply($"datasetA"))
      .withColumn("datasetB", udf((input: Row) => {input(0).toString}).apply($"datasetB"))
      .drop("EuclideanDistance")
      .toDF("id_i", "id_j")
      .filter("id_i != id_j")
      .groupBy($"id_i")
      .agg(concat_ws(",", collect_list($"id_j")))
      .toDF("id", "recommend")

    getid_df.write.format("org.apache.phoenix.spark")
      .option("zkurl", "localhost:2181")
      .option("table", "RECSYS_SIMILAR_LSH")
      .mode("overwrite")
      .save()

    df2.unpersist()
    spark.close()

  }
}

