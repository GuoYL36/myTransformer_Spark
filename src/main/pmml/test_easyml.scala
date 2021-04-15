

import java.io.FileOutputStream

import javax.xml.transform.stream.StreamResult
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FSDataOutputStream, FileSystem, Path}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{Mytransformer, VectorAssembler}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.{DataFrame, Row, SaveMode, SparkSession}
import org.apache.spark.sql.functions.{col, split, udf}
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.{SparkConf, SparkContext}
import org.jpmml.model.{JAXBUtil, MetroJAXBUtil}
import org.jpmml.sparkml.PMMLBuilder

import scala.collection.mutable.ArrayBuffer

/*
 * @Author: guoyilin
 * @Date: 2021-04-09
 * @Time: 9:57    
 */

object test_easyml {

  def processdata(spark:SparkSession): Unit ={

    // 数据转换
    val str2Int: Map[String, Double] = Map(
      "Iris-setosa" -> 0.0,
      "Iris-versicolor" -> 1.0,
      "Iris-virginica" -> 2.0
    )
    var str2double = (x: String) => str2Int(x)
    var myFun = udf(str2double)
    val data = spark.read.textFile("D:\\gyl\\scalaProgram\\PMML\\iris1.txt").toDF()
        .withColumn("splitcol", split(col("value"), ","))
        .select(
          col("splitcol").getItem(0).as("sepal_length"),
          col("splitcol").getItem(1).as("sepal_width"),
          col("splitcol").getItem(2).as("petal_length"),
          col("splitcol").getItem(3).as("petal_width"),
          col("splitcol").getItem(4).as("label")
        )
        .withColumn("label", myFun(col("label")))
        .select(
          col("sepal_length").cast(DoubleType),
          col("sepal_width").cast(DoubleType),
          col("petal_length").cast(DoubleType),
          col("petal_width").cast(DoubleType),
          col("label").cast(DoubleType)
        )


    val data1 = data.na.drop()
    println("data: " + data1.count().toString)
    val schema = data1.schema
    println("data1 schema: " + schema)


    val features: Array[String] = Array("sepal_length", "sepal_width", "petal_length", "petal_width")
    //    // merge multi-feature to vector features
    val assembler: VectorAssembler = new VectorAssembler().setInputCols(features).setOutputCol("features")
    val data2 = assembler.transform(data1)
    println("data2 schema: " + data2.schema)
    println("assembler transform class: "+assembler.getClass )

    // convert features vector-data to string
    val convertFunction = (x: DenseVector) => {
      x.toString
    }
    val convertUDF = udf(convertFunction)
    val newdata = data2.withColumn("features", convertUDF(col("features")))
    println(newdata.schema)
    newdata.write.mode(SaveMode.Overwrite).format("parquet").save("D:\\gyl\\scalaProgram\\PMML\\data1.parquet")

  }

  def toPmml(data:DataFrame, output_model:String, model:RandomForestClassifier): Unit ={
    val firstRow: Array[Row] = data.select("features").take(1)
    val lengthOfFeature = firstRow(0)(0).asInstanceOf[DenseVector].size  // 获取特征数量

    val vec2str = org.apache.spark.sql.functions.udf((x:scala.collection.mutable.WrappedArray[Double]) => org.apache.spark.ml.linalg.Vectors.dense(x.toArray).toString)

    val vec2arr = org.apache.spark.sql.functions.udf(
      (x:org.apache.spark.ml.linalg.Vector) =>
        x.toDense.toArray
    )
    var data1 = data.withColumn("features",vec2arr(org.apache.spark.sql.functions.col("features")))

    // 还原特征列，并以col为开头进行命名
    val features0 = new ArrayBuffer[String]()
    for(i <- 0 until lengthOfFeature){
      val tmp = "col"+i.toString
      features0.+=(tmp)
      data1 = data1.withColumn(tmp,org.apache.spark.sql.functions.col("features").getItem(i))

    }
    val features: Array[String] = features0.toArray

    data1 = data1.withColumn("features", vec2str(org.apache.spark.sql.functions.col("features")))

    val mytransformer = new Mytransformer().setInputCol(features).setOutputCol("features")

    val pipeline = new Pipeline().setStages(Array(mytransformer,model))
    val pipelineModel = pipeline.fit(data1)

    val pmml = new PMMLBuilder(data1.schema, pipelineModel).build()

    val hadoopConf = new Configuration()
    val fs = FileSystem.get(hadoopConf)
    val output_path = output_model+"_pmml/randomForestModel.pmml"
    val fpath = new Path(output_path)
    if(fs.exists(fpath)){
      fs.delete(fpath, true)
    }

    val fout = fs.create(fpath)
    val fout1 = new StreamResult(fout)
    JAXBUtil.marshalPMML(pmml, fout1)
  }

  def PmmlToHdfs(sc: SparkSession): Unit ={

    val path = "/guoyilin/uci_data/process_data/"
    var data = sc.read.parquet(path)   // 该数据只有两列，一列为features，一列为label

    data.show(10)

    val rf = new RandomForestClassifier()
        .setLabelCol("label")
        .setFeaturesCol("features")
        .setMaxDepth(8)
        .setNumTrees(30)
        .setSeed(1234)
        .setMinInfoGain(0)
        .setMinInstancesPerNode(1)

    toPmml(data, "/guoyilin/test", rf)

//    val pipeline = new Pipeline().setStages(Array(mytransformer, rf))
//    //
//    val pipelineModel = pipeline.fit(data)
//
//    val pmml = new PMMLBuilder(data.schema, pipelineModel).build()
//    val targetFile = "/guoyilin/pipemodel.pmml"
//
//    // hdfs上创建文件路径
//    val hadoopConf: Configuration = new Configuration()  // 获取hadoop的配置文件，其值为hadoopConf: org.apache.hadoop.conf.Configuration = Configuration: core-default.xml, core-site.xml, mapred-default.xml, mapred-site.xml, yarn-default.xml, yarn-site.xml, hdfs-default.xml, hdfs-site.xml
//    val fs = FileSystem.get(hadoopConf)   // 从配置文件中获取文件系统，其值为fs: org.apache.hadoop.fs.FileSystem = DFS[DFSClient[clientName=DFSClient_NONMAPREDUCE_504658679_1, ugi=root (auth:SIMPLE)]]
//
//    val fpath = new Path(targetFile)  // 将字符串targetFile转换为hdfs上的路径
//    if(fs.exists(fpath)){
//      fs.delete(fpath, true)
//    }
//    val fout: FSDataOutputStream = fs.create(fpath)   // hdfs上创建路径targetFile
//
//    val fout1: StreamResult = new StreamResult(fout)
//    JAXBUtil.marshalPMML(pmml,fout1)
//
    println("pipelineModel success......")
  }



  def main(args:Array[String]): Unit ={
    val conf = new SparkConf()
    conf.setAppName("test_easyml")
    val sc = SparkSession.builder().config(conf).getOrCreate()
    import sc.implicits._


//    processdata(sc)
    PmmlToHdfs(sc: SparkSession)






  }
}
