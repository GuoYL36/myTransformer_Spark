/*
 * @Author: guoyilin
 * @Date: 2019-08-22
 * @Time: 14:24
 */
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.types._


class StringDenseVector(override val uid: String) extends UnaryTransformer[String, Vector, StringDenseVector] {
    def this(dict: String) = this(Identifiable.randomUID("String2DenseVector"))    // "String2DenseVector_"和一个随机数组成的标识符

    override protected  def validateInputType(inputType: DataType): Unit = {
        require(inputType == StringType, s"Input type must be string type but got $inputType.")
    }

    override protected def outputDataType: DataType = new VectorUDT

    override protected def createTransformFunc: String => Vector  = {
        convert
    }
    private def convert(text: String) = {
        var length = text.length()
        var a = text.substring(1,length-1).split(",").map(i => i.toDouble)
        Vectors.dense(a)
    }


}
