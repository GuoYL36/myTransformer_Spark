import java.io.*;
import java.nio.charset.Charset;
import java.util.*;

import org.apache.spark.ml.evaluation.Evaluator;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.glassfish.hk2.utilities.cache.Computable;
import org.jpmml.evaluator.*;
import org.xml.sax.SAXException;

import javax.imageio.IIOException;
import javax.xml.bind.JAXBException;


public class LoadPmml {
    private Evaluator loadPmml() {
        PMML pmml = new PMML();
        InputStream inputStream = null;
        try {
            inputStream = new FileInputStream("D:\\gyl\\scalaProgram\\PMML\\pipemodel.pmml");

        } catch(IOException e) {
            e.printStackTrace();
        }
        if(inputStream == null){
            return null;
        }
        InputStream is = inputStream;
        try{
            pmml = org.jpmml.model.PMMLUtil.unmarshal(is);
        } catch(SAXException e1){
            e1.printStackTrace();
        } catch (JAXBException e1){
            e1.printStackTrace();
        }finally {
            // close inputStream
            try {
                is.close();
            } catch (IIOException e) {
                e.printStackTrace();
            }
        }
        ModelEvaluatorFactory modelEvaluatorFactory = ModelEvaluatorFactory.newInstance();
        Evaluator evaluator = modelEvaluatorFactory.newModelEvaluator(pmml);
        pmml = null;
        return evaluator;
    }
    private int predict(Evaluator evaluator, double a, double b, double c, double d){
        Map<String, Double> data = new HashMap<String, Double>();
        data.put("sepal_length", a);
        data.put("sepal_width",b);
        data.put("petal_length",c);
        data.put("petal_width",d);
        List<InputField> inputFields = evaluator.getInputFields();
        // 提取模型里的原始特征名，从data画像中获取数据，作为模型的输入
        Map<FieldName, FieldValue> arguments = new LinkedHashMap<FieldName, FieldValue>();
        for (InputField inputField: inputFields) {
            FieldName inputFiledName = inputField.getName();
            Object rawValue = data.get(inputFiledName.getValue());
            FieldValue inputFieldValue = inputField.prepare(rawValue);
            arguments.put(inputFiledName, inputFieldValue);
        }
        Map<FieldName, ?> results = evaluator.evaluate(arguments);
        List<TargetField> targetFields = evaluator.getTargetFields();

        TargetField targetField = targetFields.get(0);
        FieldName targetFieldName = targetField.getName();

        Object targetFieldValue = results.get(targetFieldName);
        System.out.println("target: "+targetFieldName.getValue()+"value: "+targetFieldValue);
        int primitiveValue = -1;
        if(targetFieldValue instanceof Computable) {
            Computable computable = (Computable) targetFieldValue;
            primitiveValue = (Integer) computable.getResult();
        }
        System.out.println(a+" "+b+" "+c+" "+d+":"+primitiveValue);
        return primitiveValue;
    }

    public static void main(String args[]){
        PMMLDemo demo = new PMMLDemo();
        Evaluator model = demo.loadPmml();
        demo.predict(model, 0.0,0.0,0.0,0.0);
    }
}
