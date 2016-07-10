package com.warrior.multidata.classification.server;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;
import weka.classifiers.Classifier;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import static com.warrior.multidata.classification.server.Application.*;

/**
 * Created by warrior on 03/07/16.
 */
@RestController
public class ClassificationRestController {

    @RequestMapping(path = "/classification", method = RequestMethod.POST)
    public ResponseEntity<Result> classification(@RequestBody String[] values) {
        if (values.length != getGenderInstances().numAttributes() - 1) {
            return new ResponseEntity<>(HttpStatus.BAD_REQUEST);
        }

        double[] doubleValues = toDoubleValues(values);

        String gender = classifyInstance(getGenderClassifier(), doubleValues, getGenderInstances());
        String relationship = classifyInstance(getRelationshipClassifier(), doubleValues, getRelationshipInstances());

        return ResponseEntity.ok(new Result(gender, relationship));
    }

    private double[] toDoubleValues(String[] values) {
        double[] doubleValues = new double[values.length + 1];
        for (int i = 0; i < values.length; i++) {
            String value = values[i];
            if (value.isEmpty() || "?".equals(value)) {
                doubleValues[i] = Utils.missingValue();
            } else {
                doubleValues[i] = Double.parseDouble(value);
            }
        }
        doubleValues[doubleValues.length - 1] = Utils.missingValue();
        return doubleValues;
    }

    private String classifyInstance(Classifier classifier, double[] values, Instances instances) {
        Instance instance = new DenseInstance(1.0, values);
        instance.setDataset(instances);

        try {
            double predictedClass = classifier.classifyInstance(instance);
            if (Utils.isMissingValue(predictedClass)) {
                return "";
            }
            return instances.classAttribute().value((int) predictedClass);
        } catch (Exception e) {
            throw new RuntimeException(e);
        } finally {
            instances.clear();
        }
    }
}
