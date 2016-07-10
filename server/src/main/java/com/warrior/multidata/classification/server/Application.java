package com.warrior.multidata.classification.server;

import org.apache.commons.cli.*;
import org.apache.log4j.Logger;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;

import java.io.InputStream;

/**
 * Created by warrior on 03/07/16.
 */
@SpringBootApplication
public class Application {

    private static final Logger LOGGER = Logger.getLogger(Application.class);

    private static final Options OPTIONS = new Options();
    private static final Option GENDER = Option.builder("g")
            .longOpt("gender-model")
            .required()
            .hasArg()
            .desc("path to built relationship classifier model")
            .build();
    private static final Option RELATIONSHIP = Option.builder("r")
            .longOpt("relationship-model")
            .required()
            .hasArg()
            .desc("path to built relationship classifier model")
            .build();

    private static Instances genderInstances;
    private static Instances relationshipInstances;

    private static Classifier genderClassifier;
    private static Classifier relationshipClassifier;

    static {
        OPTIONS.addOption(GENDER);
        OPTIONS.addOption(RELATIONSHIP);

        InputStream stream = Application.class.getClassLoader().getResourceAsStream("attrs.arff");
        try {
            Instances instances = ConverterUtils.DataSource.read(stream);
            Attribute genderAttr = instances.attribute("gender");
            Attribute relationshipAttr = instances.attribute("relationship");

            genderInstances = new Instances(instances);
            relationshipInstances = new Instances(instances);

            genderInstances.setClassIndex(genderAttr.index());
            genderInstances.deleteAttributeAt(relationshipAttr.index());
            relationshipInstances.setClassIndex(relationshipAttr.index());
            relationshipInstances.deleteAttributeAt(genderAttr.index());
        } catch (Exception e) {
            LOGGER.fatal(e);
            System.exit(1);
        }
    }

    public static void main(String[] args) {
        CommandLineParser parser = new DefaultParser();
        try {
            CommandLine line = parser.parse(OPTIONS, args, true);
            genderClassifier = load(line.getOptionValue(GENDER.getOpt()));
            relationshipClassifier = load(line.getOptionValue(RELATIONSHIP.getOpt()));

        } catch (ParseException e) {
            LOGGER.fatal(e.getMessage());
            HelpFormatter help = new HelpFormatter();
            help.printHelp("help", OPTIONS);
            System.exit(1);
        }
        SpringApplication.run(Application.class, args);
    }

    private static Classifier load(String path) {
        try {
            return (Classifier) SerializationHelper.read(path);
        } catch (Exception e) {
            LOGGER.fatal(e.getMessage(), e);
            System.exit(1);
        }
        return null;
    }

    public static Classifier getGenderClassifier() {
        return genderClassifier;
    }

    public static Classifier getRelationshipClassifier() {
        return relationshipClassifier;
    }

    public static Instances getGenderInstances() {
        return genderInstances;
    }

    public static Instances getRelationshipInstances() {
        return relationshipInstances;
    }
}
