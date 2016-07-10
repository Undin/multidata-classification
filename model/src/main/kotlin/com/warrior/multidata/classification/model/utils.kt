package com.warrior.multidata.classification.model

import weka.classifiers.Classifier
import weka.core.Attribute
import weka.core.Instances
import weka.core.SerializationHelper
import weka.core.converters.ArffSaver
import weka.core.converters.ConverterUtils
import weka.core.converters.Saver
import java.io.File

/**
 * Created by warrior on 04/07/16.
 */
fun load(path: String): Instances {
    val instances = ConverterUtils.DataSource.read(path)
    instances.setClassIndex(instances.numAttributes() - 1)
    return instances
}

fun save(dataSet: Instances, dst: String, saver: Saver = ArffSaver()) {
    saver.setInstances(dataSet)
    saver.setFile(File(dst))
    saver.writeBatch()
}

fun saveModel(classifier: Classifier, dst: String) = SerializationHelper.write(dst, classifier)

@Suppress("UNCHECKED_CAST")
fun loadModel(dst: String): Classifier = SerializationHelper.read(dst) as Classifier

fun Attribute.clone(): Attribute = copy(name())

fun cores() = Runtime.getRuntime().availableProcessors()
