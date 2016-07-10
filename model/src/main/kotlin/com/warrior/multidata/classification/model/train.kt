package com.warrior.multidata.classification.model

import kotlinx.support.jdk8.collections.parallelStream
import weka.classifiers.Classifier
import weka.classifiers.evaluation.Evaluation
import weka.classifiers.meta.FilteredClassifier
import weka.classifiers.meta.Stacking
import weka.classifiers.meta.Vote
import weka.classifiers.trees.RandomForest
import weka.core.Instances
import weka.filters.Filter
import weka.filters.MultiFilter
import weka.filters.unsupervised.attribute.Remove
import weka.filters.unsupervised.instance.RemoveWithValues
import java.io.File
import java.util.*

/**
 * Created by warrior on 04/07/16.
 */

fun main(args: Array<String>) {
    searchTreeNumber(Class.GENDER)
    trainSingleModels(Class.GENDER)
    fullInstancesTrain(Class.GENDER)
    vote(Class.GENDER)
    stacking(Class.GENDER)
    boosting(Class.GENDER)

    searchTreeNumber(Class.RELATIONSHIP)
    trainSingleModels(Class.RELATIONSHIP)
    fullInstancesTrain(Class.RELATIONSHIP)
    vote(Class.RELATIONSHIP)
    stacking(Class.RELATIONSHIP)
    boosting(Class.RELATIONSHIP)
}


private fun searchTreeNumber(c: Class) {
    val train = load("$DATASET_FOLDER/fullTrain${c.suffix}.arff")
    DatasetInfo.values()
            .map { info ->
                val (num, fMeasure) = (10..150 step 10)
                        .toList()
                        .parallelStream()
                        .map { num ->
                            val classifier = createClassifier(info, randomForest(num))
                            val evaluation = Evaluation(train)
                            evaluation.crossValidateModel(classifier, train, 10, Random())
                            val fMeasure = evaluation.unweightedMacroFmeasure()
                            println("${info.datasetName}: $num -> $fMeasure")
                            Pair(num, fMeasure)
                        }
                        .max({ f, s -> f.second.compareTo(s.second) })
                        .get()
                Triple(info.datasetName, num, fMeasure)
            }
            .forEach {
                val (name, num, fMeasure) = it
                println("$name. best tree number $num with $fMeasure")
            }
}

private fun trainSingleModels(c: Class) {
    val (test, train) = loadDataset(c)

    val modelFolder = "$MODEL_FOLDER/${c.suffix}"
    File(modelFolder).mkdirs()

    Arrays.stream(DatasetInfo.values())
            .parallel()
            .map { info ->
                val forestSize = forestSize(c, info)
                val classifier = createClassifier(info, randomForest(forestSize))
                info to classifier
            }
            .forEach {
                val (info, classifier) = it
                buildAndEvaluate(classifier, train, test, info.datasetName)
                saveModel(classifier, "$modelFolder/${info.datasetName}.model")
            }
}

private fun fullInstancesTrain(c: Class) {
    val (train, test) = loadDataset(c)

    val classifier = RandomForest()
    classifier.numExecutionSlots = cores()
    classifier.debug = true
    buildAndEvaluate(classifier, train, test, "full")

    val modelFolder = "$MODEL_FOLDER/${c.suffix}"
    File(modelFolder).mkdirs()
    saveModel(classifier, "$modelFolder/full.model")
}

private fun vote(c: Class) {
    val (train, test) = loadDataset(c)

    val modelFolder = "$MODEL_FOLDER/${c.suffix}"
    File(modelFolder).mkdirs()

    val vote = Vote()

    Arrays.stream(DatasetInfo.values())
            .forEach { info ->
                val classifier = loadModel("$modelFolder/${info.datasetName}.model")
                vote.addPreBuiltClassifier(classifier)
            }

    buildAndEvaluate(vote, train, test, "vote")
    saveModel(vote, "$modelFolder/vote.model")
}

private fun stacking(c: Class, metaClassifier: Classifier = RandomForest()) {
    val (train, test) = loadDataset(c)

    val stacking = Stacking()
    stacking.metaClassifier = metaClassifier
    stacking.classifiers = DatasetInfo.values()
            .map { info ->
                val forestSize = forestSize(c, info)
                createClassifier(info, randomForest(forestSize))
            }.toTypedArray()
    stacking.numExecutionSlots = Math.min(cores(), stacking.classifiers.size)

    stacking.debug = true
    buildAndEvaluate(stacking, train, test, "stacking")

    val modelFolder = "$MODEL_FOLDER/${c.suffix}"
    File(modelFolder).mkdirs()
    saveModel(stacking, "$modelFolder/stacking(${metaClassifier.javaClass.simpleName}).model")
}

private fun boosting(c: Class) {
    val (test, train) = loadDataset(c)

    val boosting = AdaBoosting()
    val classifiers = DatasetInfo.values().map { info ->
                val forestSize = forestSize(c, info)
                createClassifier(info, randomForest(forestSize, cores()))
            }
    boosting.setClassifiers(classifiers)
    boosting.debug = true
    buildAndEvaluate(boosting, train, test, "boosting")

    val modelFolder = "$MODEL_FOLDER/${c.suffix}"
    File(modelFolder).mkdirs()
    saveModel(boosting, "$modelFolder/boosting.model")
}

private fun loadDataset(c: Class): Pair<Instances, Instances> {
    val train = load("$DATASET_FOLDER/fullTrain${c.suffix}.arff")
    val test = load("$DATASET_FOLDER/fullTest${c.suffix}.arff")
    return Pair(train, test)
}

private fun randomForest(size: Int, threads: Int = 1): RandomForest {
    val classifier = RandomForest()
    classifier.numIterations = size
    classifier.numExecutionSlots = threads
    return classifier
}

private fun buildAndEvaluate(classifier: Classifier, train: Instances, test: Instances, name: String) {
    println("--- $name start ---")
    classifier.buildClassifier(train)
    val evaluation = Evaluation(test)
    evaluation.evaluateModel(classifier, test)
    println("--- $name end ---")
    printEvaluationResult(evaluation, test)
}

private fun printEvaluationResult(evaluation: Evaluation, test: Instances) {
    println("accuracy: " + evaluation.pctCorrect())
    for (i in 0 until test.numClasses()) {
        println("f-measure for '${test.classAttribute().value(i)}': ${evaluation.fMeasure(i)}")
    }
    println("macro f-measure: ${evaluation.unweightedMacroFmeasure()}")
}

private fun createClassifier(info: DatasetInfo, baseClassifier: Classifier): FilteredClassifier {
    val classifier = FilteredClassifier()
    classifier.filter = createFilter(info)
    classifier.classifier = baseClassifier
    return classifier
}

private fun createFilter(info: DatasetInfo): Filter {
    val removeAttrs = removeAttrs(info)
    val removeInstances = removeMissingValues()
    val filter = MultiFilter()
    filter.filters = arrayOf(removeAttrs, removeInstances)
    return filter
}

private fun removeAttrs(info: DatasetInfo): Remove {
    val remove = Remove()
    remove.attributeIndices = info.indices
    remove.invertSelection = true
    return remove
}

private fun removeMissingValues(): RemoveWithValues {
    val filter = RemoveWithValues()
    filter.attributeIndex = "first"
    filter.matchMissingValues = true
    filter.splitPoint = Double.NEGATIVE_INFINITY
    return filter
}

private fun forestSize(c: Class, info: DatasetInfo): Int = when (c) {
    Class.GENDER -> info.forestSizeGender
    Class.RELATIONSHIP -> info.forestSizeRelationship
}
