package com.warrior.multidata.classification.model

import com.fasterxml.jackson.core.JsonEncoding
import com.fasterxml.jackson.core.type.TypeReference
import com.fasterxml.jackson.databind.ObjectMapper
import kotlinx.support.jdk8.collections.parallelStream
import kotlinx.support.jdk8.streams.toList
import weka.classifiers.Classifier
import weka.classifiers.evaluation.Evaluation
import weka.classifiers.meta.FilteredClassifier
import weka.classifiers.meta.Stacking
import weka.classifiers.meta.Vote
import weka.classifiers.trees.RandomForest
import weka.core.Instances
import weka.filters.Filter
import weka.filters.MultiFilter
import weka.filters.supervised.instance.SMOTE
import weka.filters.unsupervised.attribute.Remove
import weka.filters.unsupervised.instance.RemoveWithValues
import java.io.File
import java.io.PrintWriter
import java.util.*

/**
 * Created by warrior on 04/07/16.
 */

var treeSizes: Map<Class, Map<DatasetInfo, Map<Metric, Int>>> = mapOf()

fun main(args: Array<String>) {
    treeSizes = searchTreeSize(Class.values().asList())

    val resultFolder = File("results")
    if (!resultFolder.exists()) {
        resultFolder.mkdir()
    }
    for (c in listOf(Class.AGE_GROUP, Class.GENDER, Class.RELATIONSHIP, Class.EDUCATION_LEVEL_BINARY, Class.EDUCATION_LEVEL_TERNARY/*, Class.OCCUPATION*/)) {
        println("---- ${c.name} ----")
        val results = ArrayList<EvaluationResult>()
        for (useOversampling in listOf(false, true)) {
            results += trainSingleModels(c, useOversampling, Metric.values())
            results += boosting(c, useOversampling, Metric.values())
            results += stacking(c, useOversampling, Metric.values())
            results += vote(c, useOversampling, Metric.values())
        }

        PrintWriter(File(resultFolder, "${c.name}.txt")).use {
            val resultMap = results.groupBy { it.javaClass }
            for ((k, v) in resultMap) {
                val result = v.maxBy { it.value }
                println(result)
                it.println(result)
            }
        }
    }
}

private fun searchTreeSize(classes: List<Class>): Map<Class, Map<DatasetInfo, Map<Metric, Int>>> {
    val mapper = ObjectMapper()
    val treeNumbers = HashMap<Class, Map<DatasetInfo, Map<Metric, Int>>>()
    val treeSizeFolder = File("tree-sizes")
    if (!treeSizeFolder.exists()) {
        treeSizeFolder.mkdir()
    }
    for (c in classes) {
        val file = File(treeSizeFolder, "${c.suffix}.json")
        if (file.exists()) {
            treeNumbers[c] = mapper.readValue(file, object : TypeReference<Map<DatasetInfo, Map<Metric, Int>>>() {})
        } else {
            val generator = mapper.factory.createGenerator(file, JsonEncoding.UTF8)
            generator.use { g ->
                g.writeStartObject()
                val map = searchTreeSize(c, *Metric.values())
                treeNumbers[c] = map
                g.writeObject(map)
                g.flush()
                g.writeEndObject()
            }
        }
    }

    return treeNumbers
}

private fun searchTreeSize(c: Class, vararg metrics: Metric): Map<DatasetInfo, Map<Metric, Int>> {
    println("search for $c")
    val train = load("$DATASET_FOLDER/fullTrain${c.suffix}.arff")
    return DatasetInfo.values()
            .map { info ->
                val results = (10..150 step 10)
                        .toList()
                        .parallelStream()
                        .map { num ->
                            val classifier = createClassifier(info, randomForest(num))
                            val evaluation = Evaluation(train)
                            evaluation.crossValidateModel(classifier, train, 10, Random())
                            val results = metrics.map {
                                evaluation.macro(it.fn)
                            }

                            println("${info.datasetName}: $num ->")
                            for ((i, m) in metrics.withIndex()) {
                                println("    ${m.name}: ${results[i]}")
                            }
                            Pair(num, results)
                        }
                        .toList()

                val bestResults = DoubleArray(metrics.size) { -1.0 }
                val bestSizes = IntArray(metrics.size)
                for ((num, values) in results) {
                    for ((i, v) in values.withIndex()) {
                        if (bestResults[i] < v) {
                            bestResults[i] = v
                            bestSizes[i] = num
                        }
                    }
                }

                val map = HashMap<Metric, Int>()
                println("-- ${info.datasetName}. best tree sizes: --")
                for ((i, m) in metrics.withIndex()) {
                    map[m] = bestSizes[i]
                    println("    $m -> ${bestSizes[i]}(${bestResults[i]})")
                }
                Pair(info, map)
            }
            .toMap()
}

private fun trainSingleModels(c: Class, useOversampling: Boolean, metrics: Array<Metric>): List<EvaluationResult> {
    val (train, test) = loadDataset(c, useOversampling)

    val modelFolder = "$MODEL_FOLDER/${c.suffix}"
    File(modelFolder).mkdirs()

    val results = ArrayList<EvaluationResult>()
    for (m in metrics) {
        results += Arrays.stream(DatasetInfo.values())
                .parallel()
                .map { info ->
                    val forestSize = treeSizes.get(c)?.get(info)?.get(m)!!
                    val classifier = createClassifier(info, randomForest(forestSize))
                    val res = buildAndEvaluate(classifier, train, test, info.datasetName, m)
                    saveModel(classifier, "$modelFolder/${info.datasetName}-$m.model")
                    res
                }
                .toList()
                .flatten()
    }
    results.forEach { it.useOversampling = useOversampling }
    return results
}

private fun vote(c: Class, useOversampling: Boolean, metrics: Array<Metric>): List<EvaluationResult> {
    return train(c, useOversampling, metrics) { c, m, modelFolder ->
        val vote = Vote()
        Arrays.stream(DatasetInfo.values())
                .forEach { info ->
                    val classifier = loadModel("$modelFolder/${info.datasetName}-$m.model")
                    vote.addPreBuiltClassifier(classifier)
                }
        "vote" to vote
    }
}

private fun stacking(c: Class, useOversampling: Boolean, metrics: Array<Metric>,
                      metaClassifier: Classifier = RandomForest()): List<EvaluationResult> {
    return train(c, useOversampling, metrics) { c, m, f ->
        val stacking = Stacking()
        stacking.metaClassifier = metaClassifier
        stacking.classifiers = DatasetInfo.values()
                .map { info ->
                    val forestSize = treeSizes.get(c)?.get(info)?.get(m)!!
                    createClassifier(info, randomForest(forestSize))
                }.toTypedArray()
        stacking.numExecutionSlots = Math.min(cores(), stacking.classifiers.size)
        "stacking" to stacking
    }
}

private fun boosting(c: Class, useOversampling: Boolean, metrics: Array<Metric>): List<EvaluationResult> {
    return train(c, useOversampling, metrics) { c, m, f ->
        val boosting = AdaBoosting()
        val classifiers = DatasetInfo.values().map { info ->
            val forestSize = treeSizes.get(c)?.get(info)?.get(m)!!
            createClassifier(info, randomForest(forestSize, cores()))
        }
        boosting.setClassifiers(classifiers)
        boosting.debug = true
        "boosting" to boosting
    }
}

private fun train(c: Class, useOversampling: Boolean, metrics: Array<Metric>,
                  block: (c: Class, m: Metric, modelFolder: String) -> Pair<String, Classifier>): List<EvaluationResult> {
    val (train, test) = loadDataset(c, useOversampling)

    val modelFolder = "$MODEL_FOLDER/${c.suffix}"
    File(modelFolder).mkdirs()

    val results = ArrayList<EvaluationResult>()
    for (m in metrics) {
        val (name, classifier) = block(c, m, modelFolder)
        results += buildAndEvaluate(classifier, train, test, name, m)
        saveModel(classifier, "$modelFolder/$name-$m.model")
    }

    results.forEach { it.useOversampling = useOversampling }
    return results
}

private fun loadDataset(c: Class, useOversampling: Boolean = false): Pair<Instances, Instances> {
    var train = load("$DATASET_FOLDER/fullTrain${c.suffix}.arff")
    if (useOversampling) {
        train = train.oversampling()
    }
    val test = load("$DATASET_FOLDER/fullTest${c.suffix}.arff")
    return Pair(train, test)
}

private fun Instances.oversampling(): Instances {
    val distribution = classDistribution()

    var outInstances = this
    val mostCommon = distribution.max()!!
    for ((i, num) in distribution.withIndex()) {
        if (num != mostCommon) {
            outInstances = oversampling(outInstances, i, (mostCommon.toDouble() / num - 1) * 100)
        }
    }

    outInstances.classDistribution()
    return outInstances
}

private fun Instances.classDistribution(): IntArray {
    val distribution = IntArray(numClasses())
    for (inst in this) {
        if (!inst.classIsMissing()) {
            distribution[inst.classValue().toInt()]++
        }
    }
    println(relationName())
    for ((i, num) in distribution.withIndex()) {
        println("${classAttribute().value(i)}: $num")
    }
    return distribution
}

private fun oversampling(instances: Instances, classIndex: Int, percentage: Double): Instances {
    val smote = SMOTE()
    smote.classValue = (classIndex + 1).toString()
    smote.percentage = percentage
    smote.setInputFormat(instances)
    return Filter.useFilter(instances, smote)
}

private fun randomForest(size: Int, threads: Int = 1): RandomForest {
    val classifier = RandomForest()
    classifier.numIterations = size
    classifier.numExecutionSlots = threads
    return classifier
}

private fun buildAndEvaluate(classifier: Classifier, train: Instances, test: Instances, name: String,
                             metric: Metric): List<EvaluationResult> {
    classifier.buildClassifier(train)
    val evaluation = Evaluation(test)
    evaluation.evaluateModel(classifier, test)
    val out = StringBuilder("$name\n")
    out.append(metric.name).append("\n")
    val results = evaluation.evaluationResults(name, metric, out)
    println(out.toString())
    return results
}

private fun Evaluation.evaluationResults(name: String, metric: Metric, out: StringBuilder): List<EvaluationResult> {
    val results = ArrayList<EvaluationResult>(2)
    results += EvaluationResult.Accuracy(name, pctCorrect())
    out.append("  accuracy: ${pctCorrect()}\n")
    for (i in 0 until header.numClasses()) {
        val fn = metric.fn
        out.append("  metric for '${header.classAttribute().value(i)}': ${fn(i)}\n")
    }
    results += when (metric) {
        Metric.F_MEASURE -> EvaluationResult.MacroFMeasure(name, macro(metric.fn))
        Metric.RECALL -> EvaluationResult.MacroRecall(name, macro(metric.fn))
    }
    out.append("  macro metric: ${macro(metric.fn)}\n")
    return results
}

private fun Evaluation.macro(metric: Evaluation.(Int) -> Double): Double {
    return (0 until header.numClasses())
            .map { metric(it) }
            .average()
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

enum class Metric(val fn: Evaluation.(Int) -> Double) {
    RECALL(Evaluation::recall),
    F_MEASURE(Evaluation::fMeasure)
}

sealed class EvaluationResult(val name: String, val value: Double) {

    var useOversampling: Boolean = false

    class Accuracy(name: String, value: Double): EvaluationResult(name, value)
    class MacroRecall(name: String, value: Double): EvaluationResult(name, value)
    class MacroFMeasure(name: String, value: Double): EvaluationResult(name, value)

    private fun metricName(): String = when (this) {
        is EvaluationResult.Accuracy -> "accuracy"
        is EvaluationResult.MacroRecall -> "macro recall"
        is EvaluationResult.MacroFMeasure -> "macro f-measure"
    }

    override fun toString(): String {
        return "${metricName()}: $value ($name${if (useOversampling) {"(oversampling)"} else {""}})"
    }
}
