package com.warrior.multidata.classification.model

import com.warrior.multidata.classification.model.AttributeMapper.*
import weka.core.Attribute
import weka.core.DenseInstance
import weka.core.Instances
import weka.core.Utils
import weka.filters.Filter
import weka.filters.unsupervised.attribute.Remove
import java.util.*

/**
 * Created by warrior on 04/07/16.
 */

fun main(args: Array<String>) {
    generateFullDatasets()
}

private fun generateFullDatasets() {
    val testIds = load("featuresSingapore/TestSet.csv").map { it.stringValue(0) }.toSet()
    val groundTruth = load("featuresSingapore/SingaporeGroundTruth.csv")

    val idAttr = groundTruth.attribute("row ID")
    val attrMappers = listOf(
            AgeGroup(groundTruth),
            Gender(groundTruth),
            Relationship(groundTruth),
            EducationLevelBinary(groundTruth),
            EducationLevelTernary(groundTruth),
            Occupation(groundTruth)
    )


    val infoList = loadDatasets()

    val attrs = ArrayList<Attribute>()
    for (info in infoList) {
        println("${info.name} attrs: ${attrs.size}-${attrs.size + info.instances.numAttributes() - 2}")
        for (i in 0 until info.instances.numAttributes()) {
            if (i != info.idIndex) {
                attrs += info.instances.attribute(i).clone()
            }
        }
    }

    for (mapper in attrMappers) {
        attrs += mapper.getNewAttr()
    }

    val trainInstances = Instances("fullTrain", ArrayList(attrs), groundTruth.size - testIds.size)
    val testInstances = Instances("fullTest", ArrayList(attrs), testIds.size)

    for (classesInst in groundTruth) {
        val values = ArrayList<Double>(attrs.size)
        val stringId = classesInst.stringValue(idAttr)
        for (info in infoList) {
            val inst = info[stringId]
            if (inst != null) {
                for (i in 0 until inst.numAttributes()) {
                    if (i != info.idIndex) {
                        values += inst.value(i)
                    }
                }
            } else {
                values += Collections.nCopies(info.instances.numAttributes() - 1, Utils.missingValue())
            }
        }
        val hasValues = values.any { it != Utils.missingValue() }
        if (hasValues) {
            for (mapper in attrMappers) {
                val value = classesInst.value(mapper.attr)
                values += if (Utils.isMissingValue(value)) {
                    value
                } else {
                    mapper.map(value)
                }
            }

            val mergedInstance = DenseInstance(1.0, values.toDoubleArray())
            if (stringId !in testIds) {
                trainInstances += mergedInstance
            } else {
                testInstances += mergedInstance
            }
        }
    }

    println("--- train ---")
    saveToDatasetFolder(trainInstances, attrMappers)
    println("--- test ---")
    saveToDatasetFolder(testInstances, attrMappers)
}

private fun loadDatasets(): List<InstancesInfo> = listOf(
        InstancesInfo(DatasetInfo.FOURSQUARE.datasetName, load("featuresSingapore/Foursquare/venueCategoriesFeatures5Months.csv")),
        InstancesInfo(DatasetInfo.FOURSQUARE_LDA.datasetName, load("featuresSingapore/Foursquare/venueCategoriesLDA6Features.csv")),
        InstancesInfo(DatasetInfo.TWITTER_LDA.datasetName, load("featuresSingapore/Twitter/LDA50Features.csv")),
        InstancesInfo(DatasetInfo.TWITTER_LIWC.datasetName, load("featuresSingapore/Twitter/LIWCFeatures.csv")),
        InstancesInfo(DatasetInfo.TWITTER.datasetName, load("featuresSingapore/Twitter/manuallyDefinedTextFeatures.csv")),
        InstancesInfo(DatasetInfo.INSTAGRAM.datasetName, load("featuresSingapore/Instagram/imageConceptsFeatures.csv"))
)

private fun saveToDatasetFolder(instances: Instances, mappers: List<AttributeMapper>) {
    val relationName = instances.relationName()
    save(instances, "$DATASET_FOLDER/$relationName.arff")

    for ((i, mapper) in mappers.withIndex()) {
        val remove = Remove()
        val indices = ((0 until mappers.size) - i).map { v -> v + instances.numAttributes() - mappers.size }
        remove.setAttributeIndicesArray(indices.toIntArray())
        remove.setInputFormat(instances)
        val filtered = Filter.useFilter(instances, remove)
        filtered.deleteWithMissing(filtered.numAttributes() - 1)
        println("${mapper.name}: ${filtered.size}")
        save(filtered, "$DATASET_FOLDER/$relationName${mapper.name}.arff")
    }
}
