package com.warrior.multidata.classification.model

import weka.core.Attribute
import weka.core.DenseInstance
import weka.core.Instances
import weka.core.Utils
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
    val genderAttr = groundTruth.attribute(GENDER_ATTR)
    val relationshipAttr = groundTruth.attribute(RELATIONSHIP_ATTR)
    val binaryRelationshipAttr = Attribute(RELATIONSHIP_ATTR, listOf(SINGLE, IN_A_RELATIONSHIP))

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

    attrs += genderAttr.clone()
    attrs += binaryRelationshipAttr

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
            values += classesInst.value(genderAttr)
            values += if (classesInst.isMissing(relationshipAttr)) {
                Utils.missingValue()
            } else if (classesInst.stringValue(relationshipAttr) == SINGLE) {
                binaryRelationshipAttr.indexOfValue(SINGLE).toDouble()
            } else {
                binaryRelationshipAttr.indexOfValue(IN_A_RELATIONSHIP).toDouble()
            }

            val mergedInstance = DenseInstance(1.0, values.toDoubleArray())
            if (stringId !in testIds) {
                trainInstances += mergedInstance
            } else {
                testInstances += mergedInstance
            }
        }
    }

    saveToDatasetFolder(trainInstances)
    saveToDatasetFolder(testInstances)
}

private fun loadDatasets(): List<InstancesInfo> = listOf(
        InstancesInfo(DatasetInfo.FOURSQUARE.datasetName, load("featuresSingapore/Foursquare/venueCategoriesFeatures5Months.csv")),
        InstancesInfo(DatasetInfo.FOURSQUARE_LDA.datasetName, load("featuresSingapore/Foursquare/venueCategoriesLDA6Features.csv")),
        InstancesInfo(DatasetInfo.TWITTER_LDA.datasetName, load("featuresSingapore/Twitter/LDA50Features.csv")),
        InstancesInfo(DatasetInfo.TWITTER_LIWC.datasetName, load("featuresSingapore/Twitter/LIWCFeatures.csv")),
        InstancesInfo(DatasetInfo.TWITTER.datasetName, load("featuresSingapore/Twitter/manuallyDefinedTextFeatures.csv")),
        InstancesInfo(DatasetInfo.INSTAGRAM.datasetName, load("featuresSingapore/Instagram/imageConceptsFeatures.csv"))
)

private fun saveToDatasetFolder(instances: Instances) {
    val relationName = instances.relationName()
    save(instances, "$DATASET_FOLDER/$relationName.arff")

    val relationshipAttr = instances.attribute(RELATIONSHIP_ATTR)
    val genderAttr = instances.attribute(GENDER_ATTR)

    val genderInstances = Instances(instances)
    genderInstances.deleteWithMissing(genderAttr)
    genderInstances.deleteAttributeAt(relationshipAttr.index())
    save(genderInstances, "$DATASET_FOLDER/${relationName}Gender.arff")

    val relationshipInstances = Instances(instances)
    relationshipInstances.deleteWithMissing(relationshipAttr)
    relationshipInstances.deleteAttributeAt(genderAttr.index())
    save(relationshipInstances, "$DATASET_FOLDER/${relationName}Relationship.arff")
}
