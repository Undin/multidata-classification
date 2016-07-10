package com.warrior.multidata.classification.model

import weka.core.Attribute
import weka.core.Instance
import weka.core.Instances

/**
 * Created by warrior on 08/07/16.
 */

const val DATASET_FOLDER = "datasets"
const val MODEL_FOLDER = "models"

const val ID_ATTR = "_id"
const val GENDER_ATTR = "gender"
const val RELATIONSHIP_ATTR = "relationship"

const val SINGLE = "single"
const val IN_A_RELATIONSHIP = "in a relationship"

enum class DatasetInfo(
        val datasetName: String,
        val indices: String,
        val forestSizeGender: Int,
        val forestSizeRelationship: Int
) {
    FOURSQUARE("foursquare", "1-765,last", 120, 10),
    FOURSQUARE_LDA("foursquareLDA", "766-771,last", 60, 140),
    TWITTER_LDA("twitterLDA", "772-821,last", 140, 90),
    TWITTER_LIWC("twitterLIWC", "822-891,last", 120, 90),
    TWITTER("twitter", "892-905,last", 130, 130),
    INSTAGRAM("instagram", "906-1906,last", 110, 20);
}

enum class Class(val suffix: String) {
    GENDER("Gender"),
    RELATIONSHIP("Relationship");
}

data class InstancesInfo(val name: String, val instances: Instances) {

    val id: Attribute = instances.attribute(ID_ATTR)
    val idIndex: Int = id.index()

    private val map: Map<String, Instance> = instances.map { (it.stringValue(id) to it) }.toMap()

    operator fun get(id: String): Instance? = map[id]
}
