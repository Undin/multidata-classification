package com.warrior.multidata.classification.model

import weka.core.Attribute
import weka.core.Instances
import weka.core.Utils

/**
 * Created by warrior on 29/07/16.
 */

enum class Class(val suffix: String, val attributeName: String) {
    AGE_GROUP("AgeGroup", "ageGroup"),
    GENDER("Gender", "gender"),
    RELATIONSHIP("Relationship", "relationship"),
    EDUCATION_LEVEL_BINARY("EducationLevelBinary", "education_level"),
    EDUCATION_LEVEL_TERNARY("EducationLevelTernary", "education_level"),
    OCCUPATION("Occupation", "occupation")
}

sealed class AttributeMapper(clazz: Class, instances: Instances) {

    val attr: Attribute = instances.attribute(clazz.attributeName)
    val name = clazz.suffix

    open fun getNewAttr(): Attribute = attr.clone()
    open fun map(value: Double): Double = value

    class AgeGroup(instances: Instances) : AttributeMapper(Class.AGE_GROUP, instances) {
        override fun getNewAttr(): Attribute = Attribute("ageGroup", listOf("AGE10_20", "AGE20_30", "AGE30_40","AGE40_INF"))
        override fun map(value: Double): Double = if (attr.value(value.toInt()).contains("50")) { 3.0 } else { value }
    }
    class Gender(instances: Instances) : AttributeMapper(Class.GENDER, instances)
    class Relationship(instances: Instances) : AttributeMapper(Class.RELATIONSHIP, instances) {
        override fun getNewAttr(): Attribute = Attribute("relationship", listOf("single", "in a relationship"))
        override fun map(value: Double): Double = if (attr.value(value.toInt()) == "single") { 0.0 } else { 1.0 }
    }
    class EducationLevelBinary(instances: Instances) : AttributeMapper(Class.EDUCATION_LEVEL_BINARY, instances) {
        override fun getNewAttr(): Attribute = Attribute("education_level_binary", listOf("school", "university"))
        override fun map(value: Double): Double = if (attr.value(value.toInt()).contains("school")) { 0.0 } else { 1.0 }
    }
    class EducationLevelTernary(instances: Instances) : AttributeMapper(Class.EDUCATION_LEVEL_TERNARY, instances) {
        override fun getNewAttr(): Attribute = Attribute("education_level_ternary", listOf("school", "undergraduate", "graduate"))
        override fun map(value: Double): Double {
            val name = attr.value(value.toInt())
            if (name.contains("school")) {
                return 0.0
            }
            if (name.contains("student")) {
                return 1.0
            }
            return 2.0
        }
    }

    class Occupation(instances: Instances) : AttributeMapper(Class.OCCUPATION, instances) {

        private val newAttr = Attribute("occupation", listOf(
                "archetecture and engineering",
                "protective service",
                "food preparation and service related",
                "management",
                "arts, design, entertainment, sports, and media",
                "office and administrative support",
                "personal care and service",
                "sales and related",
                "legal",
                "transportation and material moving",
                "production",
                "construction and extraction",
                "education, training, and library",
                "business and financial operations"
        ))

        override fun getNewAttr(): Attribute = newAttr

        override fun map(value: Double): Double {
            val originalValue = attr.value(value.toInt())
            val index = newAttr.indexOfValue(originalValue)
            return if (index == -1) { Utils.missingValue() } else { index.toDouble() }
        }
    }
}
