package com.warrior.multidata.classification.model

import weka.classifiers.Classifier
import weka.classifiers.meta.AdaBoostM1
import weka.core.Instances

/**
 * Created by warrior on 10/07/16.
 */
class AdaBoosting : AdaBoostM1() {

    private var classifiers: List<Classifier>? = null

    override fun initializeClassifier(data: Instances) {
        super.initializeClassifier(data)
        val localClassifiers = classifiers
        if (localClassifiers != null) {
            m_Classifiers = localClassifiers.toTypedArray()
        }
    }

    fun setClassifiers(classifiers: List<Classifier>) {
        this.classifiers = classifiers
        this.numIterations = classifiers.size
    }
}