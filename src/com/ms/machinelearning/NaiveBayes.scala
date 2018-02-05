package com.ms.machinelearning

import com.ms.machinelearning.common.Utils
import com.ms.machinelearning.common.TrainSample
import scala.collection.mutable.HashMap

/*
 * 1. 基于贝叶斯定理与特征条件独立假设的分类方法
 * 2. 对于给定训练集，首先基于特征条件独立假设
 * 		学习输入/输出的联合分布概率，然后基于此模型
 * 		对给定的输入x，利用贝叶斯定理求出后验概率最
 * 		大的输出y。实现简单，学习与预测效率都很高
 * 		1) 特征条件独立性假设：
 * 			P(X=x|Y=ck) = ∏P(X(j)=x(j)|Y=ck)
 * 		2) 贝叶斯定理：
 * 			P(Y=ck|X=x)=P(X=x|Y=ck)P(Y=ck)/ΣP(X=x|Y=ck)P(Y=ck)
 * 3. 参数估计：
 * 		1) 极大似然估计：P(Y=ck) = ΣI（yi=ck）/N
 * 4. 算法：
 * 		1) 计算先验概率：
 * 			 P(Y=ck)=ΣI(yi=ck)/N
 * 			 P(X(j)=ajl|Y=ck)=ΣI(xi(j)=ajl,yi=ck)/ΣI(yi=ck)
 * 		2) 给定实例x=(x(1),x(2),...,x(n))^T
 * 			 P(Y=ck)∏P(X(j)=x(j)|Y=ck)
 * 		3) 确定实例的类
 * 5. 拉普拉斯平滑：
 * 		使用极大似然估计可能会出现需要的估计概率为0的情况，此时会影响到
 * 		后验概率的计算结果，使分类产生偏差。为解决这一问题，需要加入拉普
 * 		拉斯平滑
 */
object NaiveBayes {
  def trainModel(trainSamples: Seq[TrainSample]): Tuple2[HashMap[String, Double], Map[Double, Double]] = {
    val N = trainSamples.length
    val groupSamplesByLabel = trainSamples.groupBy(ts => ts.label)
    val possibilityGroupedByLabel = groupSamplesByLabel.map(g => (g._1, g._2.length * 1.0 / N))
    val map = HashMap[String, Double]()
    groupSamplesByLabel.foreach(g => {
      val label = g._1
      val samples = g._2
      //特征的每一维度进行遍历
      for (index <- 0 until samples(0).features.length) {
        val distinctValues = samples.map(s => s.features(index)).distinct
        for (valueType <- 0 until distinctValues.length) {
          val key = s"X" + (index + 1) + " = " + (valueType + 1) + s"|Y = $label"
          val poss = samples.filter(s => s.features(index) == (valueType + 1)).length * 1.0 / samples.length
          map.+=((key, poss))
        }
      }
    })
    (map, possibilityGroupedByLabel)
  }

  def trainModelWithLaplaceSmooth(trainSamples: Seq[TrainSample]): Tuple2[HashMap[String, Double], Map[Double, Double]] = {
    val N = trainSamples.length
    val groupSamplesByLabel = trainSamples.groupBy(ts => ts.label)
    val distinctLabels = groupSamplesByLabel.size
    val possibilityGroupedByLabel = groupSamplesByLabel.map(g => (g._1, (g._2.length + 1) * 1.0 / (N + distinctLabels)))
    val map = HashMap[String, Double]()
    groupSamplesByLabel.foreach(g => {
      val label = g._1
      val samples = g._2
      //特征的每一维度进行遍历
      for (index <- 0 until samples(0).features.length) {
        val distinctValues = samples.map(s => s.features(index)).distinct
        for (valueType <- 0 until distinctValues.length) {
          val key = s"X" + (index + 1) + " = " + (valueType + 1) + s"|Y = $label"
          val poss = (samples.filter(s => s.features(index) == (valueType + 1)).length + 1) * 1.0 / (samples.length + distinctValues.length)
          map.+=((key, poss))
        }
      }
    })
    (map, possibilityGroupedByLabel)
  }

  def predict(model: Tuple2[HashMap[String, Double], Map[Double, Double]], features: Seq[Double]): Double = {
    val featurePossibilities = model._1
    val labelPossibilities = model._2
    var maxPossibility = 0.0
    var selectedLabel = Double.MaxValue
    labelPossibilities.foreach(f => {
      val label = f._1
      val labelPossibility = f._2
      var prediction = labelPossibility
      for (featureIndex <- 0 until features.length) {
        val key = "X" + (featureIndex + 1) + " = " + features(featureIndex).toInt + "|Y = " + label
        val featurePossibility = featurePossibilities.get(key)
        prediction = prediction.*(featurePossibility.get)
      }
      if (prediction > maxPossibility) {
        maxPossibility = prediction
        selectedLabel = label
      }
    })
    selectedLabel
  }

  def main(args: Array[String]) {
    val fileName = System.getProperty("user.dir") + "/src/com/ms/machinelearning/trainsample_naivebayes.txt"
    val trainSamples = Utils.readData(fileName)
    val model = trainModelWithLaplaceSmooth(trainSamples)
    val input = Seq[Double](2, 1)
    val prediction = predict(model, input)
    println(prediction)
  }
}