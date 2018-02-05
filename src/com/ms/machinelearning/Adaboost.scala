package com.ms.machinelearning

import scala.collection.mutable.ListBuffer

import com.ms.machinelearning.common.TrainSample
import com.ms.machinelearning.common.Utils

/*
 * 1. Boost在分类问题中，通过改变训练样本的权重，多个学习多个分类器，并将
 * 		这些分类器进行线性组合，提高分类的性能
 * 2. 提升方法的思想：对于一个复杂任务来说，将多个专家的判断进行适当的综合
 * 		所得出来的判断，要比其中任何一个专家单独的判断好。
 * 		弱可学习：一个概念，如果存在一个多项式的学习算法能够学习它，学习的
 * 		正确率仅比随机猜测略好，那就是弱科学系的。弱可学习是强可学习的充分
 * 		必要条件
 * 2. Adboost：给定一个训练样本集，求比较粗糙的分类规则要比求精确的分类
 * 		规则容易得多。提升方法就是从弱学习算法出发，反复学习，得到一系列
 * 		弱分类器，然后组合这些分类器，构成一个强分类器。
 * 		两个问题：
 * 		1) 每一轮如何改变训练数据的权值或概率分布：提高那些被前一轮弱分类器
 * 			 错误分类的数据，由于其权值的加大而受到后一轮的弱分类器的更大关注，
 * 			 于是分类问题被一系列弱分类器分而治之
 * 		2) 如何将弱分类器组合成一个强分类器：采取加权多数表决的方法。具体地，
 * 			 加大分类误差率小的弱分类器的权值，使其在表决中起较大的作用，减小
 * 			 分类误差率大的弱分类器的权值，使其在表决中起较小的作用
 * 		算法：
 * 		1) 初始化训练数据的权值分布
 * 			 D1=(W(1)1,W(1)2,...,W(1)i,...,W(1)N), W(1)i=1/N
 * 		2) 对m = 1,2,...,M:
 * 					a) 使用权值分布Dm的训练数据集学习，得到基本分类器Gm(x)
 * 					b) 计算Gm(x)在训练数据集上的分类误差率em
 * 					c) 计算Gm(x)的系数：αm = 0.5ln((1-em)/em)
 * 					d) 更新训练集的权值分布：
 * 						 Dm+1 = (W(m+1)1,...,W(m+1)N)
 * 						 W(m+1)i = W(m)i×exp(-αmyiGm(xi))/Zm
 * 						 Zm = ΣW(m)i×exp(-αmyiGm(xi))
 * 	 3) 构建基本分类器的线性组合f(x)=ΣαmGm(x)
 * 			得到最终分类器G(x) = sign(f(x))
 */
object Adaboost {
  case class BestStump(dim: Int, thresh: Double, ineq: String) {
    var alpha: Double = 0.0
    def setAlpha(a: Double) = (alpha = a)
    override def toString: String = "{'dim':" + dim + ", 'ineq':" + ineq + ", 'thresh':" + thresh + "alpha = "+alpha+"}"
  }
  case class StumpModel(bestStump: BestStump, minError: Double, bestClasEst: Seq[Double]) {
    def getBestStump: BestStump = bestStump
    def getMinError: Double = minError
    def getBestClasEst: Seq[Double] = bestClasEst
    override def toString: String = "(" + bestStump.toString() + ", minError:" + minError + ", bestClasEst:" + bestClasEst
  }
  case class AdaboostModel(weakClassArr:Seq[BestStump],aggClassEst:Seq[Double])
  //构建单层决策树
  def stumpClassify(trainSamples: Seq[TrainSample], dimen: Int, threshVal: Double, operator: String): Seq[Double] = {
    val column: Seq[Double] = trainSamples.map(ts => ts.features(dimen))
    val retArray = operator match {
      case "lt" => column.map(c => c <= threshVal match {
        case true => -1.0
        case false => 1.0
      })
      case _ => column.map(c => c > threshVal match {
        case true => -1.0
        case false => 1.0
      })
    }
    retArray
  }

  def buildStump(trainSamples: Seq[TrainSample], d: Seq[Double]): StumpModel = {
    val features = trainSamples.map(fs => fs.features)
    val labels = trainSamples.map(ts => ts.label)
    val m = trainSamples.length
    val n = features(0).length
    val numSteps = 10.0
    val operators = Seq("lt", "gt")
    var bestClasEst = Utils.initZeros(m)
    var minError = Double.MaxValue
    var bestStump = BestStump(-1, 0, "")
    for (i <- 0 until n) {
      val rangeMin = features.map(f => f(i)).min
      val rangeMax = features.map(f => f(i)).max
      val stepSize = (rangeMax - rangeMin) / numSteps
      for (j <- -1 to numSteps.toInt) {
        for (inequal <- operators) {
          val threshVal = rangeMin + j * stepSize
          val predictedVals = stumpClassify(trainSamples, i, threshVal, inequal)
          val errArr = (0 until m).toList.map(e => predictedVals(e).==(labels(e)) match {
            case true => 0.0
            case false => 1.0
          })
          val weightedError = Utils.vectorDotVector(d, errArr)
          // println("split: dim"+i+", thresh "+threshVal+", thresh inequal:"+inequal+", the weighted error is "+weightedError)
          if (weightedError.<(minError)) {
            minError = weightedError
            bestClasEst = predictedVals.map(e => e)
            bestStump = BestStump(i, threshVal, inequal)
          }
        }
      }
    }
    StumpModel(bestStump, minError, bestClasEst)
  }

  //
  def adaBoostTrainDS(trainSamples: Seq[TrainSample], numIter: Int = 40):AdaboostModel = {
    val weakClassArr = new ListBuffer[BestStump]
    val m = trainSamples.length
    var d = Utils.initOnes(m).map(e => e / m)
    var aggClassEst = Utils.initZeros(m)
    val labels = trainSamples.map(ts => ts.label)
    var shouldBreak = false
    for (i <- 0 until numIter) {
      if(!shouldBreak){
        val model = buildStump(trainSamples, d)
        val alpha = 0.5 * Math.log((1 - model.minError) / Math.max(model.minError, Math.pow(10, -16)))
        model.bestStump.setAlpha(alpha)//告诉总分类器本次单层决策树输出结果的权重
        weakClassArr.+=(model.bestStump)
        //println("classEst: "+model.getBestClasEst.toString())
        val expon = Utils.vectorMulVector(Utils.constantMulVector(-alpha, labels), model.getBestClasEst)
        d = Utils.vectorMulVector(d, Utils.vectorExp(expon))
        d = Utils.constantMulVector(1.0/d.sum, d)
        aggClassEst = Utils.vectorAddVector(aggClassEst, Utils.constantMulVector(alpha, model.getBestClasEst))
        val aggError = Utils.vectorMulVector(Utils.vectorInequalVector(Utils.vectorSign(aggClassEst), labels), Utils.initOnes(m))
        val errorRate = aggError.sum/m
        if(errorRate == 0.0) shouldBreak = true;
      }
    }
    AdaboostModel(weakClassArr,aggClassEst)
  }

  def adaboostClassification(features:Seq[Seq[Double]],adaboostModel:AdaboostModel):Seq[Double] = {
    val m = features.length
    var aggClassEst = Utils.initZeros(m)
    val weakClassifiers = adaboostModel.weakClassArr
    val trainSamples = features.map(f => TrainSample(f,0.0))
    for(i<- 0 until weakClassifiers.length){
      val classEst = stumpClassify(trainSamples,weakClassifiers(i).dim,weakClassifiers(i).thresh,weakClassifiers(i).ineq)
      aggClassEst = Utils.vectorAddVector(aggClassEst, Utils.constantMulVector(weakClassifiers(i).alpha, classEst))
    }
    Utils.vectorSign(aggClassEst)
  }
  
  def main(args: Array[String]) {
    val fileName = System.getProperty("user.dir") + "/src/com/ms/machinelearning/trainsample_adaboost.txt"
    val trainSamples = Utils.readData(fileName)
    val adaboostModel:AdaboostModel = adaBoostTrainDS(trainSamples,30)
    println(adaboostModel.aggClassEst)
    println(adaboostModel.weakClassArr)
    val result = adaboostClassification(Seq(Seq(0.0,0.0),Seq(5.0,5.0)),adaboostModel)
    println(result)
  }
}