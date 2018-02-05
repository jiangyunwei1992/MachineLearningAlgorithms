package com.ms.machinelearning

import com.ms.machinelearning.common.Utils
import com.ms.machinelearning.common.TrainSample
import scala.collection.mutable.ListBuffer
import scala.util.Random

/*
 * 1. 逻辑斯蒂分布：设X是连续随机变量，X服从逻辑斯蒂分布，则：
 * 		分布函数F(x) = P(X<=x) = 1/（1+exp(-(x-μ)/γ))
 * 		密度函数f(x) = F'(x) = exp(-(x-μ)/γ)/(γ(1+exp(-(x-μ)/γ)^2))
 * 2. 二项逻辑斯蒂回归：
 * 		P(Y=1|x) = exp(w·x+b)/(1+exp(w·x+b))
 * 		P(Y=0|x) = 1/(1+exp(w·x+b))
 * 		几率：该事件发生的概率与不发生的概率的比值，假设Y=1是该事件发生的概率，
 * 		则log(P(Y=1|x)/P(Y=0|x)) = w·x
 * 3. 参数估计：应用极大似然估计
 * 		L(w) = Σ[yi(w·xi) - log(1+exp(w·xi))]
 * 4. 多项逻辑斯谛回归
 * 		P(Y=K|x) = 1/(1+Σexp(wk·x))
 * 5. 主要思想：根据现有数据对分类边界线建立回归公式，以此分类。
 * 6. Sigmoid函数：f(x) = 1/(1+exp(w·x+b))
 * 7. 梯度上升：要找到某函数的最大值，最好的方法是沿着该函数的梯度方向探寻
 * 		算法：
 * 		1) 每个回归系数初始化为1
 * 		2) 重复R次：
 * 					计算整个数据集的梯度g
 * 					使用alpha×g更新回归系数的向量
 * 					返回回归系数
 * 	8. 随机梯度上升：梯度上升算法在每次更新回归系数时，都需要遍历整个数据集，该
 * 		 方法在处理大量数据时，计算复杂度太高。一种改进方法是每次仅用一个样本点来
 * 		 更新回归系数，即随机梯度上升。随机梯度上升是一种在线学习算法
 * 		 算法：
 * 		 1) 所有回归系数初始化为1
 * 		 2) 对训练集中每个样本：
 * 					计算该样本的梯度g
 * 					使用alpha×g更新回归系数
 * 	9. 改进的随机梯度上升：在数据集非线性可分的情况下，每次迭代都会引起系数的剧烈改变，
 * 		 为了避免来回波动以及更快的收敛，可用改进的梯度上升
 */
object LogisticRegression {
  def sigmoid(vector: Seq[Double]): Seq[Double] = {
    vector.map(v => 1.0 / (1 + Math.exp(-v)))
  }
  def sigmoid(x: Double): Double = 1.0 / (1 + Math.exp(-x))

  def initWeight(numFeature: Int): Seq[Double] = {
    val ones = new ListBuffer[Double]
    for (index <- 0 until numFeature) {
      ones.+=(1.0)
    }
    ones
  }

  def gradientAscent(trainSamples: Seq[TrainSample], alpha: Double, maxCycles: Int): Seq[Double] = {
    var weights = initWeight(trainSamples(0).features.length)
    val matrix = trainSamples.map(ts => ts.features)
    val labels = trainSamples.map(ts => ts.label)
    for (cycle <- 0 until maxCycles) {
      val h = sigmoid(Utils.matrixStarVector(matrix, weights))
      val error = Utils.vectorMinusVector(labels, h)
      weights = Utils.vectorAddVector(weights, Utils.constantMulVector(alpha, Utils.matrixStarVector(Utils.matrixTranspose(matrix), error)))
    }
    weights
  }

  def stocGradAscent(trainSamples: Seq[TrainSample], alpha: Double): Seq[Double] = {
    var weights = initWeight(trainSamples(0).features.length)
    val labels = trainSamples.map(ts => ts.label)
    for (index <- 0 until trainSamples.length) {
      val h = sigmoid(Utils.vectorDotVector(trainSamples(index).features, weights))
      val error = labels(index) - h
      weights = Utils.vectorAddVector(weights, Utils.constantMulVector(alpha * error, trainSamples(index).features))
    }
    weights
  }

  def improvedStocGradAscent(trainSamples: Seq[TrainSample], maxCycles: Int): Seq[Double] = {
    var weights = initWeight(trainSamples(0).features.length)
    val labels = trainSamples.map(ts => ts.label)
    val random = new Random()
    for (cycle <- 0 until maxCycles) {
      val sampleCopy = ListBuffer[TrainSample]()
      sampleCopy.++=(trainSamples)
      for (index <- 0 until trainSamples.length) {
        val alpha = 4 / (1.0 + cycle + index) + 0.0001
        //val alpha = 0.0001
        val randIndex = random.nextInt(sampleCopy.length)
        val h = sigmoid(Utils.vectorDotVector(sampleCopy(randIndex).features, weights))
        val error = labels(randIndex) - h
        println(error)
        weights = Utils.vectorAddVector(weights, Utils.constantMulVector(alpha * error, sampleCopy(randIndex).features))
        sampleCopy.remove(randIndex)
      }
    }
    weights
  }

  def predict(weights: Seq[Double], features: Seq[Double]): Int = {
    sigmoid(Utils.vectorDotVector(weights, features)) > 0.5 match {
      case true => 1
      case false => 0
    }
  }

  def main(args: Array[String]) {
    val fileName = System.getProperty("user.dir") + "/src/com/ms/machinelearning/trainsample_logisticregression.txt"
    val trainSamples = Utils.readDataWithAppend(fileName)
//    val weights = gradientAscent(trainSamples, 0.001, 500)
//    println(weights)
//
//    val weights2 = stocGradAscent(trainSamples, 0.01)
//    println(weights2)

    val weights3 = improvedStocGradAscent(trainSamples, 1)
    println(weights3)
  }
}