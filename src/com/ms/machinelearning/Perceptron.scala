package com.ms.machinelearning

import scala.io.Source
import scala.collection.mutable.ListBuffer
import javax.rmi.CORBA.Util
import com.ms.machinelearning.common.Utils
import com.ms.machinelearning.common.WeightBiasModel
import com.ms.machinelearning.common.TrainSample

/*
 * 1. 二类分类的线性分类模型
 * 2. 输入：实例的特征向量
 * 		输出：实例的类别，±1
 * 3. 超平面：n维空间中，(n-1)维度的子空间
 * 4. 判别模型
 * 5. f(x) = sign(w·x+b)
 * 6. 几何解释：线性方程w·x+b = 0，对应特征空间中的一个超平面，其中w是
 * 		超平面的法向量，b是超平面的截距。该超平面将特征空间划分为两个部分。
 * 		位于两部分的点分别被判定为正、负两类，因此S平面称为分离超平面。
 * 7. 原理：对所有正确分类的实例点，
 * 		y=+1 ==> w·x+b>0
 * 		y=-1 ==> w·x+b<0
 * 		所以，损失函数L(Y,f(x)) = -y(w·x+b)
 *
 * 8. 算法
 * 	  1) 初始化w0,b0
 * 		2) 选取(xi,yi)
 * 		3) 如果yi(w·xi+b)<=0:
 * 					w <- w + αyixi
 * 					b <- b + αyi
 * 		4) 调转到2)，直到没有错误分类
 */
object Perceptron {
  def initWeight(numFeatures: Int): Seq[Double] = {
    val weight = ListBuffer[Double]()
    for (index <- 0 until numFeatures) {
      weight.+=(0)
    }
    weight
  }

  def updateWeightsAndBias(model: WeightBiasModel, trainSample: TrainSample, eta: Double): WeightBiasModel = {
    val weights = Utils.vectorAddVector(model.weights, Utils.constantMulVector(eta.*(trainSample.label), trainSample.features))
    val bias = model.bias + eta * trainSample.label
    WeightBiasModel(weights, bias)
  }

  def trainModel(trainSamples: Seq[TrainSample]): WeightBiasModel = {
    val numFeatures = trainSamples(0).features.length
    //val initWeight = Seq(0 until numFeatures).map(x=>0)
    var model = WeightBiasModel(initWeight(numFeatures), 0)
    val eta = 1
    var misClassfication = false
    do {
      misClassfication = false
      trainSamples.foreach(trainSample => {
        if (trainSample.label * (Utils.vectorDotVector(model.weights, trainSample.features) + model.bias) <= 0) {
          model = updateWeightsAndBias(model, trainSample, eta)
          misClassfication = true
        }
      })
    } while (misClassfication)
    model
  }
  def main(args: Array[String]) {
    val fileName = System.getProperty("user.dir") + "/src/com/ms/machinelearning/trainsample_perceptron.txt"
    val trainSamples = Utils.readData(fileName)
    val model = trainModel(trainSamples)
    println(model)
  }
}