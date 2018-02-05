package com.ms.machinelearning

import com.ms.machinelearning.common.TrainSample
import com.ms.machinelearning.common.Utils
import scala.collection.mutable.ListBuffer

/*
 * 1. 回归：求回归系数的过程就是回归
 * 2. 线性回归：将输入项分别乘以一些常量，再将结果加起来得到输出
 * 		求解过程：Y=X·w ==>如何求解w ==>找到使误差最小的w
 * 		==> min Σ(yi-xi·w)^2
 * 		==> w = (X^T·X)^-1X^Ty
 * 		==> X^TX可能不是可逆矩阵
 * 3. 局部加权线性回归：线性回归可能出现欠拟合的现象，因为求的是具有最小均方差
 * 		的无偏估计。使用局部加权线性回归(LWLR)，可以克服欠拟合的缺点。即给待预测
 * 		点附近的每个点赋予一定的权重，这种算法每次预测均需要实现选出对应的数据子集：
 * 		w = (X^TWX)^-1Wy
 * 		LWLR使用核来对附近的点赋予更高的权重，常用的就是高斯核：
 * 		W(i,i) = exp(|x(i)-x|/(-2k^2))
 * 		如此就构建了一个只含对角元素的权重矩阵W，点x与x(i)越近，权重越大
 * 		局部加权线性回归存在一个问题，即增加了计算量，因为对每个点做预测时都必须使用
 * 		整个数据集
 * 4. 岭回归：如果数据的特征比样本点还多，此时不可使用线性回归或者局部加权线性回归，
 * 		因为此时不可以计算XTX的逆。
 * 		岭回归是在XTX上加一个λI从而使矩阵非奇异，进而使XTX+λI可逆。I是一个对角矩阵，
 * 		此时：W=(XTX+λI)^-1XTy
 * 		岭回归最先用于处理特征数多余样本数的情况，现在也用于在估计中加入偏差，从而
 * 		得到更好的估计。这里通过引入λ来限制所有w之和，通过引入该惩罚项，能够减少不
 * 		重要的参数，这个技术叫做缩减。
 * 		缩减方法可以去掉不重要的参数，因此能更好地理解数据。
 */
object regression {
  def linearRegression(trainSamples:Seq[TrainSample]):Seq[Double] = {
    val x = trainSamples.map(ts => ts.features)
    val y = trainSamples.map(ts => ts.label)
    val xTx = Utils.matrixStarMatrix(Utils.matrixTranspose(x),x)
    Utils.matrixDet(xTx) == 0.0 match{
      case true => null
      case false => 
        Utils.matrixStarVector(Utils.matrixInverse(xTx), Utils.matrixStarVector(Utils.matrixTranspose(x), y))
    }
  }
  
  def locallyWeightedLinearRegression(target:Seq[Double],trainSamples:Seq[TrainSample], k:Double = 1.0):Double = {
    val x = trainSamples.map(ts => ts.features)
    val y = trainSamples.map(ts => ts.label)
    val m = trainSamples.length
    val weights = (0 until m).toList.map(i =>(0 until m).toList.map(j=>{
      val diff = Utils.vectorMinusVector(target, x(i))
      i==j match{
        case true => Math.exp(Utils.vectorDotVector(diff, diff)/(-2*Math.pow(k, 2)))
        case false => 0
      }
    }))
    val xTx = Utils.matrixStarMatrix(Utils.matrixTranspose(x), Utils.matrixStarMatrix(weights, x))
    val ws = Utils.matrixDet(xTx) == 0.0 match{
      case true => println("determinant of X^T·X==0");null
      case false =>Utils.matrixStarVector(Utils.matrixInverse(xTx),Utils.matrixStarVector(Utils.matrixTranspose(x), Utils.matrixStarVector(weights, y)))
    }
    val prediction = ws.==(null) match {
      case true => 0 
      case false =>Utils.vectorDotVector(target, ws)
    }
    prediction
  }
  
  def ridgeRegression(trainSamples:Seq[TrainSample],lambda:Double = 0.2):Seq[Double] = {
    val x = trainSamples.map(ts => ts.features)
    val y = trainSamples.map(ts => ts.label)
    val xTx = Utils.matrixStarMatrix(Utils.matrixTranspose(x), x)
    val m = trainSamples.length
    val denom = Utils.matrixAddMatrix(xTx, Utils.constantMulMatrix(lambda, Utils.initEyes(m)))
    val ws = Utils.matrixDet(denom)==0.0 match{
      case true => null
      case false => Utils.matrixStarVector( Utils.matrixInverse(denom), Utils.matrixStarVector(Utils.matrixTranspose(x), y))
    }
    ws
  }
  def main(args: Array[String]) {
    val fileName = System.getProperty("user.dir") + "/src/com/ms/machinelearning/trainsample_regression.txt"
    val trainSamples = Utils.readData(fileName)
//    val efficiencies = linearRegression(trainSamples)
//    println(efficiencies)
    val x0 = trainSamples(0).features
    val prediction = locallyWeightedLinearRegression(x0,trainSamples,0.001)
    println(prediction)
  }
}