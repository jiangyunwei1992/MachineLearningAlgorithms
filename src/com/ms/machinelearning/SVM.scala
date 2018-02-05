package com.ms.machinelearning

import com.ms.machinelearning.common.Utils
import scala.util.Random
import com.ms.machinelearning.common.TrainSample
import scala.collection.mutable.ListBuffer
import com.ms.machinelearning.common.WeightBiasModel

/*
 * 1. SVM是一种二类分类模型，基本模型定义在特征空间上的间隔最大的线性分类器。
 * 		间隔最大使其有别于感知机。支持向量机还包括核技巧，使其成为实质上的非线
 * 		性分类器。
 * 2. 学习策略：间隔最大化，可以形式化为一个求解凸二次规划的问题，也可以等价
 * 		于正则化的合页损失函数最小化问题
 * 3. 由简到繁分类：
 * 		1) 线性可分支持向量机：又称硬间隔支持向量机
 * 			超平面：w·x+b = 0
 * 			决策函数：f(x) = sign(w·x+b)
 * 		2) 线性支持向量机：软间隔最大化，又称软间隔支持向量机
 * 		3) 非线性支持向量机：核技巧及软间隔最大化
 * 4. 核函数：将输入从输入控件映射到特征空间得到的特征向量之间的内积。可函数
 * 		等价于隐式地在高维特征空间中学习线性支持向量机
 * 5. 目标：在特征空间中找到一个分离超平面，能将实例分到不同的类，超平面对应
 * 		方程w·x+b=0，其中w是该超平面的法向量，b表示该超平面的截距。
 * 		1) 当训练集线性可分时，根据间隔最大化原则，可以得到唯一最优解
 * 6. 函数间隔：给定训练集T和超平面(w,b)，函数间隔γi = yi(w·xi+b)
 * 		函数间隔可以表示分类预测的正确性以及确信度。但是只有函数间隔还不够，因为
 * 		只要成比例地改变w和b，即可在保持超平面不变的情况下改变函数间隔。
 * 7. 几何间隔：对于给定的训练集T和超平面(w,b)，几何间隔
 * 		γi = yi(w·xi+b)/‖w‖
 * 8. 间隔最大化：支持向量机的基本想法是求解能够正确划分训练数据集并且几何间隔
 * 		最大的分离超平面。对线性可分支持向量机来说，几何间隔最大的分离超平面有且
 * 		仅有一个，此时称为硬间隔最大化。
 * 		间隔最大化的直观解释：对训练集找到几何间隔最大的超平面意味着以充分大的确
 * 		信度对训练数据进行风雷，即不仅将正负实例点分开，而且对最难分的实例点也有
 * 		足够大的确信度将它们分开
 * 9. max(1/‖w‖) s.t. yi(w·xi+b)>=1
 * 		等价于
 * 		min(0.5‖w‖^2), s.t. yi(w·xi+b)>=1
 * 		解决上述问题，除了使用常规方法以外，还可以通过求解对偶问题得到最优解。
 * 		优点在于：对偶问题更容易求解；更自然地引入核函数，推广到非线性分类问题。
 * 		根据拉格朗日对偶性，即将每一个约束条件加上一个拉格朗日因子α，将该约束
 * 		条件融合到目标函数中：
 * 		L(w,b,α) = 0.5‖w‖^2-Σαi(yi(w·xi+b)-1)(αi>=0)
 *			所以原问题转化为θ(w) = max L(w,b,α)
 * 		容易验证，在某个约束条件不满足时，显然有θ(w)=∞(只要αi=∞即可)。当所有
 * 		约束条件都满足时，择优θ(w) = 0.5‖w‖^2。因此，在要求约束条件得到满足的
 * 		情况下，最小化0.5‖w‖^2，则目标函数变成了
 * 		min θ(w) = min_w,b[max_αi≥0 L(w,b,α)]
 * 9. KKT条件：一个非线性规划问题能够有最优化解法的充分必要条件
 * 		min f(x)
 * 		s.t. hj(x) = 0
 * 				 gk(x) ≤ 0
 * 		===> 根据KKT条件，上面最优化数学模型的标准形式中的最小点x*必须满足以下
 *			条件
 * 		1) hj(X*) = 0
 * 		2) ▽f(X*) + Σλj▽hj(X*)+Σμk▽gk(X*) = 0
 * 			 λj≠0， μk≥0， μkgk(X*) = 0
 * 		===>SVM问题满足KKT条件，所以转换为对偶学习问题，所以：
 * 		1) 要让L(w,b,α) 关于w和b最小化
 * 			 分别对w和b求偏导数，令偏导数为0，即
 * 			 ∂L/∂w = 0 => w = Σαiyixi
 * 			 ∂L/∂b = 0 => Σαiyi = 0
 * 			 ===> L(w,b,α) = 0.5‖w‖^2-Σαi(yi(w·xi+b)-1)
 * 			 ===> L(w,b,α) = 0.5Σαiαjyiyjxi^Txj - Σαiαjyiyjxi^Txj - bΣαiyi + Σαi
 * 			               = Σαi - 0.5Σαiαjyiyjxi^Txj
 * 		2) 求L(w,b,α)对α的极大
 * 			 经过步骤1)的变换，问题转换为
 * 			 max{Σαi -  0.5Σαiαjyiyjxi^Txj}
 * 			 s.t. αi≥0
 * 						Σαiyi = 0
 * 		3) SMO算法求对偶因子
 * 10. 核函数：
 *			w = Σαiyixi ===> f(x) = (Σαiyixi)·x+b = Σαiyi<xi,x>+b
 * 		<xi,x>表示二者的内积
 * 		对非线性情况，SVM的处理方法是选择一个核函数<xi,x>，通过将数据映射到
 * 		高维空间，来解决原始空间中线性不可分的问题。
 * 		线性不可分的情况下，支持向量机通过事先选择的非线性映射，即核函数将输入
 * 		变量映射到一个高维空间，在这个空间构造最优分类超平面，使得在高维空间
 * 		中有可能将训练数据实现超平面分割，避免了在原始输入空间中进行非线性曲面
 * 		分割计算。
 *			f(x) = (Σαiyixi)·x+b = Σαiyi<xi,x>+b ==>f(x) = (Σαiyixi)·x+b = Σαiyi<φ(xi),φ(x)>+b
 *			如果纯粹地使用φ，则存在一个问题，即低维映射到高维空间，计算难度爆炸性增长。此时
 * 		应当使用核函数Kernel：
 * 		设向量x1=(η1,η2), x2=(ξ1,ξ2)
 * 		(<φ(x1),φ(x2)>+1)^2 = η1ξ1 + η1^2ξ1^2+η2ξ2+η2^2ξ2^2+η1η2ξ1ξ2+1
 * 		===> φ(X1,X2) = (√2X1，X1^2,√2X2,X2^2,√2X1X2,1)
 * 		区别在于：前者是映射到高维空间中然后根据内积公式计算；后者是直接在原来的低维空间中
 * 		直接计算，不需要显式地写出映射后的结构
 * 		常用核函数：
 * 		高斯核k(x1,x2) = exp(-‖x1-x2‖^2/(2δ^2)),该核函数会将原始空间映射为无穷维空间。
 * 		如果δ选得很大，高次特征的权重也会很快衰减，所以实际上相当于一个低维子空间；如果δ
 * 		选得很小，可以将任意的数据映射为线性可分，有可能造成过拟合 。总体来说，高斯核具有很
 * 		高的灵活性，因此被广泛使用
 * 11.松弛变量：因为数据本身有噪声，而偏离正常位置很远的数据点，称为outlier，为了处理这些
 * 		情况，SVM允许数据点在一定程度上偏离超平面：
 * 		yi(w·xi+b)≥1 =====> yi(w·xi+b)≥1-ξi(ξi≥0，被称为松弛变量)
 * 		====> min 0.5‖w‖^2+CΣξi
 * 					s.t. yi(w·xi+b)≥1-ξi
 * 							 ξi≥0
 * 		====>L(w,b,ξ,α,r) = 0.5‖w‖^2+CΣξi - Σαi(yi(w·xi+b)-1+ξi)-Σriξi
 * 		====> ∂L/∂w = 0 => w = Σαiyixi
 * 					∂L/∂b = 0 => Σαiyi = 0
 * 					∂L/∂ξ = 0 => C-αi-ri = 0
 * 		====> max Σαi - 0.5Σαiαjyiyj<xi,xj>
 * 					s.t. 0≤αi≤C
 * 							 Σαiyi = 0
 * 				(C是一个事先确定好的常量)
 * 		==KKT==>αi = 0 <===> yiμi≥1 正常分类
 * 						0<αi<C <===> yiμi=1 边界上
 * 						αi = C <====> yiμi≤1 两条边界之间
 * 		=======>找到不满足KTT的αi并更新这些αi即可
 * 		由于 w = Σαiyixi 且 Σαiyi = 0
 * 		因此，通过同时更新αi和αj，要求
 * 		αi*yi+αj*yj = αiyi+αjyj = 常数
 * 		如果不考虑0≤αj≤C，则可以得到
 * 		αj* = αj + yi(Ei-Ej)/η
 * 		其中Ei = μi - yi，η = K(x1,x1) +K(x2,x2) - 2K(x1,x2)
 * 		考虑约束，则
 * 		如果αj*≥H ==> αj* <=== H
 *  		如果αj*≤L ==> αj* <=== L
 *  		否则αj* = αj + yi(Ei-Ej)/η
 *  		如果yi≠yj:
 *  		L = max(0,αj-αi), H = min(C,C+αj-αi)
 *  		如果yi=yj
 *  		L = max(0,αi+αj-C), H = min(C, αi+αj)
 *12. 算法：
 * 		创建一个α变量将其初始化为0
 * 		当迭代次数小于最大迭代次数时：
 * 				对数据集中每个数据向量：
 * 					如果该数据向量可以被优化：
 * 						随机选择另外一个数据向量
 * 						同时优化这两个向量
 * 						如果两个向量都不能被优化，退出内循环
 * 				如果所有向量都没有被优化，进行下一次迭代
 * 13. 径向基核函数：采用向量作为自变量的函数，能够基于向量距离输出一个标量。
 * 		高斯核函数：k(x,y) = exp(-‖x-y‖^2/(2δ^2))
 * 							δ是用户定义的用于确定到达率或者函数值跌落到0的速度参数
 */
object SVM {
  
  def selectAnotherAlpha(selectedIndex:Int,numSamples:Int):Int = {
    val random = new Random()
    var nextIndex = selectedIndex
    while(nextIndex == selectedIndex){
      nextIndex = random.nextInt(numSamples)
    }
    nextIndex
  }
  
  def clipAlpha(alpha:Double,upperLimit:Double,lowerLimit:Double) = {
    alpha > upperLimit match {
      case true => upperLimit
      case false => alpha < lowerLimit match{
        case true => lowerLimit
        case false => alpha
      }
    }
  }
  
  def initZeros(length:Int):ListBuffer[Double] = {
    val zeros = new ListBuffer[Double]
    for(index<- 0 until length){
      zeros.+=(0)
    }
    zeros
  }
  
  def simpleSMO(trainSamples:Seq[TrainSample],c:Double,toler:Double,maxCycle:Int):WeightBiasModel = {
    val m = trainSamples.length
    val n = trainSamples(0).features.length
    val labels = trainSamples.map(ts => ts.label)
    val features = trainSamples.map(ts => ts.features)
    var alphas = initZeros(m)
    var b = 0.0
    var iter = 0
    while(iter<maxCycle) {
      var alphaPairsChanged = 0
      for(i<- 0 until m){
        // w=Σαiyixi f(x) = wx+b ==>
        val fXi = Utils.vectorDotVector(Utils.vectorMulVector(alphas, labels), Utils.matrixStarVector(features, features(i)))+b
        val ei = fXi - labels(i)
        if(((labels(i)*ei< -toler) && (alphas(i)<c)) ||((labels(i)*ei > toler) &&(alphas(i)>0))){
          val j = selectAnotherAlpha(i,m)
          val fXj = Utils.vectorDotVector(Utils.vectorMulVector(alphas, labels), Utils.matrixStarVector(features, features(j)))+b
          val ej = fXj - labels(j)
          val alphaIOld = alphas(i)
          val alphaJOld = alphas(j)
          val l = labels(i).!=(labels(j)) match{
            case true => Math.max(0, alphas(j) - alphas(i))
            case false => Math.max(0, alphas(j)+alphas(i)-c)
          }
          val h = labels(i).!=(labels(j)) match{
            case true => Math.min(c, c+alphas(j)-alphas(i))
            case false => Math.min(c, alphas(j)+alphas(i))
          }
          if(l!=h){
            val eta = 2.0*Utils.vectorDotVector(features(i), features(j)) - Utils.vectorDotVector(features(i), features(i)) - Utils.vectorDotVector(features(j), features(j))
            if(eta<0){
              alphas(j) = alphas(j) - labels(j)*(ei-ej)/eta
              alphas(j) = clipAlpha(alphas(j),h,l)
              if(Math.abs(alphas(j) - alphaJOld)>=0.00001){
                alphas(i) = alphas(i)+labels(j)*labels(i)*(alphaJOld-alphas(j))
                val b1 = b-ei - labels(i)*(alphas(i) - alphaIOld)*Utils.vectorDotVector(features(i), features(i)) - labels(j)*(alphas(j)-alphaJOld)*Utils.vectorDotVector(features(i), features(j))
                val b2 = b-ej - labels(i)*(alphas(i) - alphaIOld)*Utils.vectorDotVector(features(i), features(j)) - labels(j)*(alphas(j)-alphaJOld)*Utils.vectorDotVector(features(j), features(j))
                if(0<alphas(i) && alphas(i)<c) b = b1
                else if(0<alphas(j) && alphas(j)<c) b=b2
                else b = (b1+b2)/2.0
                alphaPairsChanged +=1
              }
            }
          }
        }
      }
     if(alphaPairsChanged ==0) 
        iter = iter+1
     else
        iter = 0
    }
     //w=Σαiyixi
    //Utils.vectorDotVector(Utils.vectorMulVector(alphas, labels), Utils.matrixStarVector(features, features(i)))
    val x = alphas.filter(_.>(0))
    val alphay = Utils.vectorMulVector(x, labels)
    var weights = initZeros(m).toSeq
    for(rowIndex<- 0 until features.length){
      weights = Utils.vectorAddVector(weights, features(rowIndex))
    }
    weights = Utils.vectorMulVector(alphay, weights)
    println("x = "+x)
    println("weights = "+weights)
    WeightBiasModel(weights,b)
  }

  def kernelTrans(sampleFeatures:Seq[Seq[Double]],selectedSampleFeature:Seq[Double],delta:Double,kernelType:String){
    val m = sampleFeatures.length
    val n = sampleFeatures(0).length
    var K = kernelType match {
      case "linear" => Utils.matrixStarVector(sampleFeatures, selectedSampleFeature)
      case "rbf" => {
          sampleFeatures.map(f => {
          val deltaRow = Utils.vectorMinusVector(f, selectedSampleFeature)
          Utils.vectorDotVector(deltaRow, deltaRow)
          })
          }
      case _ => Nil
    }
    if(kernelType.equals("rbf")){
      K = Utils.vectorGaussian(K, delta)
    }
    K
  }
  
 
  def main(args: Array[String]) {
    val fileName = System.getProperty("user.dir") + "/src/com/ms/machinelearning/trainsample_svm.txt"
    val trainSamples = Utils.readDataWithAppend(fileName)
    val model = simpleSMO(trainSamples,0.6,0.001,40)
  }
}