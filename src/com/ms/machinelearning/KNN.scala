package com.ms.machinelearning

import com.ms.machinelearning.common.TrainSample
import com.ms.machinelearning.common.Utils
import scala.collection.mutable.ListBuffer

/*
 * 1. KNN是一种基恩分类和回归方法
 * 2. 输入：实例的特征变量
 * 		输出：实例的类别
 * 3. 给定训练集，其中实例类别已定，分类时，对新的实例点，
 * 		根据其k个最邻近的实力类别，通过多数表决等方式进行预测。
 * 		KNN不具有显式学习过程
 * 4. 单元：对每个训练实例点xi，距离该点比其他点更近的所有点
 * 		组成一个区域，叫做单元。每个训练实例点拥有一个单元，所有
 * 		训练实例点的单元构成对特征空间的一个划分。KNN将实例xi的
 * 		类yi作为其单元中所有类的别急
 * 5. 距离度量：欧氏距离、Lp距离、Minkowski距离
 * 6. K值的选择：
 * 		1) 如果K较小，用较小的邻域中的训练实例进行预测，近似误差会减小，
 * 		估计误差会增大，预测结果会对接近邻域的实例点非常敏感，容易过拟合
 * 		2) 如果K较大，用较大的邻域中的训练实例进行预测，近似误差较大，
 * 		估计误差减小，K的增大意味着整体模型变得简单
 * 		3) K值一般用一个比较小的值，交叉验证选取最优
 * 7. 最简单的KNN是线性扫描，训练集较大时，十分耗时；
 * 		Kd树：对k维空间中的实例点进行存储，以便对其进行快速检索的树形数据
 * 		结构。二叉树，表示对k维空间的一个划分。
 * 8. KD树算法：
 *		 1.选择x(1)为坐标轴，将训练集中所有实例的x(1)坐标的中位数为切分点，
 * 		 将根节点分为对应两个子区域由根节点生成深度为1的左右子节点：左子
 *  		 节点对应的x(1)小于切分点为的子区域，右子节点对应的x(1)大于切分
 *  		 点为的子区域 
 *  2. 对深度j的节点，选择x(l)为切分的坐标轴，l=j%k 
 *  3. 重复上述步骤，直到两个子区域没有实例存在时为止
 */
object KNN {
  class KDTreeNode(trainSample:TrainSample){
    var parentTree:KDTreeNode = null
    var leftTree:KDTreeNode = null
    var rightTree:KDTreeNode = null
    var isLeftTree:Boolean = false
    var isRightTree:Boolean = false
    override def toString:String = {
      trainSample.features.toString()
    }
    def getTrainSample:TrainSample = {
      return trainSample
    }
    def getParent:KDTreeNode = parentTree
  }
  def buildKDTree(trainSamples:Seq[TrainSample],k:Int,depth:Int):KDTreeNode={
    if(trainSamples == null || trainSamples.isEmpty || k == 0) null
    //if there is only 1 sample, then return itself as the tree
    if(trainSamples.length == 1){
      val root:KDTreeNode = new KDTreeNode(trainSamples(0))
      return root
    } else if(trainSamples.length == 2){
      val feature1 = trainSamples(0).features(depth%k)
      val feature2 = trainSamples(1).features(depth%k)
      if(feature1<feature2){
         val root = new KDTreeNode(trainSamples(1))
         val leftTree = new KDTreeNode(trainSamples(0))
         leftTree.isLeftTree = true
         leftTree.parentTree = root
         root.leftTree = leftTree
         return root
      } else {
        val root = new KDTreeNode(trainSamples(0))
        val rightTree = new KDTreeNode(trainSamples(1))
        rightTree.isRightTree = true
        rightTree.parentTree = root
        root.rightTree = rightTree
        return root
      }
    }
    val dimension = depth%k;
    val selectedFeatureValues = trainSamples.map(ts=>ts.features(dimension)).sorted
    //find mid value of the selected feature values
    val midValue = selectedFeatureValues(selectedFeatureValues.length/2)
    //separate the trainSamples into 3 parts
    var root:KDTreeNode = null;
    val leftSide = ListBuffer[TrainSample]()
    val rightSide = ListBuffer[TrainSample]()
    for(index <- 0 until trainSamples.length){
      val trainSample = trainSamples(index)
      if(trainSample.features(dimension)<midValue){
        leftSide.+=(trainSample)
      } else if(root==null&&trainSample.features(dimension)==midValue){
        root = new KDTreeNode(trainSample)
      } else {
        rightSide.+=(trainSample)
      }
    }
    val leftTree = buildKDTree(leftSide,k,depth+1)
    val rightTree = buildKDTree(rightSide,k,depth+1)
    root.leftTree = leftTree;
    root.rightTree = rightTree
    if(leftTree!=null){
      leftTree.parentTree = root
      leftTree.isLeftTree = true
    }
    if(rightTree!=null){
      rightTree.parentTree = root;
      rightTree.isRightTree = true
    }
    return root
  }
  
  def traverseTree(root:KDTreeNode){
    if(root==null) return
    println(root.toString())
    traverseTree(root.leftTree)
    traverseTree(root.rightTree)
  }
  
  def calculateDistance(node1:Seq[Double],node2:Seq[Double]):Double = {
    Math.sqrt(node1.zip(node2).map(p => Math.pow(p._1-p._2, 2)).sum)
  }
  
  def findStartNode(features:Seq[Double],root:KDTreeNode,depth:Int):KDTreeNode = {
    if(root == null || root.leftTree == null && root.rightTree == null) root
    val k = features.length
    val index = depth%k
    val element1 = features(index)
    val element2 = root.getTrainSample.features(index)
    if(root.rightTree == null) {
      element1<element2 match{
        case true => return findStartNode(features,root.leftTree,depth+1)
        case false => return root
      }
    }
    if(root.leftTree == null){
      element1>element2 match{
        case true => return findStartNode(features,root.rightTree,depth+1)
        case false => return root
      }
    }
    element1<element2 match{
      case true => return findStartNode(features,root.leftTree,depth+1)
      case false => return findStartNode(features,root.rightTree,depth+1)
    }
  }
  
  def searchKDTree(features:Seq[Double],startNode:KDTreeNode,closestNode:KDTreeNode,depth:Int):KDTreeNode = {
    var closest =  closestNode == null match{
      case true => startNode
      case false => closestNode
    }
    if(startNode == null || startNode.getParent == null){
      return closestNode
    }
    val currentDistance = calculateDistance(features,closest.getTrainSample.features)
    val nextDistance = calculateDistance(features,startNode.getTrainSample.features)
    if(nextDistance<currentDistance){
      val nextDimensionIndex = (depth+1)%features.size
      val nextDimensionDistance = Math.abs(features(nextDimensionIndex)-startNode.getTrainSample.features(nextDimensionIndex))
      val intersect = nextDimensionDistance < nextDistance
      if(intersect){
        closest = startNode
        closest.isLeftTree match{
          case true => return searchKDTree(features, closest.rightTree,closest,depth+1);
          case false =>return searchKDTree(features, closest.leftTree,closest,depth+1);
        }
      }
    }
    return searchKDTree(features, startNode.parentTree,closest,depth);
  }
  
  def main(args:Array[String]):Unit={
     val fileName = System.getProperty("user.dir") + "/src/com/ms/machinelearning/trainsample_knn.txt"
     val trainSamples = Utils.readData(fileName)
     val root = buildKDTree(trainSamples,2,0)
     val features = Seq(51.0,41.0)
     val startNode = findStartNode(features,root,0)
     val result = searchKDTree(features,startNode,null,0)
     println(result)
  }
}