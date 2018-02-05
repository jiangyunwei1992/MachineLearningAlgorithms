package com.ms.machinelearning

import scala.collection.mutable.ListBuffer
import scala.util.Random

import com.ms.machinelearning.common.Utils

/*
 * 1. 聚类是一种无监督学习，将类似对象归到同一个簇中。
 * 2. KMeans可以发现k个不同的簇，且每个簇的中心采用簇中所含值
 * 		的均值计算而成.
 * 		算法：
 * 		创建k个点作为起始质心，通常随机选择
 * 		当任意一个点的簇分配结果发生改变时：
 * 			  对数据集中的每个数据点：
 * 					 对每个质心：
 * 						  计算质心与数据点之间的距离：
 * 					 将数据点分配到距其最近的簇
 * 				对每一个簇，计算簇中所有点的均值并将其均值作为质心
 */
object KMeans {
  def calculateDistanceSum(point: Seq[Double], otherPoints: Seq[Seq[Double]]): Double = otherPoints.map(p => calculateEclud(p, point)).sum
  def calculateEclud(vectorA:Seq[Double],vectorB:Seq[Double]):Double = Math.sqrt(vectorA.zip(vectorB).map(p => Math.pow(p._1-p._2, 2)).sum)

  def mostDistantPoint(dataset: Seq[Seq[Double]], selectedCentroids: ListBuffer[Seq[Double]], selectedIndex: ListBuffer[Int])= {

    val (index, distance) = (0 until dataset.length).toList.filter(!selectedIndex.contains(_)).toList.map(index => (index, calculateDistanceSum(dataset(index), selectedCentroids))).maxBy(p => p._2)
    selectedCentroids.+=(dataset(index))
    selectedIndex.+=(index)
  }
  def randCent(dataset: Seq[Seq[Double]], k: Int): ListBuffer[Seq[Double]] = {
    val initialCentroids = new ListBuffer[Seq[Double]]
    val selectedIndex = new ListBuffer[Int]
    //首先随机选择一个点作为第一个初始类簇中心，然后选择距离该点最远的那个点作为第二个簇中心，
    //然后在选择距离前两个点的距离最大的点作为第三个中心，以此类推
    val random = new Random
    val first = random.nextInt(dataset.size)
    selectedIndex.+=(first)
    initialCentroids.+=(dataset(first))

    while (initialCentroids.length < k) {
      mostDistantPoint(dataset, initialCentroids, selectedIndex)
    }
    initialCentroids
  }
  
  def kMeans(dataset:Seq[Seq[Double]],k:Int){
    val m = dataset.length
    val clusterAssment:ListBuffer[ListBuffer[Double]] = Utils.initZerosMatrix(m, 2)
    val centroids = randCent(dataset,k)
    var clusterChanged = true
    while(clusterChanged){
      clusterChanged = false
      for(i<- 0 until m){
        val (minIndex,minDist) = (0 until k).toList.map(j =>(j,calculateEclud(centroids(j),dataset(i)))).maxBy(p => p._2)
        if(clusterAssment(i)(0).!=(minIndex)) clusterChanged = true
        clusterAssment(i)(0) = minIndex
        clusterAssment(i)(1) = Math.pow(minDist, 2)
      }
      //val currentCluster = (0 until m).toList.filter(index => clusterAssment(index)(0)!=0.0)
      
    }
  }
  
  def main(args: Array[String]) {
       val fileName = System.getProperty("user.dir") + "/src/com/ms/machinelearning/trainsamples_knn.txt"
       val dataset = Utils.readDataWithoutLabel(fileName)
       val centroids = randCent(dataset,3)
       println(centroids)
  }
}