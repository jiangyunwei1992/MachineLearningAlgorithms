package com.ms.machinelearning

import com.ms.machinelearning.common.Utils
import scala.collection.mutable.ListBuffer

/*
 * 1. 从大规模数据集中寻找物品建的隐含关系称作关联分析
 * 2. 频繁项集：经常出现在一块的物品的集合 
 * 		支持度：数据集中该项集的记录所占的比例
 * 		可信度/置信度：针对一条关联规则定义的
 * 3. Apriori原理：如果某个项集是频繁的，那么它所有的
 * 		子集也是频繁的 <===>如果某个项集是非频繁集，那么
 * 		它所有的超集也是非频繁的
 */
object Apriori {
  def scanDataset(dataset:Seq[Seq[Double]],ck:Seq[Double],minSupport:Double):Tuple2[Seq[Double],Map[Double,Double]]={
    val ssCnt = dataset.flatMap(row => row).groupBy(v =>v).map(e => (e._1,e._2.size))
    val numItems = dataset.length
    val supportData = ssCnt.map(e =>(e._1,e._2*1.0/numItems))
    val retList = supportData.filter(e => e._2>=minSupport).keys.toSeq
    (retList,supportData)
  }
  def apriori(dataset:Seq[Seq[Double]],minSupport:Double){
    val c1 = dataset.flatMap(row => row).distinct.sorted
    val (l1,supportData) = scanDataset(dataset,c1,minSupport)
    val l = ListBuffer(l1)
    var k = 2
    while(l(k-2).length>0){
      val ck = aprioriGen
    }
  }
  
  def main(args: Array[String]) {
       val fileName = System.getProperty("user.dir") + "/src/com/ms/machinelearning/trainsample_apriori.txt"
       val dataset = Utils.readDataWithoutLabel(fileName)
       apriori(dataset,0.5)
  }
}