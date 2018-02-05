package com.ms.machinelearning.common

import scala.collection.mutable.ListBuffer
import scala.io
import scala.io.Source
import Jama.Matrix

object Utils {
  def readData(fileName:String):Seq[TrainSample] = {
    val array = Source.fromFile(fileName).getLines().toList.map(_.split(","))
    val trainSamples = ListBuffer[TrainSample]()
    for(index<- 0 until array.length){
      val features = ListBuffer[Double]()
      for(eleIndex<- 0 until array(index).length-1){
         features.+=(array(index)(eleIndex).toDouble)
      }
      val label = array(index)(array(index).length-1).toDouble
      val trainSample = TrainSample(features,label)
      trainSamples.+=(trainSample)
    }
    trainSamples
  }
  
  def readDataWithAppend(fileName:String):Seq[TrainSample] = {
    val array = Source.fromFile(fileName).getLines().toList.map(_.split(","))
    val trainSamples = ListBuffer[TrainSample]()
    for(index<- 0 until array.length){
      val features = ListBuffer[Double]()
      features.+=(1)
      for(eleIndex<- 0 until array(index).length-1){
         features.+=(array(index)(eleIndex).toDouble)
      }
      val label = array(index)(array(index).length-1).toDouble
      val trainSample = TrainSample(features,label)
      trainSamples.+=(trainSample)
    }
    trainSamples
  }
  
  def readDataWithoutLabel(fileName:String):Seq[Seq[Double]] = {
    val array = Source.fromFile(fileName).getLines().toList.map(_.split(","))
    val trainSamples = array.map(arr => arr.map(e => e.toDouble).toSeq)
    trainSamples
  }
  
  def vectorDotVector(vector1:Seq[Double],vector2:Seq[Double]):Double = {
    vector1.zip(vector2).map(p => p._1*p._2).sum
  }
  
   def vectorMulVector(vector1:Seq[Double],vector2:Seq[Double]):Seq[Double] = {
    vector1.zip(vector2).map(p => p._1*p._2)
  }
  
  def constantMulVector(constant:Double,vector:Seq[Double]):Seq[Double] = {
    vector.map(x => x.*(constant))
  }
  
  def vectorAddVector(vector1:Seq[Double],vector2:Seq[Double]):Seq[Double] = {
    vector1.zip(vector2).map(p => p._1 + p._2)
  }
  
  def matrixStarVector(matrix: Seq[Seq[Double]], vector: Seq[Double]): Seq[Double] = {
    matrix.map(row => vectorDotVector(row, vector))
  }
  
  def matrixTranspose(matrix:Seq[Seq[Double]]):Seq[Seq[Double]] = {
    val result = new ListBuffer[Seq[Double]]
    for(col<-0 until matrix(0).length){
      val row = matrix.map(row => row(col))
      result.+=(row)
    }
    result
  }
  
  def vectorMinusVector(vector1:Seq[Double],vector2:Seq[Double]):Seq[Double] = vector1.zip(vector2).map(z =>z._1-z._2)

  def vectorGaussian(vector:Seq[Double],delta:Double):Seq[Double] = {
    vector.map(v => Math.exp(-v/(2*Math.pow(delta, 2))))
  }
  
  def initZeros(m:Int):Seq[Double] = (0 until m).toList.map(e => 0.0)
  def initOnes(m:Int):Seq[Double] = (0 until m).toList.map(e => 1.0)
  def initEyes(m:Int):Seq[Seq[Double]] = (0 until m).toList.map(i => (0 until m).toList.map(j =>(i==j) match{
    case true => 1.0
    case false => 0.0
  }))
  def initZerosMatrix(m:Int,n:Int):ListBuffer[ListBuffer[Double]] ={
    val result = new ListBuffer[ListBuffer[Double]]
    for(i<-0 until m){
      for(j<-0 until n){
        result(i).+=(0)
      }
    }
    null
  }
  
  def vectorExp(vector:Seq[Double]):Seq[Double] = vector.map(v => Math.exp(v))
  def vectorSign(vector:Seq[Double]):Seq[Double] = vector.map(v => v.>(0) match {
    case true => 1.0
    case false => -1.0
  })
  
  def vectorInequalVector(vector1:Seq[Double],vector2:Seq[Double]):Seq[Double] = vector1.zip(vector2).map(p =>p._1.!=(p._2) match {
    case true => 1.0
    case false => 0
  })
  
  def getMatrixColumn(matrix:Seq[Seq[Double]], column:Int):Seq[Double] = matrix.map(row => row(column))
  
  def matrixStarMatrix(matrix1:Seq[Seq[Double]],matrix2:Seq[Seq[Double]]):Seq[Seq[Double]] =     (0 until matrix1.length).toList.map(i => (0 until matrix2(0).length).toList.map(j => vectorDotVector(matrix1(i),getMatrixColumn(matrix2,j))))

  def matrixDet(matrix:Seq[Seq[Double]]):Double= {
    val matrixArray = Array.ofDim[Double](matrix.length,matrix(0).length)
    (0 until matrix.length).toList.map(i =>(0 until matrix(0).length).toList.map(j =>matrixArray(i)(j) = matrix(i)(j)))
    val orginalMatrix = new Matrix(matrixArray)
    orginalMatrix.det()
  }
  
  def matrixInverse(matrix:Seq[Seq[Double]]):Seq[Seq[Double]] = {
      val matrixArray = Array.ofDim[Double](matrix.length,matrix(0).length)
    (0 until matrix.length).toList.map(i =>(0 until matrix(0).length).toList.map(j =>matrixArray(i)(j) = matrix(i)(j)))
    val orginalMatrix = new Matrix(matrixArray)
    orginalMatrix.inverse().getArray.map(arr => arr.toSeq).toSeq
  }

  def matrixAddMatrix(matrix1:Seq[Seq[Double]],matrix2:Seq[Seq[Double]]):Seq[Seq[Double]] = (0 until matrix1.length).toList.map(index => Utils.vectorAddVector(matrix1(index), matrix2(index)))
  def constantMulMatrix(constant:Double,matrix:Seq[Seq[Double]]) = matrix.map(row => Utils.constantMulVector(constant, row))
}