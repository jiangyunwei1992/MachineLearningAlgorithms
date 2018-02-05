package com.ms.machinelearning

import com.ms.machinelearning.common.TrainSample
import com.ms.machinelearning.common.Utils

/*
 *  1. 分类决策树是一种描述对实例进行分类的树形结构。
 *  2. 决策树由节点节点和有向边组成。
 *  3. 决策树进行分类时，从根节点开始，对实例的某一特征进行测试，
 *  		 根据测试结果，将实例分配到子节点。此时每一个叶子节点对应
 *  		 该特征的一个取值。如此递归对实例进行测试并分类，直到到达
 *  		 叶子节点，最后将实例分到叶子节点的类中。
 *  4. 决策树可以看成一个if-else规则的集合：由决策树的根节点到
 *  		 叶节点的每一条路径构建一条规则，路径上的内部节点的特征对
 *  		 应规则的条件，而叶子节点的类对应规则的结论。决策树的路径
 *  		 或其对应的if-else规则集合具有一个重要性质：互斥并且完备。
 *  		 即每一个实例都被一条路径或者一条规则覆盖，而且只被一条规则
 *  		 覆盖。
 *  5. 本质：从训练集中归纳出一组分类规则，与训练数据集不矛盾的
 *  		 决策树有多个或者一个都没有。此时，需要找到一个与训练数据
 *  		 矛盾较小的决策树，同时具有很好的泛化能力。从另外一个角度
 *  		 来看，决策树是训练集估计条件概率模型。
 *  6. 特征选择：信息增益或者信息增益比
 *  		 1) 熵：H(X) = -Σp_ilogp_i，熵越大，随机变量不确定性越大
 *  		 2) 条件熵：H(Y|X)表示在已知X条件下，Y的不确定性
 *  				H(Y|X) = Σp_iH(Y|X=xi)
 *  				熵H(Y)与H(Y|X)之差称为互信息
 *  		 3) 信息增益：表示得知特征X的信息而使得类Y的信息不确定性减少的
 *  				程度：g(D,A) = H(D) - H(D|A)
 *  			  计算过程：
 *  				a) 计算数据集D的经验熵H(D) = -Σ|Ck/D|log2|Ck/D|
 *  				b) 计算特征A对数据集D的经验条件熵H(D|A) = -Σ|Di/D|Σ|Dik/Di|log2|Dik/Di|
 *  				c) 计算信息增益g(D,A) = H(D) - H(D|A)
 *  		 4) 信息增益比：信息增益的大小是相对于训练集而言的，无绝对意义。在分类问题困难时，
 *  				即训练集的经验熵大的时候，信息增益会偏大；繁殖，信息增益偏小。此时使用信息增
 *  				益比可以对这一问题进行校正:
 *  				gr(D,A) = g(D,A)/H(D)
 *  7. ID3:核心是在决策树的各个节点上应用信息增益原则选择特征。递归地构建决策树。
 *  		 算法：
 *  		 1) 若D中所有实例均属于同一类Ck，则T为单节点树，将Ck作为该节点的类标记，返回T
 *  		 2) 如果特征集A=∅，则T为单节点树，并将D中实例数最大的类Ck作为该节点的类标记，返回T
 *  		 3) 计算A中各特征对D的信息增益，选择信息增益最大的特征Ag
 *  		 4) 如果Ag的信息增益小于阈值ε，则T为单节点数，将D中实例数最大的类Ck作为该节点的
 *  				类标记，返回T
 *  		 5) 否则，对Ag的每一个可能值ai，依照Ag = ai将D分割成若干非空子集Di，将Di中实例数
 *  				最大的类作为标记，构建子节点，由节点及其子节点构成树，返回T；
 *  		 6) 对第i个子节点，以Di为训练集，以A-{Ag}为特征集，递归地调用1)~5)步，得到子树Ti,
 *  				返回Ti
 *  8. C4.5 与ID3基本相同，区别在于使用信息增益比，而不是信息增益
 *  9. CART树：递归地构建二叉决策树，对回归树用平方误差最小化准则；对分类树用基尼指数最小化
 *  		 准则，进行特征选择，生成二叉树。
 *  		 1) 回归树的生成：假设X和Y分别表示输入/输出变量，且Y是连续变量。一棵回归树对应输入空间
 *  				的划分以及在划分的单元上的输出值。假设将输入控件划分为M个单元R1, R2, ..., RM，且
 *  				每个单元Rm上有一个固定的输出值cm，于是回归树模型可以表示为
 *  						f(x)=ΣcmI
 *  				当输入空间的划分确定时，可以用平方误差表示回归树对于训练数据的预测误差，用平方误差最
 *  				小的准则求解每个单元上的最优输出值。
 *  			 	算法：
 *  					a) 选择最优切分变量j和切分点s，求解
 *  							min{min{Σ(yi-c1)^2}+min{Σ(yi-c2)^2}}
 *  						 遍历变量j，对固定的切分变量j扫描切分点s，选择使上式最小的(j,s)对
 *  					b) 用选定的(j,s)划分区域并决定相应的输出值
 *  						 R1(j,s) = {x|x(j)<=s}, R2(j,s) = {x|x(j)>s}
 *  						 cm = Σyi/Nm
 *  					c) 继续对两个子区域调用步骤a)和b)，直到满足停止条件
 *  					d) 将输入控件划分为M个区域，生成决策树
 *  			2) 分类树的生成：
 *  				 基尼指数Gini(p) = 1-Σp_k^2
 *  				 基尼指数表示集合的不确定性，基尼指数Gini(D,A)表示经A=a分割后
 *  				 集合D的不确定性，基尼指数越大，样本集合的不确定性也就越大。
 *  				 算法：
 *  				 a) 设节点的训练数据集为D，计算现有特征对该数据集的基尼指数。
 *  						此时，对每一个特征A，对其可能取的每个值a，根据样本点对
 *  						A=a的测试为"是"或者"否"将D分割成D1和D2两个部分，按照
 *  						下式计算基尼指数：
 *  						Gini(D,A) = |D1/D|Gini(D1)+|D2/D|Gini(D2)
 *  				 b) 在所有可能的特征A以及它们所有可能的切分点a中，选择基尼指数
 *  						最小的特征及其对应的切分点作为最有特征与最优切分点，依最优
 *  						特征与最优切分点，从现节点生成两个子节点，将训练集依照特征
 *  						分配到两个子节点中
 *  				 c) 对两个子节点递归调用a)和b)，直到满足停止条件
 *  				
 *
 */
object DecisionTree {
  val featureDescriptions = Seq("年龄","有工作","有自己的房子","信贷情况")
  val featureValueMap = Map(("年龄",Map((1,"青年"),(2,"中年"),(3,"老年"))),("有工作",Map((1,"是"),(2,"否"))),("有自己的房子",Map((1,"是"),(2,"否"))),("信贷情况",Map((1,"一般"),(2,"好"),(3,"非常好"))))
  class DecisionTreeNode{
     var featureDescription:String = null
     var featureIndex:Int = -1
     var featureValue:Double = -1
     var subTrees:Seq[DecisionTreeNode] = null
     var label:Double = -1
  }
  def calculateEntropy(trainSamples:Seq[TrainSample]):Double = {
    val numSamples = trainSamples.length
    val groupedSamples = trainSamples.groupBy(ts => ts.label)
    groupedSamples.map(gs => gs._2.length*1.0/numSamples).map(v => -v*Math.log(v)/Math.log(2)).sum
  }
  
  def calculateConditionalEntropy(trainSamples:Seq[TrainSample],featureIndex:Int):Double = {
    val numSamples = trainSamples.length
    val groupedSamples = trainSamples.groupBy(ts => ts.features(featureIndex))
    val conditionalEntropy = groupedSamples.map(gs =>calculateEntropy(gs._2)*gs._2.size/numSamples).sum
    conditionalEntropy
  }
  
  def calculateInfoGain(trainSamples:Seq[TrainSample],featureIndex:Int):Double = {
    calculateEntropy(trainSamples) - calculateConditionalEntropy(trainSamples,featureIndex)
  }
  
  def calculateInfoGainRatio(trainSamples:Seq[TrainSample],featureIndex:Int): Double ={
    val entropy = calculateEntropy(trainSamples)
    val infoGain = calculateInfoGain(trainSamples,featureIndex)
    infoGain/entropy
  }
  
  def buildID3Tree(trainSamples:Seq[TrainSample],featureCandidates:Seq[Int],epsilon:Double):DecisionTreeNode = {
    if(trainSamples == null|| trainSamples.isEmpty) null
    //1. 若D中所有实例属于同一类Ck，则T为单节点树，将Ck作为该节点的类标记，返回T
    val numClasses = trainSamples.map(ts => ts.label).distinct.length
    if(numClasses == 1) {
      val tree:DecisionTreeNode = new DecisionTreeNode();
      tree.label = trainSamples(0).label
      return tree
    }
    //2. 如果A=∅，则T为单节点树，将D中实例最大的类Ck作为该节点的类标记，返回T
    if(featureCandidates == null || featureCandidates.isEmpty){
      val groupedSamples = trainSamples.groupBy(ts => ts.label)
      val targetLabel = groupedSamples.maxBy(gs => gs._2.length)._2(0).label
      val tree:DecisionTreeNode = new DecisionTreeNode()
      tree.label = targetLabel
      return tree
    }
    //3. 计算A中各特征值D的信息增益，选择信息增益最大的特征Ag
    val indexedInfoGain = featureCandidates.map(fc => (fc,calculateInfoGain(trainSamples,fc)))
    val selectedFeature = indexedInfoGain.maxBy(ig => ig._2)._1
    val maxInfoGain = indexedInfoGain.maxBy(ig => ig._2)._2
    //4. 如果Ag的信息增益小于阈值，则置T为单节点，将D中实例数最大的类Ck作为该节点的类标记，返回T
    if(maxInfoGain<epsilon){
      val groupedSamples = trainSamples.groupBy(ts => ts.label)
      val targetLabel = groupedSamples.maxBy(gs => gs._2.length)._2(0).label
      val tree:DecisionTreeNode = new DecisionTreeNode()
      tree.label = targetLabel
      return tree
    }
    //5. 对Ag的每一可能值ai，依照Ag=ai将D划分为若干个非空子集Di，将Di中实例最大的数作为类标记，
    //   构建子节点，由节点及其子节点构成树T，返回T
    val subsets = trainSamples.groupBy(ts => ts.features(selectedFeature))
    val root:DecisionTreeNode = new DecisionTreeNode()
    root.featureDescription = featureDescriptions(selectedFeature)
    root.featureIndex = selectedFeature
    val remainedFeatures = featureCandidates.filter(fc => fc!=selectedFeature)
    val subTrees = subsets.map(ss =>{
      val st = buildID3Tree(ss._2,remainedFeatures,epsilon)
      st.featureValue = ss._2(0).features(selectedFeature)
      st
    }).toList
    root.subTrees = subTrees
    return root
  }
  
  def buildC45Tree(trainSamples:Seq[TrainSample],featureCandidates:Seq[Int],epsilon:Double):DecisionTreeNode = {
    if(trainSamples == null|| trainSamples.isEmpty) null
    //1. 若D中所有实例属于同一类Ck，则T为单节点树，将Ck作为该节点的类标记，返回T
    val numClasses = trainSamples.map(ts => ts.label).distinct.length
    if(numClasses == 1) {
      val tree:DecisionTreeNode = new DecisionTreeNode();
      tree.label = trainSamples(0).label
      return tree
    }
    //2. 如果A=∅，则T为单节点树，将D中实例最大的类Ck作为该节点的类标记，返回T
    if(featureCandidates == null || featureCandidates.isEmpty){
      val groupedSamples = trainSamples.groupBy(ts => ts.label)
      val targetLabel = groupedSamples.maxBy(gs => gs._2.length)._2(0).label
      val tree:DecisionTreeNode = new DecisionTreeNode()
      tree.label = targetLabel
      return tree
    }
    //3. 计算A中各特征值D的信息增益，选择信息增益最大的特征Ag
    val indexedInfoGainRatio = featureCandidates.map(fc => (fc,calculateInfoGainRatio(trainSamples,fc)))
    val selectedFeature = indexedInfoGainRatio.maxBy(ig => ig._2)._1
    val maxInfoGain = indexedInfoGainRatio.maxBy(ig => ig._2)._2
    //4. 如果Ag的信息增益小于阈值，则置T为单节点，将D中实例数最大的类Ck作为该节点的类标记，返回T
    if(maxInfoGain<epsilon){
      val groupedSamples = trainSamples.groupBy(ts => ts.label)
      val targetLabel = groupedSamples.maxBy(gs => gs._2.length)._2(0).label
      val tree:DecisionTreeNode = new DecisionTreeNode()
      tree.label = targetLabel
      return tree
    }
    //5. 对Ag的每一可能值ai，依照Ag=ai将D划分为若干个非空子集Di，将Di中实例最大的数作为类标记，
    //   构建子节点，由节点及其子节点构成树T，返回T
    val subsets = trainSamples.groupBy(ts => ts.features(selectedFeature))
    val root:DecisionTreeNode = new DecisionTreeNode()
    root.featureDescription = featureDescriptions(selectedFeature)
    root.featureIndex = selectedFeature
    val remainedFeatures = featureCandidates.filter(fc => fc!=selectedFeature)
    val subTrees = subsets.map(ss =>{
      val st = buildID3Tree(ss._2,remainedFeatures,epsilon)
      st.featureValue = ss._2(0).features(selectedFeature)
      st
    }).toList
    root.subTrees = subTrees
    return root
  }
  
  def traverseDecisionTree(tree:DecisionTreeNode) {
    if(tree == null) return
    val description = tree.featureDescription == null match{
      case true => "label is "+tree.label+", featureValue = "+tree.featureValue
      case false => "feature is "+tree.featureDescription+", and featureIndex = "+tree.featureIndex
    }
    println(description)
    if(tree.subTrees==null) return
    tree.subTrees.foreach(st =>traverseDecisionTree(st))
  }
  
  def predict(root:DecisionTreeNode,feature:Seq[Double]):Double = {
    if(root.featureIndex < 0 || root.featureDescription == null || root.subTrees == null) 
      return root.label   
    val selectedFeatureDesc = featureDescriptions(root.featureIndex)
    val selectedFeatureValue = feature(root.featureIndex)
    val selectedSubTree = root.subTrees.filter(st => st.featureValue==selectedFeatureValue)(0)
    val label = predict(selectedSubTree,feature)
    label
  }
  
 
  ////////////////////////////////
  def main(args: Array[String]) {
     val fileName = System.getProperty("user.dir") + "/src/com/ms/machinelearning/trainsample_decisiontree.txt"
     val trainSamples = Utils.readData(fileName)
     val featureCandidates = Seq(0,1,2,3)
     val tree = buildID3Tree(trainSamples,featureCandidates,0)
     val sample = Seq(1.0,0.0,1.0,1.0)
     val label = predict(tree,sample)
     println(label)
     //CART
  }
}