---
layout: post
title: Sklearn学习笔记 - 来自莫烦PYTHON
category: Machine Learning
description: 初步了解Sklearn及其基本使用方法，介绍了部分Sklearn的模块，内容较为基础，适合入门学习，内容参看自 莫烦PYTHON。可以参看原博客，支持原创！
---

# Sklearn Study-Notes from 莫烦PYTHON

~~以下代码所使用的 `scikit-learn` 的版本号为 `0.20.3`~~
以下内容参考自 `莫烦PYTHOH` 的博客Sklearn介绍，该博客的发表时间为2016-11左右，由于sklearn更新导致的版本问题，以下内容对代码有部分修改，原博客内容可以参看网址：https://morvanzhou.github.io/tutorials/machine-learning/sklearn/。 

其他参考资料：  

- [sklearn官方文档](https://scikit-learn.org/stable/documentation.html)
- [sklearn官方文档中文版](http://sklearn.apachecn.org/#/)



支持原创，拒绝非法搬运。  -- 2019-03-31

## 1. Sklearn 简介

### 1.1 Why Sklearn?

Scikit learn 也简称 sklearn, 是机器学习领域当中最知名的 python 模块之一.

Sklearn 包含了很多种机器学习的方式:

- Classification 分类
- Regression 回归
- Clustering 非监督分类
- Dimensionality reduction 数据降维
- Model Selection 模型选择
- Preprocessing 数据预处理

我们总能够从这些方法中挑选出一个适合于自己问题的, 然后解决自己的问题.

### 1.2 Sklearn安装

#### 1.2.1 pip安装

```
# python 2+ 版本复制:
pip install -U scikit-learn

# python 3+ 版本复制:
pip3 install -U scikit-learn
```

#### 1.2.2 anacodna安装

```
conda install scikit-learn

```

## 2. 一般使用

### 2.1 选择学习方法

#### 2.1.1 看图选方法

Sklearn 官网提供了一个流程图， 蓝色圆圈内是判断条件，绿色方框内是可以选择的算法：

![image]({{site.baseurl}}/assets/img/ml/2019_3_31/2_1_1.png)

从 START 开始，首先看数据的样本是否 >50，小于则需要收集更多的数据。

由图中，可以看到算法有四类，**分类，回归，聚类，降维**。

其中 **分类和回归**是监督式学习，即每个数据对应一个 label。 **聚类** 是非监督式学习，即没有 label。 另外一类是 降维，当数据集有很多很多属性的时候，可以通过 **降维** 算法把属性归纳起来。例如 20 个属性只变成 2 个，注意，这不是挑出 2 个，而是压缩成为 2 个，它们集合了 20 个属性的所有特征，相当于把重要的信息提取的更好，不重要的信息就不要了。

然后看问题属于哪一类问题，是分类还是回归，还是聚类，就选择相应的算法。 当然还要考虑**数据的大小**，例如 100K 是一个阈值。

可以发现有些方法是**既可以作为分类，也可以作为回归**，例如 SGD。

### 2.2 通用学习模式

#### 2.2.1 要点

Sklearn 把所有机器学习的模式整合统一起来了，学会了一个模式就可以通吃其他不同类型的学习模式。

例如，分类器，

Sklearn 本身就有很多数据库，可以用来练习。 以 Iris 的数据为例，这种花有四个属性，花瓣的长宽，茎的长宽，根据这些属性把花分为三类。

我们要用 分类器 去把四种类型的花分开。

![image]({{site.baseurl}}/assets/img/ml/2019_3_31/2_2_1.png)

今天用 KNN classifier，就是选择几个临近点，综合它们做个平均来作为预测值。

#### 2.2.2 导入模块

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
```

#### 2.2.3 创建数据

加载 `iris` 的数据，把属性存在 `X`，类别标签存在 `y`：

```python
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
```

观察一下数据集，`X` 有四个属性，`y` 有 0，1，2 三类：

```python
print(iris_X[:2, :])
print(iris_y)
```

```
[[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]]
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
```

把数据集分为训练集和测试集，其中 test_size=0.3，即测试集占总数据的 30%：

```python
X_train, X_test, y_train, y_test = train_test_split(
    iris_X, iris_y, test_size=0.3)
```

可以看到分开后的数据集，顺序也被打乱，这样更有利于学习模型：

```python
print(y_train)
print(y_test)
```

```
[2 1 1 0 0 1 0 2 1 2 0 2 2 1 0 1 2 1 2 2 0 2 2 1 1 2 2 2 1 2 2 1 0 0 2 2 2
 2 1 1 2 0 1 2 1 1 2 1 0 1 2 2 0 2 0 1 2 0 1 2 0 1 0 1 2 1 2 0 0 0 2 2 0 0
 0 1 1 2 1 2 1 0 2 2 0 2 1 2 1 0 2 0 1 1 2 1 2 0 1 0 0 1 0 0 0]
[0 2 0 0 1 1 0 0 1 1 0 1 2 2 0 0 0 1 1 2 1 0 2 2 0 2 0 1 0 1 1 1 2 0 1 0 2
 1 0 1 0 0 0 2 1]
```

#### 2.2.4 建立模型－训练－预测

定义模块方式 `KNeighborsClassifier()`， 用 `fit` 来训练 `training data`，这一步就完成了训练的所有步骤， 后面的 `knn` 就已经是训练好的模型，可以直接用来 `predict` 测试集的数据， 对比用模型预测的值与真实的值，可以看到大概模拟出了数据，但是有误差，是不会完完全全预测正确的。



```python
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print(knn.predict(X_test))
print(y_test)
```

```
[0 2 0 0 1 1 0 0 1 1 0 2 2 2 0 0 0 1 1 2 1 0 2 2 0 2 0 2 0 1 1 1 2 0 1 0 2
 1 0 1 0 0 0 1 1]
[0 2 0 0 1 1 0 0 1 1 0 1 2 2 0 0 0 1 1 2 1 0 2 2 0 2 0 1 0 1 1 1 2 0 1 0 2
 1 0 1 0 0 0 2 1]
```

### 2.3 sklearn强大数据库

今天来看 Sklearn 中的 data sets，很多而且有用，可以用来学习算法模型。

#### 2.3.1 要点

eg: boston 房价, 糖尿病, 数字, Iris 花。

也可以生成虚拟的数据，例如用来训练线性回归模型的数据，可以用函数来生成。

![image]({{site.baseurl}}/assets/img/ml/2019_3_31/2_3_1.png)

例如，点击进入 boston 房价的数据，可以看到 sample 的总数，属性，以及 label 等信息

![image]({{site.baseurl}}/assets/img/ml/2019_3_31/2_3_2.png)

如果是自己生成数据，按照函数的形式，输入 sample，feature，target 的个数等等。

```python
sklearn.datasets.make_regression(n_samples=100, n_features=100, n_informative=10, n_targets=1, bias=0.0, effective_rank=None, tail_strength=0.5, noise=0.0, shuffle=True, coef=False, random_state=None)[source]
```

#### 2.3.2 导入模块

导入 `datasets` 包，本文以 `Linear Regression` 为例。

```python
from __future__ import print_function
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
```

#### 2.3.3 导入数据-训练模型

用 `datasets.load_boston()` 的形式加载数据，并给 `X` 和 `y` 赋值，这种形式在 `Sklearn` 中都是高度统一的。

```python
loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target
```

定义模型。

可以直接用默认值去建立 `model`，默认值也不错，也可以自己改变参数使模型更好。 然后用 `training data` 去训练模型。

```python
model = LinearRegression()
model.fit(data_X, data_y)
```



```
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
         normalize=False)
```



再打印出预测值，这里用 `X` 的前 4 个来预测，同时打印真实值，作为对比，可以看到是有些误差的。

```python
print(model.predict(data_X[:4, :]))
print(data_y[:4])
```

```
[30.00384338 25.02556238 30.56759672 28.60703649]
[24.  21.6 34.7 33.4]
```

为了提高准确度，可以通过尝试不同的 `model`，不同的参数，不同的预处理等方法，入门的话可以直接用默认值。

#### 2.3.4 创建虚拟数据－可视化

下面是创造数据的例子。

用函数来建立 100 个 `sample`，有一个 `feature`，和一个 `target`，这样比较方便可视化。

```python
X, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=10)
```

用 `scatter` 的形式来输出结果。

```python
plt.scatter(X, y)
plt.show()
```

![image]({{site.baseurl}}/assets/img/ml/2019_3_31/output_27_0.png)

可以看到用函数生成的 `Linear Regression` 用的数据。

`noise` 越大的话，点就会越来越离散，例如 `noise` 由 10 变为 50.

```python
X, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=50)
plt.scatter(X, y)
plt.show()
```

![image]({{site.baseurl}}/assets/img/ml/2019_3_31/output_29_0.png)

### 2.4 sklearn常用属性与功能

上次学了 `Sklearn` 中的 `data sets`，今天来看 `Model` 的属性和功能。

这里以 `LinearRegressor` 为例，所以先导入包，数据，还有模型。

```python
from sklearn import datasets
from sklearn.linear_model import LinearRegression

loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target

model = LinearRegression()
```

#### 2.4.1 训练和预测

接下来 `model.fit` 和 `model.predict` 就属于 `Model` 的功能，用来训练模型，用训练好的模型预测。

```python
model.fit(data_X, data_y)

print(model.predict(data_X[:4, :]))
```

```
[30.00384338 25.02556238 30.56759672 28.60703649]
```

#### 2.4.2 参数和分数

然后，`model.coef_` 和 `model.intercept_` 属于 `Model` 的属性， 例如对于 `LinearRegressor` 这个模型，这两个属性分别输出模型的斜率和截距（与y轴的交点）。

```python
print(model.coef_)
print(model.intercept_)
```

```
[-1.08011358e-01  4.64204584e-02  2.05586264e-02  2.68673382e+00
 -1.77666112e+01  3.80986521e+00  6.92224640e-04 -1.47556685e+00
  3.06049479e-01 -1.23345939e-02 -9.52747232e-01  9.31168327e-03
 -5.24758378e-01]
36.459488385089855
```

`model.get_params()` 也是功能，它可以取出之前定义的参数。

```python
print(model.get_params())
```

```
{'copy_X': True, 'fit_intercept': True, 'n_jobs': None, 'normalize': False}
```

`model.score(data_X, data_y)` 它可以对 Model 用 `R^2` 的方式进行打分，输出精确度。关于 `R^2 coefficient of determination` 可以查看 `wiki`

```python
print(model.score(data_X, data_y)) # R^2 coefficient of determination
```

```
0.7406426641094095
```

## 3. 高级使用

### 3.1 正规化 Normalization

由于资料的偏差与跨度会影响机器学习的成效，因此正规化(标准化)数据可以提升机器学习的成效。首先由例子来讲解:

#### 3.1.1 数据标准化

```python
from sklearn import preprocessing #标准化数据模块
import numpy as np

#建立Array
a = np.array([[10, 2.7, 3.6],
              [-100, 5, -2],
              [120, 20, 40]], dtype=np.float64)

#将normalized后的a打印出
print(preprocessing.scale(a))
# [[ 0.         -0.85170713 -0.55138018]
#  [-1.22474487 -0.55187146 -0.852133  ]
#  [ 1.22474487  1.40357859  1.40351318]]
```

```
[[ 0.         -0.85170713 -0.55138018]
 [-1.22474487 -0.55187146 -0.852133  ]
 [ 1.22474487  1.40357859  1.40351318]]
```

#### 3.1.2 数据标准化对机器学习成效的影响

加载模块

```python
# 标准化数据模块
from sklearn import preprocessing 
import numpy as np

# 将资料分割成train与test的模块
from sklearn.model_selection import train_test_split

# 生成适合做classification资料的模块
from sklearn.datasets.samples_generator import make_classification 

# Support Vector Machine中的Support Vector Classifier
from sklearn.svm import SVC 

# 可视化数据的模块
import matplotlib.pyplot as plt 
```

生成适合做Classification数据

```python
#生成具有2种属性的300笔数据
X, y = make_classification(
    n_samples=300, n_features=2,
    n_redundant=0, n_informative=2, 
    random_state=22, n_clusters_per_class=1, 
    scale=100)

#可视化数据
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
```

![image]({{site.baseurl}}/assets/img/ml/2019_3_31/output_45_0.png)

数据标准化前

标准化前的预测准确率只有0.477777777778

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = SVC()
#clf = SVC(gamma='scale')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
# 0.477777777778

# 不知道为什么，改为SVC(gamma='scale')后，score可以达到0.97+

```

```
0.4444444444444444

```

```
/anaconda3/envs/deeplearning/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)

```

数据标准化后

数据的单位发生了变化, X 数据也被压缩到差不多大小范围.

```python
X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = SVC(gamma='scale')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
# 0.9

#可视化数据
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

```

```
0.9444444444444444

```


![image]({{site.baseurl}}/assets/img/ml/2019_3_31/output_49_1.png)

标准化后的预测准确率提升至0.9

### 3.2 检验神经网络（Evaluation）

今天我们会来聊聊在做好了属于自己的神经网络之后, 应该如何来评价自己的神经网络, 从评价当中如何改进我们的神经网络. 其实评价神经网络的方法, 和评价其他机器学习的方法大同小异. 我们首先说说为什么要评价,检验学习到的神经网络.

在神经网络的训练当中, 神经网络可能会因为各种各样的问题, 出现学习的效率不高, 或者是因为干扰太多, 学到最后并没有很好的学到规律 . 而这其中的原因可能是多方面的, 可能是数据问题, 学习效率 等参数问题.

#### 3.2.1 Training and Test data

为了检验,评价神经网络, 避免和改善这些问题, 我们通常会把收集到的数据分为训练数据 和 测试数据, 一般用于训练的数据可以是所有数据的70%, 剩下的30%可以拿来测试学习结果.如果你想问为什么要分开成两批, 那就想想我们读书时的日子, 考试题和作业题大部分都是不一样的吧. 这也是同一个道理.

#### 3.2.2 误差曲线

接着, 对于神经网络的评价基本上是基于这30%的测试数据. 想想期末考试虽然花的时间少, 但是占得总成绩肯定要比你平时作业的分多吧. 所以说这30%虽然少, 但是很重要. 然后, 我们就可以开始画图啦! 评价机器学习可以从误差这个值开始, 随着训练时间的变长, 优秀的神经网络能预测到更为精准的答案, 预测误差也会越少 . 到最后能够提升的空间变小, 曲线也趋于水平 . 班上的差生, 从不及格到80分已经不容易啦, 再往上冲刺100分, 就变成了更难的事了. 机器学习也一样. 所以, 如果你的机器学习的误差曲线是这样一条曲线, 那就已经是很不错的学习成果啦.

#### 3.2.3 准确度曲线

同样, 除了误差曲线, 我们可以看他的精确度曲线. 最好的精度是趋向于100%精确. 比如在神经网络的分类问题中, 100个样本中, 我有90张样本分类正确, 那就是说我的预测精确度是90%. 不过, 不知道大家有没有想过对于回归的问题呢? 怎样看预测值是连续数字的精确度? 这时, 我们可以引用 R2 分数在测量回归问题的精度 . R2给出的最大精度也是100%, 所以分类和回归就都有的统一的精度标准. 除了这些评分标准, 我们还有很多其他的标准, 比如 F1 分数 , 用于测量不均衡数据的精度. 由于时间有限, 我们会在今后的视频中继续详细讲解.

#### 3.2.4 正规化

有时候, 意外是猝不及防的, 比如有时候我们明明每一道作业习题都会做, 可是考试分数为什么总是比作业分数低许多? 原来, 我们只复习了作业题,并没有深入, 拓展研究作业反映出来的知识. 这件事情发生在机器学习中, 我们就叫做过拟合. 我们在回到误差曲线, 不过这时我们也把训练误差画出来. 红色的是训练误差, 黑色的是测试误差. 训练时的误差比测试的误差小, 神经网络虽然学习到了知识, 但是对于平时作业太过依赖, 到了考试的时候, 却不能随机应变, 没有成功的把作业的知识扩展开来. 在机器学习中, 解决过拟合也有很多方法 , 比如 l1, l2 正规化, dropout 方法.

#### 3.2.5 交叉验证

神经网络也有很多参数, 我们怎么确定哪样的参数能够更有效的解决现有的问题呢? 这时, 交叉验证 就是最好的途径了. 交叉验证不仅仅可以用于神经网络的调参, 还能用于其他机器学习方法的调参. 同样是选择你想观看的误差值或者是精确度, 不过横坐标不再是学习时间, 而是你要测试的某一参数 (比如说神经网络层数) . 我们逐渐增加神经层, 然后对于每一个不同层结构的神经网络求出最终的误差或精度, 画在图中. 我们知道, 神经层越多, 计算机所需要消耗的时间和资源就越多, 所以我们只需要找到那个能满足误差要求, 有节约资源的层结构. 比如说误差在0.005一下都能接受 , 那我们就可以采用30层的神经网络结构 .

### 3.3 交叉验证 1 Cross-validation

Sklearn 中的 Cross Validation (交叉验证)对于我们选择正确的 Model 和 Model 的参数是非常有帮助的， 有了他的帮助，我们能直观的看出不同 Model 或者参数对结构准确度的影响。

#### 3.3.1 Model 基础验证法

```python
from sklearn.datasets import load_iris # iris数据集
from sklearn.model_selection import train_test_split # 分割数据模块
from sklearn.neighbors import KNeighborsClassifier # K最近邻(kNN，k-NearestNeighbor)分类算法

#加载iris数据集
iris = load_iris()
X = iris.data
y = iris.target

#分割数据并
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)

#建立模型
knn = KNeighborsClassifier()

#训练模型
knn.fit(X_train, y_train)

#将准确率打印出
print(knn.score(X_test, y_test))
# 0.973684210526
```

```
0.9736842105263158
```

可以看到基础验证的准确率为0.973684210526

#### 3.3.2 Model 交叉验证法(Cross Validation)

```python
from sklearn.model_selection import cross_val_score # K折交叉验证模块

#使用K折交叉验证模块
scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')

#将5次的预测准确率打印出
print(scores)
# [ 0.96666667  1.          0.93333333  0.96666667  1.        ]

#将5次的预测准确平均率打印出
print(scores.mean())
# 0.973333333333
```

```
[0.96666667 1.         0.93333333 0.96666667 1.        ]
0.9733333333333334
```

可以看到交叉验证的准确平均率为0.973333333333

#### 3.3.3 以准确率(accuracy)判断 

一般来说`准确率(accuracy)`会用于判断分类(Classification)模型的好坏。

```python
import matplotlib.pyplot as plt #可视化模块

#建立测试参数集
k_range = range(1, 31)

k_scores = []

#藉由迭代的方式来计算不同参数对模型的影响，并返回交叉验证后的平均准确率
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())

#可视化数据
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()
```

![image]({{site.baseurl}}/assets/img/ml/2019_3_31/output_59_0.png)

从图中可以得知，选择 `12~18` 的 `k` 值最好。高过 `18` 之后，准确率开始下降则是因为过拟合(Over fitting)的问题。

#### 3.3.4 以平均方差(Mean squared error) 

一般来说平均方差(Mean squared error)会用于判断回归(Regression)模型的好坏。

```python
import matplotlib.pyplot as plt
k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    loss = -cross_val_score(knn, X, y, cv=10, scoring='neg_mean_squared_error')
    k_scores.append(loss.mean())

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated MSE')
plt.show()
```

![image]({{site.baseurl}}/assets/img/ml/2019_3_31/output_62_0.png)

由图可以得知，平均方差越低越好，因此选择`13~18`左右的`K`值会最好。

### 3.4 交叉验证 2 Cross-validation

sklearn.model_selection 中的 learning curve 可以很直观的看出我们的 model 学习的进度, 对比发现有没有 overfitting 的问题. 然后我们可以对我们的 model 进行调整, 克服 overfitting 的问题.

#### 3.4.1 Learning curve 检视过拟合

加载对应模块:

```python
from sklearn.model_selection import learning_curve #学习曲线模块
from sklearn.datasets import load_digits #digits数据集
from sklearn.svm import SVC #Support Vector Classifier
import matplotlib.pyplot as plt #可视化模块
import numpy as np
```

加载digits数据集，其包含的是手写体的数字，从0到9。数据集总共有1797个样本，每个样本由64个特征组成， 分别为其手写体对应的8×8像素表示，每个特征取值0~16。

```python
digits = load_digits()
X = digits.data
y = digits.target
```

观察样本由小到大的学习曲线变化, 采用K折交叉验证 cv=10, 选择平均方差检视模型效能 scoring='mean_squared_error', 样本由小到大分成5轮检视学习曲线(10%, 25%, 50%, 75%, 100%):

```python
train_sizes, train_loss, test_loss = learning_curve(
    SVC(gamma=0.001), X, y, cv=10, scoring='neg_mean_squared_error',
    train_sizes=[0.1, 0.25, 0.5, 0.75, 1])

#平均每一轮所得到的平均方差(共5轮，分别为样本10%、25%、50%、75%、100%)
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)
```

可视化图形:

```python
plt.plot(train_sizes, train_loss_mean, 'o-', color="r",
         label="Training")
plt.plot(train_sizes, test_loss_mean, 'o-', color="g",
        label="Cross-validation")

plt.xlabel("Training examples")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()
```

![image]({{site.baseurl}}/assets/img/ml/2019_3_31/output_71_0.png)

### 3.5 交叉验证 3 Cross-validation

连续三节的交叉验证(cross validation)让我们知道在机器学习中验证是有多么的重要, 这一次的 sklearn 中我们用到了`learning_curve`当中的另外一种, 叫做`validation_curve`,用这一种曲线我们就能更加直观看出改变模型中的参数的时候有没有过拟合(overfitting)的问题了. 这也是可以让我们更好的选择参数的方法.

#### 3.5.1 validation_curve 检视过拟合 

继续上一节的例子，并稍作小修改即可画出图形。这次我们来验证SVC中的一个参数 gamma 在什么范围内能使 model 产生好的结果. 以及过拟合和 gamma 取值的关系.

```python
from sklearn.model_selection import validation_curve #validation_curve模块
from sklearn.datasets import load_digits 
from sklearn.svm import SVC 
import matplotlib.pyplot as plt 
import numpy as np

#digits数据集
digits = load_digits()
X = digits.data
y = digits.target

#建立参数测试集
param_range = np.logspace(-6, -2.3, 5)

#使用validation_curve快速找出参数对模型的影响
train_loss, test_loss = validation_curve(
    SVC(), X, y, param_name='gamma', param_range=param_range, cv=10, scoring='neg_mean_squared_error')

#平均每一轮的平均方差
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

#可视化图形
plt.plot(param_range, train_loss_mean, 'o-', color="r",
         label="Training")
plt.plot(param_range, test_loss_mean, 'o-', color="g",
        label="Cross-validation")

plt.xlabel("gamma")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()
```

![image]({{site.baseurl}}/assets/img/ml/2019_3_31/output_73_0.png)

由图中可以明显看到gamma值大于0.001，模型就会有过拟合(Overfitting)的问题。

### 3.6 保存模型

总算到了最后一次的课程了,我们训练好了一个Model 以后总需要保存和再次预测, 所以保存和读取我们的sklearn model也是同样重要的一步。这次主要介绍两种保存Model的模块 `pickle` 与 `joblib` 。

#### 3.6.1 使用 pickle 保存

首先简单建立与训练一个 `SVC` Model。

```python
from sklearn import svm
from sklearn import datasets

clf = svm.SVC(gamma='scale')
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X,y)
```



```
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
```



使用pickle来保存与读取训练好的Model。 (若忘记什么是pickle，可以回顾13.8 pickle 保存数据视频。)

```python
import pickle #pickle模块

#保存Model(注:save文件夹要预先建立，否则会报错)
with open('save/clf.pickle', 'wb') as f:
    pickle.dump(clf, f)

#读取Model
with open('save/clf.pickle', 'rb') as f:
    clf2 = pickle.load(f)
    #测试读取后的Model
    print(clf2.predict(X[0:1]))

# [0]
```

```
[0]
```

#### 3.6.2 使用 joblib 保存 

joblib是sklearn的外部模块。

```python
from sklearn.externals import joblib #jbolib模块

#保存Model(注:save文件夹要预先建立，否则会报错)
joblib.dump(clf, 'save/clf.pkl')

#读取Model
clf3 = joblib.load('save/clf.pkl')

#测试读取后的Model
print(clf3.predict(X[0:1]))

# [0]
```

```
[0]
```

最后可以知道joblib在使用上比较容易，读取速度也相对pickle快。