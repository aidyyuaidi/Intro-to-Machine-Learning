# Naive Bayes
## Used to find the decision surface
## Gaussian Naive Bayes: implements the Gaussian Naive Bayes algorithm for classification.
## The likelihood of the features is assumed to be Gaussian.

import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
## Supervised learning process will always have the features first, and then the lables.
clf.fit(X, Y)
print(clf.predict([[-0.8, -1]]))

## You should always save about 10% of your data as the test data.

## Text Learning
## Naive Bayes does huge multiplications and gives you the ratios for the observation w.r.t. different person.
## However, it ignores the order of the text.

## Naive Bayes is easy to impliment, can handle large number of features and is efficient.
## But it cannot handle meaningful phrases since it does not consider orders.

## When you do ML, you should always think about what the problem is and which algorithm is suitable. Also, you should run tests.

## Mini Projecct: identify the author of the emails.

# SVM
## SVM is used for finding the hyper plane between data from two classes.
## SVM maximizes the margins between the line and the nearest points for either class.
## Maximizing the margins ensures the fit is as robust as possible.
## Note that SVM put different groups of the data on different side of the hyperplane first and then maximize the margin!
## SVMs are somewhat robust to outliers. The parameters determines how willing it is w.r.t. ignoring outliers.

## Advantages:
## Effective in high dimensional spaces.
## Still effective in cases where number of dimensions is greater than the number of samples.
## Memory efficient.
## Versatile. Can specify different kernel functions.

## Disadvantages:
## If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.
## SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation.
## Doesn't perform so well on extreme large data sets (MUCH slower than Naive Bayes!).
## Doesn't work well with lots of noise (When the classes are overlapping, Naive Bayes will be better!).
## If you have a very large data set with lots lots of features, SVM might overfit.

from sklearn.svm import SVC
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = SVC(kernel = "linear")
clf.fit(X, y)
clf.predict(([[2., 2.]]))

## Adding non-linear features will make SVM develope a non-linear hyperplane!
## But you don't have to mannualy add those non-linear features because there is Kernel tricks!
## You use Kernels to change from small input space to a much larger input space so that the points are now linearly seperable.

## Parameter: C
## Controls the tradeoff between smooth decision bundary and classifying training points correctly.
## Be careful of overfitting. If C is too large, the classifier might not work as well on your testing set!

## Parameter: Gamma
## Defines how far the influence of a single training example reaches.
## Low values: far reach; High values: close reach.
## Meaning if we have a high value of Gamma, the decision bundary will only depends on points that are very close to the line.
## So if Gamma is large, essentially we are ignoring the points that are far from the decision bundaries.
## High Gamma: wiggly desicion bundary.
## Low Gamma: less jagging, smoother decision bundary.

## AVOID OVERFITTING!!
## In SVM, Kernel, C and Gamma all affects the level of overfitting.

## Note that if a process needs to be done real-time, then it certainly need to train very fast.

## Knowing which one to try when you're tackling a problem for the first time is part of the art and science of machine learning.
## In addition to picking your algorithm, depending on which one you try, there are parameter tunes to worry about as well, and the possibility of overfitting (especially if you don't have lots of training data).
## General suggestion: try a few different algorithms for each problem.

# Decision Trees
## Decision trees allow you to ask muiltiple linear questions one after another, and then build up a decision bundary out of it.
## It can be used for classifications or regressions.

from sklearn import tree
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier(min)
clf = clf.fit(X, Y)
clf.predict([[2., 2.]])

## Parameters
### min_samples_split: whether I can keep splitting or not, whether there is enough samples in the leaf for me to keep splitting.
### small values of min_sample_split will give more complecated bundaries but may overfitting.

## Entropy: how the DT decides where to split the data.
## Entropy definition: it is a measure of impurity in a bunch of examples. Opposition of purity.
## When DT decides, it tries to make every subset as pure as possible.
## Entropy is between 0 and 1. When the points are completely pure, entropy = 0; when points are evenly splitted, entropy = 1.

## Calculating entropy
import math
from fractions import Fraction

-Fraction(2,3)*math.log(Fraction(2,3), 2)-Fraction(1,3)*math.log(Fraction(1,3),2)

## Information Gain:
### IG = Entropy(parent) - (weighted average)entropy(children)
## The decision tree will maximize the information gain when deciding which feature to choose first.

## Bias-Variance Dilemma
## An extremely high bias ML algorithm won't change based on new training sets.
## An extremely high variance ML algorithm will behave only based on the training set.
## In reality you want something in the middle.

## DTs are very easy to use and graphically good.
## However, it tends to overfit especially when you have a lot of features.

# K nearest neighbours: classic, simple, easy to understand.
## The classification is based on the votes of K points that's closest to the test point.
## If the classes overlap each other, KNN is better.
## It has the disadvantage of requiring the training set to be balanced.
X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

print(neigh.predict([[1.1]]))

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=10)

# Adaboost & Random Forest: ensemble methods.
## meta classifiers built from (usually) decision trees.

# Adaboost
## Use different weak classifiers w.r.t. the same training set, then combine them to form a strong classifier.
## Won't overfitting.
## Feature selection.
from sklearn.ensemble import AdaBoostClassifier
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = AdaBoostClassifier()
clf = clf.fit(X, Y)
print clf.predict([[2., 2.]])

from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(learning_rate=1)

# Random Forest
## Use subset of training set to train classifiers, and then combine them to form a big DT.
from sklearn.ensemble import RandomForestClassifier
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = RandomForestClassifier()
clf = clf.fit(X, Y)
print clf.predict([[2., 2.]])

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=6)

# Datasets and Questions
## Identify the POIs.
## The size of the training set usually have a big impact on the accuracy of your classifier.
## The speed of accuracy improvement drops significantly as the size of the training set goes up.
## Important: how many examples we can get for POIs?
## If the training sets are too small, the accuracy of the classifier will be low.
## In general, More Data > Fine-Tuned Algorithm

## When generating or augmenting a dataset, you should be exceptionally careful if your data are coming from different sources for different classes.

# Regressions
## Continuous Supervised Learning
## Continuous: some kind of ordering to the dataset.

from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
print reg.coef_
print reg.intercept_
reg.predict([[2, 3]])

## Use reg.score(ages_test, net_worths_test) to see the R square score on your testing data.
## If there's overfitting issue going on, you will see a quite low R-square score on your testing data.

## Linear Regression Errors
### error = actual value - predicted value
### actural values are taken from your training data, and predicted value comes from the regression.

### Algorithms to minimize the sum of squared errors:
### OLS
### Gradient Descent

### Minimizing SSE isn't perfect!
### As you add more data, SSE will almost certainly go up.
### That's why r^2 score is a better measurement to evaluate your regression.

### r^2: How much of my change in output is explained by the change in my input.
### r^2 is independent from the number of the training points.

### Note: R2 is only bounded from below by 0 when evaluating a linear regression on its training set. If evaluated on a significantly different set, where the predictions of a regressor are worse than simply guessing the mean value for the whole set, the calculation of R2 can be negative.

# Outliers
## What causes outliers?
### Sensor Malfunction
### Data entru error
### Freak event (note that you want to pay attention to this type of outliers!
### For example, fraud ditection: you want to find those outliers!

## Outlier Detection
## Step 1: Train with all data
## Step 2: find the points in your dataset that have the highest residual error and remove them (usually about 10%).
## Step 3: Train again using the rest of the dataset.

## Identifying and cleaning away outliers is something you should always think about when looking at a dataset for the first time.

# Unsupoervised Learning
## Clustering
## Dimensionality Reduction

## Clustering

## Most-famous algorithm: K-Means
### Step 1: Assign: Randomly choose centres, and calculate the points that are closer to each centre.
### Step 2: Optimize: Minimize the "energy" of the "rubber band" within each group, and then adjust the centre. - Move the centre to the mean of all associated points.
### Step 3: Return to Step 1.

### Note that the initial position of the centroids and very important!
### Depending on the initial position of the centroids, you might end up with totally different clusters.

### Limitations of k-means:
### "Hill-Climbing": depend heavily on how you place your initial centroids.
### Could hit local-minimization instead of global one.

## sklearn
## Scalability: How the cluster behaves when you have lots of data.

# Feature Scaling
## If a feature has a larger scale, it will probably dominate the result.
## Thus, we use feature scaling to balance the weights in the features.
## Usually we rescale them to between 0 and 1.
## x'=(x-xmin)/(xmax-xmin)
## Benefit: you have a much more reliable number.
## Drawbacks: sensible to outliers.

## Putting "." after a number transformed it from integer to floating points!
## SVM and K-Means are all affected by feature rescaling.


