<h1 align="center">Iris classifier based on Numpy</h1>

- First of all, we need to download the iris data set we want to implement from the following URL
https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
- Then we import Numpy and Pandas, Pandas is used to read the dataset.
- Pandas is used to read the data, here we use it to read the iris.data data set and process it into an array form.

## Import
```
import numpy as np
import pandas as pd
  
df = pd.read_csv('iris.data', header=None)
print(df)
```
<p align="center"><img src="https://github.com/chiardy90/iris_readme_pic/blob/main/Iris/iris_1.png" width="40%"></p>

- Columns 0-3 are the characteristics of the flower(sepal length, sepal width, petal length, petal width), and the fourth column is the name of the flower variety.
- The first 50 groups are **Iris-setona**, the middle 50 groups are **Iris-versicolor**, and the last 50 groups are **Iris-virginica**.</p>

## Data processing
```
x = df.iloc[0:100,[0, 1, 2, 3]].values
y = df.iloc[0:100,4].values
y = np.where(y=='Iris-setosa', 0, 1)
```
- Use **np.where** to convert data, Convert y to 0 if it meets the condition, otherwise it is 1.

<p align="center"><img src="https://github.com/chiardy90/iris_readme_pic/blob/main/Iris/iris_2.png" width="70%"></p>

## Split dataset
- Divide 100 sets of data into "Training" and "Testing".
- **Training**:For training, set the feature of the flower as x, the result is y.
- **Testing**:Test results after training.
```
x_train = np.empty((80, 4))
x_test = np.empty((20, 4))
y_train = np.empty(80)
y_test = np.empty(20)
```
- The shape **x_train** is (80,4) → Each set of samples has four features
- The shape **x_test** is (20,4) → Sample test set
- The shape **y_train** is (80,) → Training set of labels
- The shape **y_test** is (20,) → Labels test set
```
x_train[:40],x_train[40:] = x[:40],x[50:90]
x_test[:10],x_test[10:] = x[40:50],x[90:100]
y_train[:40],y_train[40:] = y[:40],y[50:90]
y_test[:10],y_test[10:] = y[40:50],y[90:100]
```
- Then use the **slicing** to copy the previous x and y data and fill in these four variables.

## Define function
```
def sigmoid(x):
    return 1/(1+np.exp(-x))
```
- sigmoid()is a common **Activation Function**,For more introduction can reference to the following URL 
https://clay-atlas.com/blog/2019/10/19/machine-learning-chinese-sigmoid-function/
<p align="center"><img src="https://github.com/chiardy90/iris_readme_pic/blob/main/Iris/iris_3.png" width="40%"></p>

```
def activation(x, w, b):
    return sigmoid(np.dot(x, w)+b)
```
- The feature data can be regarded as a vector, then the inner product of the vectors can be passed back to **sigmoid**.
- **b** is bias, can also be regarded as a weight.

```
def update(x, y_train, w, b, eta): 
    y_pred = activation(x, w, b)
    a = (y_pred - y_train) * y_pred * (1- y_pred) #partial differential
    for i in range(4): #Update 4 weights
        w[i] -= eta * 1/float(len(y)) * np.sum(a*x[:,i])
    b -= eta * 1/float(len(y))*np.sum(a)
    return w, b
```

## To train

```
weights = np.ones(4)/10 
bias = np.ones(1)/10 
eta=0.1
for _ in range(1000): 
	weights, bias = update(x_train, y_train, weights, bias, eta=0.1)
print('weights = ', weights, 'bias = ', bias)

activation(x_test, weights, bias)
```
<p align="center"><img src="https://github.com/chiardy90/iris_readme_pic/blob/main/Iris/iris_4.png" width="80%"></p>

- Here is the result of training 1000 times.
- The first 10 data are in the index 40-49, the last 10 data are in the index 50-59.
- It can be found that the first 10 results are close to 0(is Iris-setosa), and the last 10 results are closer to 1 (Iris-versicolor).






<p></p>
