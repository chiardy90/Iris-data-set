<h1 align="center">Iris classifier based on Numpy</h1>

<p>First of all, we need to download the iris data set we want to implement from the following URL<p>
https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

<p>Then we import Numpy and Pandas, Pandas is used to read the dataset.<p>
<p>Pandas is used to read the data, here we use it to read the iris.data data set and process it into an array form.<p>
  
## Import
```
import numpy as np
import pandas as pd
  
df = pd.read_csv('iris.data', header=None)
print(df)
```
<img src="https://github.com/chiardy90/iris_readme_pic/blob/main/iris_1_pd%E8%AE%80%E5%8F%96%E9%99%A3%E5%88%97.png" width="50%"></p>
