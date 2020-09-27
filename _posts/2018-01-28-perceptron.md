---
title: "Analyzing Titanic Dataset In Python"
date: 2020-09-15
tags: [data analysis, data science,Python,titanic]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Titanic Project, Data Science, Python, Data Analysis"
mathjax: "true"
---




### Data Extraction and Visualization project using Python:

In this project I will try to answer some cuestions related to the titanic tragedy using Python.
First we will get our dataset from [Kaggle.com](https://www.kaggle.com/c/titanic-gettingStarted)

After importing Python liberaries such as Pandas, Numpy and seaborn we will open the dataset in Python:
```python
    titanic_df = pd.read_csv(r"C:\Users\Usuario\Desktop\Titanic Project\data.csv")
```
Let's take a look for the first 5 raws of our data:
```python
    titanic_df.head()
```





And here's some *italics*

Here's some **bold** text.

What about a [link](https://github.com/dataoptimal)?

Here's a bulleted list:
* First item
+ Second item
- Third item

Here's a numbered list:
1. First
2. Second
3. Third

Python code block:
```python
    import numpy as np

    def test_function(x, y):
      z = np.sum(x,y)
      return z
```

R code block:
```r
library(tidyverse)
df <- read_csv("some_file.csv")
head(df)
```

Here's some inline code `x+y`.

Here's an image:
<img src="{{ site.url }}{{ site.baseurl }}/images/perceptron/linsep.jpg" alt="linearly separable data">

Here's another image using Kramdown:
![alt]({{ site.url }}{{ site.baseurl }}/images/perceptron/linsep.jpg)

Here's some math:

$$z=x+y$$

You can also put it inline $$z=x+y$$
