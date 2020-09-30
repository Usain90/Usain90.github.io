---
title: "Analyzing Titanic Dataset In Python"
date: 2020-09-30
tags: [data analysis, data science,Python,titanic]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Titanic Project, Data Science, Python, Data Analysis"
mathjax: "true"
---

### Data Extraction and Visualization project using Python:
In this project I will try to answer some basics questions related to the titanic tragedy using Python.

+ Who were the passengers on the Titanic? (Ages,Gender,Class,..etc)
+ What deck were the passengers on and how does that relate to their class?
+ Where did the passengers come from?
+ Who was alone and who was with family?
+ What factors helped someone survive the sinking?

First we will get our dataset from [Kaggle.com](https://www.kaggle.com/c/titanic-gettingStarted)

After importing Python libraries such as Pandas, Numpy and seaborn we will open the dataset in Python and set it up as a Data Frame:    


```python
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
titanic_df = pd.read_csv(r"C:\Users\Usuario\Desktop\Titanic Project\data.csv")
```

Let's take a look for our data:


```python
titanic_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



After taking a look for our data set, in the way to answer the first questions, we could notice that the column **Sex** is divided to two genders, Man and Women. 
But for better analyzing we will add another gender (Child) asuming that every person is under 16 years old is a child.


```python
def male_female_child(passenger):
    # Take the Age and Sex
    age,sex = passenger
    # Compare the age, otherwise leave the sex
    if age < 16:
        return 'child'
    else:
        return sex
# We'll define a new column called 'person', remember to specify axis=1 for columns and not index
titanic_df['person'] = titanic_df[['Age','Sex']].apply(male_female_child,axis=1)    
```

Now after we created another column called *person*, let's visualize our data.


```python
sns.catplot("person",data=titanic_df,kind="count")
titanic_df["person"].value_counts()
```




    male      537
    female    271
    child      83
    Name: person, dtype: int64




![png](titanic_11-checkpoint_files/titanic_11-checkpoint_8_1.png)


As we can see there were on the Titanic:

    537 males
    271 females
    83 Children

Now let's see how they were distributed in their classes.


```python
sns.catplot("Pclass",data=titanic_df,hue="person",kind="count")
```




    <seaborn.axisgrid.FacetGrid at 0x2007b537948>




![png](titanic_11-checkpoint_files/titanic_11-checkpoint_11_1.png)


Now let's take a  more precise picture of the passengers age's normal distubiotions 
