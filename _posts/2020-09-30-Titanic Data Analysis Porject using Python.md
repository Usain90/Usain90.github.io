---
title: "Titanoc catastrophe data analysis using Python"
date: 2020-09-15
tags: [data analysis, data science,Python,]
header:
  image: "/images/titanic-checkpoint_files/titanic-678x381.jpg"
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



After taking a look for our data set, in a way to answer the first questions, we can notice that the column **Sex** is divided to two genders, Man and Women.
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




<img src="{{ site.url }}{{ site.baseurl }}/images/titanic/output_8_1.png" alt="linearly separable data">


As we can see there were on the Titanic:

   537 males
   271 females
   83 Children

Now let's see how they were distributed in their classes.


```python
sns.catplot("Pclass",data=titanic_df,hue="person",kind="count")
```




    <seaborn.axisgrid.FacetGrid at 0x2007b537948>




<img src="{{ site.url }}{{ site.baseurl }}/images/titanic/output_11_1.png" alt="linearly separable data">

Now let's get more precise picture of the normal distubiotion of the passengers age on the Titanc:


```python
titanic_df['Age'].hist(bins=70,color='indianred',alpha=0.9)
```








<img src="{{ site.url }}{{ site.baseurl }}/images/titanic/output_13_1.png" alt="linearly separable data">


Another way to visualize the data is to use FacetGrid to plot multiple kedplots on one plot



```python
fig = sns.FacetGrid(titanic_df, hue="person",aspect=4)
fig.map(sns.kdeplot,'Age',shade= True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()

```







<img src="{{ site.url }}{{ site.baseurl }}/images/titanic/output_15_1.png" alt="linearly separable data">


Let's do the same for class by changing the hue argument:


```python
fig = sns.FacetGrid(titanic_df, hue="Pclass",aspect=4)
fig.map(sns.kdeplot,'Age',shade= True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()
```




   




<img src="{{ site.url }}{{ site.baseurl }}/images/titanic/output_17_1.png" alt="linearly separable data">


We've gotten a pretty good picture of who the passengers were based on Sex, Age, and Class. So let's move on to our 2nd question: What deck were the passengers on and how does that relate to their class?

If we look back to our dataset, specially to a **Cabin** column, first we have to drop all the null values and creat a new object called *deck*.


```python
deck=titanic_df["Cabin"].dropna()
```

We only need the first letter of the deck column to classify its level, in order to do this we will create an empty list and loop it to grab the first letter.


```python
# Set empty list
levels = []

# Loop to grab first letter
for level in deck:
    levels.append(level[0])    

# Reset DataFrame and use factor plot
cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']
sns.catplot('Cabin',data=cabin_df,palette='winter_d',kind="count")
```







<img src="{{ site.url }}{{ site.baseurl }}/images/titanic/output_22_1.png" alt="linearly separable data">

nteresting to note we have a 'T' deck value there which doesn't make sense, we can drop it out with the following code:


```python
cabin_df = cabin_df[cabin_df.Cabin != 'T']
#Replot
sns.catplot('Cabin',data=cabin_df,palette='winter_d',kind="count")
```




   




<img src="{{ site.url }}{{ site.baseurl }}/images/titanic/output_24_1.png" alt="linearly separable data">


now that we've analyzed the distribution by decks, let's go ahead and answer our third question:

3.) Where did the passengers come from?

Note here that the Embarked column has C,Q,and S values. Reading about the project on Kaggle you'll note that these stand for Cherbourg, Queenstown, Southhampton.


```python
sns.catplot('Embarked',data=titanic_df,hue='Pclass',order=['C','Q','S'],kind="count")
```




  




<img src="{{ site.url }}{{ site.baseurl }}/images/titanic/output_26_1.png" alt="linearly separable data">


An interesting find here is that in Queenstown, almost all the passengers that boarded there were 3rd class. It would be intersting to look at the economics of that town in that time period for further investigation.

Now let's take a look at the 4th question:

4.) Who was alone and who was with family?

Let's start by adding a new column to define alone

We'll add the parent/child column with the sibsp column


```python
titanic_df['Alone'] =  titanic_df.Parch + titanic_df.SibSp
```

Now we know that if the Alone column is anything but 0, then the passenger had family aboard and wasn't alone. So let's change the column now so that if the value is greater than 0, we know the passenger was with his/her family, otherwise they were alone.


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
      <th>person</th>
      <th>Alone</th>
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
      <td>male</td>
      <td>With Family</td>
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
      <td>female</td>
      <td>With Family</td>
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
      <td>female</td>
      <td>Alone</td>
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
      <td>female</td>
      <td>With Family</td>
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
      <td>male</td>
      <td>Alone</td>
    </tr>
  </tbody>
</table>
</div>




```python
def alone(passenger):
    sib,parch = passenger
    if sib ==0 and parch==0:
        return "alone"
    else:
        return "family"
titanic_df['came_with'] = titanic_df[["SibSp","Parch"]].apply(alone,axis=1)
```


```python
sns.catplot('came_with',data=titanic_df,kind='count',order=(["alone","family"]))
```









<img src="{{ site.url }}{{ site.baseurl }}/images/titanic/output_32_1.png" alt="linearly separable data">


Great work! Now that we've throughly analyzed the data let's go ahead and take a look at the most interesting (and open-ended) question: What factors helped someone survive the sinking?


```python
# Let's start by creating a new column for legibility purposes through mapping (Lec 36)
titanic_df["Survivor"] = titanic_df.Survived.map({0: "no", 1: "yes"})

# Let's just get a quick overall view of survied vs died.
sns.catplot('Survivor',data=titanic_df,palette='Set1',kind="count")
```




    



<img src="{{ site.url }}{{ site.baseurl }}/images/titanic/output_34_1.png" alt="linearly separable data">


So quite a few more people died than those who survived. Let's see if the class of the passengers had an effect on their survival rate, since the movie Titanic popularized the notion that the 3rd class passengers did not do as well as their 1st and 2nd class counterparts.


```python
# Let's use a factor plot again, but now considering class
sns.catplot('Pclass','Survived',data=titanic_df,kind="point")
```




    




<img src="{{ site.url }}{{ site.baseurl }}/images/titanic/output_36_1.png" alt="linearly separable data">


Look like survival rates for the 3rd class are substantially lower! But maybe this effect is being caused by the large amount of men in the 3rd class in combination with the women and children first policy. Let's use 'hue' to get a clearer picture on this.


```python
# Let's use a factor plot again, but now considering class and gender
sns.catplot('Pclass','Survived',hue='person',data=titanic_df,kind="point")
```








<img src="{{ site.url }}{{ site.baseurl }}/images/titanic/output_38_1.png" alt="linearly separable data">


From this data it looks like being a male or being in 3rd class were both not favourable for survival. Even regardless of class the result of being a male in any class dramatically decreases your chances of survival.

But what about age? Did being younger or older have an effect on survival rate?


```python
# Let's use a linear plot on age versus survival
sns.lmplot('Age','Survived',data=titanic_df)

```




    




<img src="{{ site.url }}{{ site.baseurl }}/images/titanic/output_40_1.png" alt="linearly separable data">


Looks like there is a general trend that the older the passenger was, the less likely they survived. Let's go ahead and use hue to take a look at the effect of class and age.


```python
# Let's use a linear plot on age versus survival using hue for class seperation
sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='winter')
```




   




<img src="{{ site.url }}{{ site.baseurl }}/images/titanic/output_42_1.png" alt="linearly separable data">


We can also use the x_bin argument to clean up this figure and grab the data and bin it by age with a std attached!


```python
# Let's use a linear plot on age versus survival using hue for class seperation
generations=[10,20,40,60,80]
sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='winter',x_bins=generations)
```







<img src="{{ site.url }}{{ site.baseurl }}/images/titanic/output_44_1.png" alt="linearly separable data">


Interesting find on the older 1st class passengers! What about if we relate gender and age with the survival set?


```python
sns.lmplot('Age','Survived',hue='Sex',data=titanic_df,palette='winter',x_bins=generations)
```




    



<img src="{{ site.url }}{{ site.baseurl }}/images/titanic/output_46_1.png" alt="linearly separable data">




