### Stock Market Analysis:

In this portfolio project we will be looking at data from the stock market, particularly some technology stocks. We will use *pandas* to get stock information, visualize different aspects of it, and finally we will look at a few ways of analyzing the risk of a stock, based on its previous performance history. We will also be predicting future stock prices through a **Monte Carlo method!**

We'll be answering the following questions along the way:

+ What was the change in price of the stock over time?
+ What was the daily return of the stock on average?
- What was the moving average of the various stocks?
- What was the correlation between different stocks' closing prices?
- What was the correlation between different stocks' daily returns?
- How much value do we put at risk by investing in a particular stock?
- How can we attempt to predict future stock behavior?


```python
from __future__ import division
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style("whitegrid")
import pandas_datareader as pdr
from datetime import datetime
```


```python
tech_list = ["AAPL","GOOG","MSFT","AMZN","TSLA"]
```


```python
end = datetime.now()
start = datetime(end.year-1,end.month,end.day)
```


```python
for stock in tech_list:
    globals()[stock]=pdr.DataReader(stock,"yahoo",start,end)
```


```python
TSLA
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
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-10-01</th>
      <td>49.189999</td>
      <td>47.826000</td>
      <td>48.299999</td>
      <td>48.938000</td>
      <td>30813000.0</td>
      <td>48.938000</td>
    </tr>
    <tr>
      <th>2019-10-02</th>
      <td>48.930000</td>
      <td>47.886002</td>
      <td>48.658001</td>
      <td>48.625999</td>
      <td>28157000.0</td>
      <td>48.625999</td>
    </tr>
    <tr>
      <th>2019-10-03</th>
      <td>46.896000</td>
      <td>44.855999</td>
      <td>46.372002</td>
      <td>46.605999</td>
      <td>75422500.0</td>
      <td>46.605999</td>
    </tr>
    <tr>
      <th>2019-10-04</th>
      <td>46.956001</td>
      <td>45.613998</td>
      <td>46.321999</td>
      <td>46.285999</td>
      <td>39975000.0</td>
      <td>46.285999</td>
    </tr>
    <tr>
      <th>2019-10-07</th>
      <td>47.712002</td>
      <td>45.709999</td>
      <td>45.959999</td>
      <td>47.543999</td>
      <td>40321000.0</td>
      <td>47.543999</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-09-25</th>
      <td>408.730011</td>
      <td>391.299988</td>
      <td>393.470001</td>
      <td>407.339996</td>
      <td>67208500.0</td>
      <td>407.339996</td>
    </tr>
    <tr>
      <th>2020-09-28</th>
      <td>428.079987</td>
      <td>415.549988</td>
      <td>424.619995</td>
      <td>421.200012</td>
      <td>49719600.0</td>
      <td>421.200012</td>
    </tr>
    <tr>
      <th>2020-09-29</th>
      <td>428.500000</td>
      <td>411.600006</td>
      <td>416.000000</td>
      <td>419.070007</td>
      <td>50219300.0</td>
      <td>419.070007</td>
    </tr>
    <tr>
      <th>2020-09-30</th>
      <td>433.929993</td>
      <td>420.470001</td>
      <td>421.320007</td>
      <td>429.010010</td>
      <td>48145600.0</td>
      <td>429.010010</td>
    </tr>
    <tr>
      <th>2020-10-01</th>
      <td>448.880005</td>
      <td>434.420013</td>
      <td>440.760010</td>
      <td>448.160004</td>
      <td>50413600.0</td>
      <td>448.160004</td>
    </tr>
  </tbody>
</table>
<p>254 rows × 6 columns</p>
</div>




```python
GOOG.head()
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
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-10-01</th>
      <td>1231.229980</td>
      <td>1203.579956</td>
      <td>1219.000000</td>
      <td>1205.099976</td>
      <td>1273500</td>
      <td>1205.099976</td>
    </tr>
    <tr>
      <th>2019-10-02</th>
      <td>1196.979980</td>
      <td>1171.290039</td>
      <td>1196.979980</td>
      <td>1176.630005</td>
      <td>1615100</td>
      <td>1176.630005</td>
    </tr>
    <tr>
      <th>2019-10-03</th>
      <td>1189.060059</td>
      <td>1162.430054</td>
      <td>1180.000000</td>
      <td>1187.829956</td>
      <td>1621200</td>
      <td>1187.829956</td>
    </tr>
    <tr>
      <th>2019-10-04</th>
      <td>1211.439941</td>
      <td>1189.170044</td>
      <td>1191.890015</td>
      <td>1209.000000</td>
      <td>1162400</td>
      <td>1209.000000</td>
    </tr>
    <tr>
      <th>2019-10-07</th>
      <td>1218.203979</td>
      <td>1203.750000</td>
      <td>1204.400024</td>
      <td>1207.680054</td>
      <td>842900</td>
      <td>1207.680054</td>
    </tr>
  </tbody>
</table>
</div>




```python
TSLA.describe()
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
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>254.000000</td>
      <td>254.000000</td>
      <td>254.000000</td>
      <td>254.000000</td>
      <td>2.540000e+02</td>
      <td>254.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>183.048851</td>
      <td>171.928733</td>
      <td>177.476189</td>
      <td>178.020858</td>
      <td>6.828537e+07</td>
      <td>178.020858</td>
    </tr>
    <tr>
      <th>std</th>
      <td>120.733026</td>
      <td>111.788912</td>
      <td>116.497418</td>
      <td>117.023693</td>
      <td>4.152941e+07</td>
      <td>117.023693</td>
    </tr>
    <tr>
      <th>min</th>
      <td>46.896000</td>
      <td>44.855999</td>
      <td>45.959999</td>
      <td>46.285999</td>
      <td>1.085270e+07</td>
      <td>46.285999</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>86.278999</td>
      <td>82.025000</td>
      <td>85.070501</td>
      <td>85.511497</td>
      <td>3.867288e+07</td>
      <td>85.511497</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>154.772003</td>
      <td>142.832001</td>
      <td>147.945000</td>
      <td>149.703003</td>
      <td>6.310600e+07</td>
      <td>149.703003</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>240.966503</td>
      <td>231.864998</td>
      <td>237.372005</td>
      <td>237.280495</td>
      <td>8.740808e+07</td>
      <td>237.280495</td>
    </tr>
    <tr>
      <th>max</th>
      <td>502.489990</td>
      <td>470.510010</td>
      <td>502.140015</td>
      <td>498.320007</td>
      <td>3.046940e+08</td>
      <td>498.320007</td>
    </tr>
  </tbody>
</table>
</div>




```python
TSLA.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 254 entries, 2019-10-01 to 2020-10-01
    Data columns (total 6 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   High       254 non-null    float64
     1   Low        254 non-null    float64
     2   Open       254 non-null    float64
     3   Close      254 non-null    float64
     4   Volume     254 non-null    float64
     5   Adj Close  254 non-null    float64
    dtypes: float64(6)
    memory usage: 13.9 KB
    


```python
TSLA["Adj Close"].plot(legend=True, figsize=(10,4))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1ced31a6888>




![png](output_9_1.png)



```python
AAPL["Adj Close"].plot(legend=True, figsize=(10,4))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1ced3168c08>




![png](output_10_1.png)



```python
MSFT["Adj Close"].plot(legend=True, figsize=(10,4))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1ced30c4748>




![png](output_11_1.png)



```python
GOOG["Adj Close"].plot(legend=True, figsize=(10,4))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1ced300d988>




![png](output_12_1.png)



```python
AMZN["Adj Close"].plot(legend=True, figsize=(10,4))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1ced32efe48>




![png](output_13_1.png)



```python
TSLA["Volume"].plot(legend=True, figsize=(10,4))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1ced3388948>




![png](output_14_1.png)



```python
ma_day =[10,20,50]
```


```python
for ma in ma_day:
    column_name = "MA for %s days" %(str(ma))
    AAPL[column_name]=AAPL["Adj Close"].rolling(ma).mean()
```


```python
AAPL[["Adj Close","MA for 10 days","MA for 20 days","MA for 50 days"]].plot(subplots=False,figsize=(10,4))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1ced3647f08>




![png](output_17_1.png)


### Section 2 - Daily Return Analysis

We're now going to analyze the risk of the stock. In order to do so we'll need to take a closer look at the daily changes of the stock, and not just its absolute value. Let's go ahead and use pandas to retrieve teh daily returns for the *Apple* stock.


```python
AAPL["Daily Return"] = AAPL["Adj Close"].pct_change()
```


```python
AAPL["Daily Return"].plot(figsize=(10,4),legend=True,linestyle="--",marker="o")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1ced397ebc8>




![png](output_20_1.png)



```python
sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color='purple')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1ced39f01c8>




![png](output_21_1.png)



```python
closing_df=pdr.DataReader(tech_list,"yahoo",start,end)["Adj Close"]
```


```python
closing_df
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
      <th>Symbols</th>
      <th>AAPL</th>
      <th>GOOG</th>
      <th>MSFT</th>
      <th>AMZN</th>
      <th>TSLA</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-10-01</th>
      <td>55.595886</td>
      <td>1205.099976</td>
      <td>135.527100</td>
      <td>1735.650024</td>
      <td>48.938000</td>
    </tr>
    <tr>
      <th>2019-10-02</th>
      <td>54.202213</td>
      <td>1176.630005</td>
      <td>133.134323</td>
      <td>1713.229980</td>
      <td>48.625999</td>
    </tr>
    <tr>
      <th>2019-10-03</th>
      <td>54.662643</td>
      <td>1187.829956</td>
      <td>134.745972</td>
      <td>1724.420044</td>
      <td>46.605999</td>
    </tr>
    <tr>
      <th>2019-10-04</th>
      <td>56.194942</td>
      <td>1209.000000</td>
      <td>136.565262</td>
      <td>1739.650024</td>
      <td>46.285999</td>
    </tr>
    <tr>
      <th>2019-10-07</th>
      <td>56.207317</td>
      <td>1207.680054</td>
      <td>135.576523</td>
      <td>1732.660034</td>
      <td>47.543999</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-09-25</th>
      <td>112.279999</td>
      <td>1444.959961</td>
      <td>207.820007</td>
      <td>3095.129883</td>
      <td>407.339996</td>
    </tr>
    <tr>
      <th>2020-09-28</th>
      <td>114.959999</td>
      <td>1464.520020</td>
      <td>209.440002</td>
      <td>3174.050049</td>
      <td>421.200012</td>
    </tr>
    <tr>
      <th>2020-09-29</th>
      <td>114.089996</td>
      <td>1469.329956</td>
      <td>207.259995</td>
      <td>3144.879883</td>
      <td>419.070007</td>
    </tr>
    <tr>
      <th>2020-09-30</th>
      <td>115.809998</td>
      <td>1469.599976</td>
      <td>210.330002</td>
      <td>3148.729980</td>
      <td>429.010010</td>
    </tr>
    <tr>
      <th>2020-10-01</th>
      <td>116.790001</td>
      <td>1490.089966</td>
      <td>212.460007</td>
      <td>3221.260010</td>
      <td>448.160004</td>
    </tr>
  </tbody>
</table>
<p>254 rows × 5 columns</p>
</div>




```python
tech_rets=closing_df.pct_change()
```


```python
tech_rets
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
      <th>Symbols</th>
      <th>AAPL</th>
      <th>GOOG</th>
      <th>MSFT</th>
      <th>AMZN</th>
      <th>TSLA</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-10-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2019-10-02</th>
      <td>-0.025068</td>
      <td>-0.023625</td>
      <td>-0.017655</td>
      <td>-0.012917</td>
      <td>-0.006375</td>
    </tr>
    <tr>
      <th>2019-10-03</th>
      <td>0.008495</td>
      <td>0.009519</td>
      <td>0.012105</td>
      <td>0.006532</td>
      <td>-0.041542</td>
    </tr>
    <tr>
      <th>2019-10-04</th>
      <td>0.028032</td>
      <td>0.017822</td>
      <td>0.013502</td>
      <td>0.008832</td>
      <td>-0.006866</td>
    </tr>
    <tr>
      <th>2019-10-07</th>
      <td>0.000220</td>
      <td>-0.001092</td>
      <td>-0.007240</td>
      <td>-0.004018</td>
      <td>0.027179</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-09-25</th>
      <td>0.037516</td>
      <td>0.011671</td>
      <td>0.022787</td>
      <td>0.024949</td>
      <td>0.050414</td>
    </tr>
    <tr>
      <th>2020-09-28</th>
      <td>0.023869</td>
      <td>0.013537</td>
      <td>0.007795</td>
      <td>0.025498</td>
      <td>0.034026</td>
    </tr>
    <tr>
      <th>2020-09-29</th>
      <td>-0.007568</td>
      <td>0.003284</td>
      <td>-0.010409</td>
      <td>-0.009190</td>
      <td>-0.005057</td>
    </tr>
    <tr>
      <th>2020-09-30</th>
      <td>0.015076</td>
      <td>0.000184</td>
      <td>0.014812</td>
      <td>0.001224</td>
      <td>0.023719</td>
    </tr>
    <tr>
      <th>2020-10-01</th>
      <td>0.008462</td>
      <td>0.013943</td>
      <td>0.010127</td>
      <td>0.023035</td>
      <td>0.044638</td>
    </tr>
  </tbody>
</table>
<p>254 rows × 5 columns</p>
</div>




```python
sns.jointplot("GOOG","GOOG",tech_rets, kind="scatter",color="seagreen")
```




    <seaborn.axisgrid.JointGrid at 0x1ced4cf8448>




![png](output_26_1.png)



```python
sns.jointplot("GOOG","TSLA",tech_rets, kind="scatter",color="seagreen")
```




    <seaborn.axisgrid.JointGrid at 0x1ced5083d48>




![png](output_27_1.png)



```python
sns.pairplot(tech_rets.dropna())
```




    <seaborn.axisgrid.PairGrid at 0x1ced5e57a88>




![png](output_28_1.png)



```python

```
