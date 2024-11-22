```python
import pandas as pd

import numpy as np

from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set_style('whitegrid')

import plotly.express as px

```


```python
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
```

**DATA READING**


```python
df = pd.read_csv("DATA.csv")
 
```


```python
df.shape
```




    (64016, 13)




```python
df.head()
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
      <th>title</th>
      <th>console</th>
      <th>genre</th>
      <th>publisher</th>
      <th>developer</th>
      <th>critic_score</th>
      <th>total_sales</th>
      <th>na_sales</th>
      <th>jp_sales</th>
      <th>pal_sales</th>
      <th>other_sales</th>
      <th>release_date</th>
      <th>last_update</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Grand Theft Auto V</td>
      <td>PS3</td>
      <td>Action</td>
      <td>Rockstar Games</td>
      <td>Rockstar North</td>
      <td>9.4</td>
      <td>20.32</td>
      <td>6.37</td>
      <td>0.99</td>
      <td>9.85</td>
      <td>3.12</td>
      <td>17-09-13</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Grand Theft Auto V</td>
      <td>PS4</td>
      <td>Action</td>
      <td>Rockstar Games</td>
      <td>Rockstar North</td>
      <td>9.7</td>
      <td>19.39</td>
      <td>6.06</td>
      <td>0.60</td>
      <td>9.71</td>
      <td>3.02</td>
      <td>18-11-14</td>
      <td>03-01-18</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Grand Theft Auto: Vice City</td>
      <td>PS2</td>
      <td>Action</td>
      <td>Rockstar Games</td>
      <td>Rockstar North</td>
      <td>9.6</td>
      <td>16.15</td>
      <td>8.41</td>
      <td>0.47</td>
      <td>5.49</td>
      <td>1.78</td>
      <td>28-10-02</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Grand Theft Auto V</td>
      <td>X360</td>
      <td>Action</td>
      <td>Rockstar Games</td>
      <td>Rockstar North</td>
      <td>NaN</td>
      <td>15.86</td>
      <td>9.06</td>
      <td>0.06</td>
      <td>5.33</td>
      <td>1.42</td>
      <td>17-09-13</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Call of Duty: Black Ops 3</td>
      <td>PS4</td>
      <td>Shooter</td>
      <td>Activision</td>
      <td>Treyarch</td>
      <td>8.1</td>
      <td>15.09</td>
      <td>6.18</td>
      <td>0.41</td>
      <td>6.05</td>
      <td>2.44</td>
      <td>06-11-15</td>
      <td>14-01-18</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
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
      <th>critic_score</th>
      <th>total_sales</th>
      <th>na_sales</th>
      <th>jp_sales</th>
      <th>pal_sales</th>
      <th>other_sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6678.000000</td>
      <td>18922.000000</td>
      <td>12637.000000</td>
      <td>6726.000000</td>
      <td>12824.000000</td>
      <td>15128.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7.220440</td>
      <td>0.349113</td>
      <td>0.264740</td>
      <td>0.102281</td>
      <td>0.149472</td>
      <td>0.043041</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.457066</td>
      <td>0.807462</td>
      <td>0.494787</td>
      <td>0.168811</td>
      <td>0.392653</td>
      <td>0.126643</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>6.400000</td>
      <td>0.030000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>0.010000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.500000</td>
      <td>0.120000</td>
      <td>0.120000</td>
      <td>0.040000</td>
      <td>0.040000</td>
      <td>0.010000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.300000</td>
      <td>0.340000</td>
      <td>0.280000</td>
      <td>0.120000</td>
      <td>0.140000</td>
      <td>0.030000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>10.000000</td>
      <td>20.320000</td>
      <td>9.760000</td>
      <td>2.130000</td>
      <td>9.850000</td>
      <td>3.120000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 64016 entries, 0 to 64015
    Data columns (total 14 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   img           64016 non-null  object 
     1   title         64016 non-null  object 
     2   console       64016 non-null  object 
     3   genre         64016 non-null  object 
     4   publisher     64016 non-null  object 
     5   developer     63999 non-null  object 
     6   critic_score  6678 non-null   float64
     7   total_sales   18922 non-null  float64
     8   na_sales      12637 non-null  float64
     9   jp_sales      6726 non-null   float64
     10  pal_sales     12824 non-null  float64
     11  other_sales   15128 non-null  float64
     12  release_date  56965 non-null  object 
     13  last_update   17879 non-null  object 
    dtypes: float64(6), object(8)
    memory usage: 6.8+ MB
    

**DATA CLEANING**

FINDING TOTAL NULL VALUES PER COLUMN & THE % OF NULL VALUES IN THE DATA 


```python
df.isnull().sum()
```




    img                 0
    title               0
    console             0
    genre               0
    publisher           0
    developer          17
    critic_score    57338
    total_sales     45094
    na_sales        51379
    jp_sales        57290
    pal_sales       51192
    other_sales     48888
    release_date     7051
    last_update     46137
    dtype: int64




```python
missing = df.isnull().sum().sum()
total = np.product(df.shape)

percent_missing = missing/total*100

print(percent_missing)
```

    40.65791587817331
    

40% OF OUR DATA IS MISSING - TOO BIG - HENCE WE CANNOT JUST DROP THE NULL VALUE ROWS. WE HAVE TO FILL IT - WITH IMPUTATION


```python
# DROP USELESS COLUMN 'IMG' WHICH CONTAINS LINKS OF IMAGES OF GAMES - NOT REQUIRED FOR OUR ANALYSIS

df.drop(columns = ['img'], inplace = True)
```


```python
df.columns
```




    Index(['title', 'console', 'genre', 'publisher', 'developer', 'critic_score',
           'total_sales', 'na_sales', 'jp_sales', 'pal_sales', 'other_sales',
           'release_date', 'last_update'],
          dtype='object')



HANDLING NULL VALUES


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 64016 entries, 0 to 64015
    Data columns (total 13 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   title         64016 non-null  object 
     1   console       64016 non-null  object 
     2   genre         64016 non-null  object 
     3   publisher     64016 non-null  object 
     4   developer     63999 non-null  object 
     5   critic_score  6678 non-null   float64
     6   total_sales   18922 non-null  float64
     7   na_sales      12637 non-null  float64
     8   jp_sales      6726 non-null   float64
     9   pal_sales     12824 non-null  float64
     10  other_sales   15128 non-null  float64
     11  release_date  56965 non-null  object 
     12  last_update   17879 non-null  object 
    dtypes: float64(6), object(7)
    memory usage: 6.3+ MB
    


```python
df.isnull().sum()
```




    title               0
    console             0
    genre               0
    publisher           0
    developer          17
    critic_score    57338
    total_sales     45094
    na_sales        51379
    jp_sales        57290
    pal_sales       51192
    other_sales     48888
    release_date     7051
    last_update     46137
    dtype: int64



WE HAVE 2 DATATYPES OF NULL VALUES TO FILL - FLOAT64 (NUMERICAL) AND DATETIME (NEED TO CHANGE FROM OBJECT TYPE)

HANDLING NUMERICAL NULL VALUES


```python
num_cols = df.select_dtypes(include = np.number).columns.tolist()
num_cols
```




    ['critic_score',
     'total_sales',
     'na_sales',
     'jp_sales',
     'pal_sales',
     'other_sales']




```python
# MEAN IMPUTATION FOR FILLING NULL VALUES 

imputer = SimpleImputer(strategy = 'median')
df[num_cols] = imputer.fit_transform(df[num_cols])
```


```python
df.isnull().sum()
```




    title               0
    console             0
    genre               0
    publisher           0
    developer          17
    critic_score        0
    total_sales         0
    na_sales            0
    jp_sales            0
    pal_sales           0
    other_sales         0
    release_date     7051
    last_update     46137
    dtype: int64



NUMERICAL DATATYPE NULL VALUES HAVE BEEN FILLED USING MEAN IMPUTATION


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 64016 entries, 0 to 64015
    Data columns (total 13 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   title         64016 non-null  object 
     1   console       64016 non-null  object 
     2   genre         64016 non-null  object 
     3   publisher     64016 non-null  object 
     4   developer     63999 non-null  object 
     5   critic_score  64016 non-null  float64
     6   total_sales   64016 non-null  float64
     7   na_sales      64016 non-null  float64
     8   jp_sales      64016 non-null  float64
     9   pal_sales     64016 non-null  float64
     10  other_sales   64016 non-null  float64
     11  release_date  56965 non-null  object 
     12  last_update   17879 non-null  object 
    dtypes: float64(6), object(7)
    memory usage: 6.3+ MB
    

HANDLING DATETIME NULL VALUES - LAST 2 COLUMNS


```python
# CONVERTING DATATYPE FROM OBJECT TO DATETIME

df['release_date'] = pd.to_datetime(df['release_date'])
df['last_update'] = pd.to_datetime(df['last_update'])

df.info()
```

    C:\Users\asus\AppData\Local\Temp\ipykernel_22864\3548395132.py:3: UserWarning:
    
    Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
    
    C:\Users\asus\AppData\Local\Temp\ipykernel_22864\3548395132.py:4: UserWarning:
    
    Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
    
    

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 64016 entries, 0 to 64015
    Data columns (total 13 columns):
     #   Column        Non-Null Count  Dtype         
    ---  ------        --------------  -----         
     0   title         64016 non-null  object        
     1   console       64016 non-null  object        
     2   genre         64016 non-null  object        
     3   publisher     64016 non-null  object        
     4   developer     63999 non-null  object        
     5   critic_score  64016 non-null  float64       
     6   total_sales   64016 non-null  float64       
     7   na_sales      64016 non-null  float64       
     8   jp_sales      64016 non-null  float64       
     9   pal_sales     64016 non-null  float64       
     10  other_sales   64016 non-null  float64       
     11  release_date  56965 non-null  datetime64[ns]
     12  last_update   17879 non-null  datetime64[ns]
    dtypes: datetime64[ns](2), float64(6), object(5)
    memory usage: 6.3+ MB
    


```python
# FILLNA() WITH MEDIAN TO FILL DATETIME NULL VALUES

df['release_date'] = df['release_date'].fillna(df['release_date'].median())
df['last_update'] = df['last_update'].fillna(df['last_update'].median())

df.isnull().sum()
```




    title            0
    console          0
    genre            0
    publisher        0
    developer       17
    critic_score     0
    total_sales      0
    na_sales         0
    jp_sales         0
    pal_sales        0
    other_sales      0
    release_date     0
    last_update      0
    dtype: int64



BOTH NUMERICAL AND DATETIME NULL VALUES HAVE BEEN FILLED.


```python
# 17 ROWS OF DEVELOPER COLUMN IS WHAT % OF THE TOTAL DATA ?

missing = df.isnull().sum().sum()
total = np.product(df.shape)

percent_missing = missing/total*100

print(percent_missing)
```

    0.0020427585411339475
    

NOW TO HANDLE OBJECT TYPE NULL VALUES IN 'DEVELOPER' COLUMN - BY DROPPING - SINCE IT'S ONLY 0.002% OF TOTAL DATA


```python
df.dropna(inplace = True)

df.isnull().sum()
```




    title           0
    console         0
    genre           0
    publisher       0
    developer       0
    critic_score    0
    total_sales     0
    na_sales        0
    jp_sales        0
    pal_sales       0
    other_sales     0
    release_date    0
    last_update     0
    dtype: int64



ALL NULL VALUES HAVE BEEN HANDLED.

NOW OUR DATA IS READY FOR THE EDA.

**EDA (EXPLORATORY DATA ANALYSIS)**


```python
df.head()
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
      <th>title</th>
      <th>console</th>
      <th>genre</th>
      <th>publisher</th>
      <th>developer</th>
      <th>critic_score</th>
      <th>total_sales</th>
      <th>na_sales</th>
      <th>jp_sales</th>
      <th>pal_sales</th>
      <th>other_sales</th>
      <th>release_date</th>
      <th>last_update</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Grand Theft Auto V</td>
      <td>PS3</td>
      <td>Action</td>
      <td>Rockstar Games</td>
      <td>Rockstar North</td>
      <td>9.40000</td>
      <td>20.32</td>
      <td>6.37</td>
      <td>0.99</td>
      <td>9.85</td>
      <td>3.12</td>
      <td>2013-09-17</td>
      <td>2019-05-06</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Grand Theft Auto V</td>
      <td>PS4</td>
      <td>Action</td>
      <td>Rockstar Games</td>
      <td>Rockstar North</td>
      <td>9.70000</td>
      <td>19.39</td>
      <td>6.06</td>
      <td>0.60</td>
      <td>9.71</td>
      <td>3.02</td>
      <td>2014-11-18</td>
      <td>2018-03-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Grand Theft Auto: Vice City</td>
      <td>PS2</td>
      <td>Action</td>
      <td>Rockstar Games</td>
      <td>Rockstar North</td>
      <td>9.60000</td>
      <td>16.15</td>
      <td>8.41</td>
      <td>0.47</td>
      <td>5.49</td>
      <td>1.78</td>
      <td>2002-10-28</td>
      <td>2019-05-06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Grand Theft Auto V</td>
      <td>X360</td>
      <td>Action</td>
      <td>Rockstar Games</td>
      <td>Rockstar North</td>
      <td>7.22044</td>
      <td>15.86</td>
      <td>9.06</td>
      <td>0.06</td>
      <td>5.33</td>
      <td>1.42</td>
      <td>2013-09-17</td>
      <td>2019-05-06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Call of Duty: Black Ops 3</td>
      <td>PS4</td>
      <td>Shooter</td>
      <td>Activision</td>
      <td>Treyarch</td>
      <td>8.10000</td>
      <td>15.09</td>
      <td>6.18</td>
      <td>0.41</td>
      <td>6.05</td>
      <td>2.44</td>
      <td>2015-06-11</td>
      <td>2018-01-14</td>
    </tr>
  </tbody>
</table>
</div>



1. Identify top-selling titles worldwide and analyze key success factors.


```python
top_selling_games = df.groupby('title').total_sales.sum().sort_values(ascending = False).reset_index()

top10_titles = top_selling_games.head(10)

top10_titles
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
      <th>title</th>
      <th>total_sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Grand Theft Auto V</td>
      <td>65.686451</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Call of Duty: Black Ops</td>
      <td>32.037338</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Call of Duty: Modern Warfare 3</td>
      <td>31.059113</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Call of Duty: Black Ops II</td>
      <td>29.939113</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Call of Duty: Ghosts</td>
      <td>29.149113</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Call of Duty: Black Ops 3</td>
      <td>26.720000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Minecraft</td>
      <td>26.104676</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Call of Duty: Modern Warfare 2</td>
      <td>26.067338</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Grand Theft Auto IV</td>
      <td>23.228225</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Call of Duty: Advanced Warfare</td>
      <td>21.780000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize = (12,8))
sns.barplot(data = top10_titles, x = 'title', y = 'total_sales', palette='viridis')
plt.title("Top 10 Best-Selling Games Worldwide")
plt.xlabel("Game Title")
plt.ylabel("Global Sales")
plt.xticks(rotation = 90)
plt.show()
```

    C:\Users\asus\AppData\Local\Temp\ipykernel_22864\4147359019.py:2: FutureWarning:
    
    
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
    
    


    
![png](output_37_1.png)
    


GRAND THEFT AUTO V IS THE HIGHEST SELLING TITLE WORLDWIDE

2. Examine sales trends over time to capture industry growth or decline.


```python
# EXTRACTING RELEASE YEAR FOR GROUPBY FUNCTION

df['release_year'] = pd.to_datetime(df['release_date']).dt.year

df.head()
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
      <th>title</th>
      <th>console</th>
      <th>genre</th>
      <th>publisher</th>
      <th>developer</th>
      <th>critic_score</th>
      <th>total_sales</th>
      <th>na_sales</th>
      <th>jp_sales</th>
      <th>pal_sales</th>
      <th>other_sales</th>
      <th>release_date</th>
      <th>last_update</th>
      <th>release_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Grand Theft Auto V</td>
      <td>PS3</td>
      <td>Action</td>
      <td>Rockstar Games</td>
      <td>Rockstar North</td>
      <td>9.40000</td>
      <td>20.32</td>
      <td>6.37</td>
      <td>0.99</td>
      <td>9.85</td>
      <td>3.12</td>
      <td>2013-09-17</td>
      <td>2019-05-06</td>
      <td>2013</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Grand Theft Auto V</td>
      <td>PS4</td>
      <td>Action</td>
      <td>Rockstar Games</td>
      <td>Rockstar North</td>
      <td>9.70000</td>
      <td>19.39</td>
      <td>6.06</td>
      <td>0.60</td>
      <td>9.71</td>
      <td>3.02</td>
      <td>2014-11-18</td>
      <td>2018-03-01</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Grand Theft Auto: Vice City</td>
      <td>PS2</td>
      <td>Action</td>
      <td>Rockstar Games</td>
      <td>Rockstar North</td>
      <td>9.60000</td>
      <td>16.15</td>
      <td>8.41</td>
      <td>0.47</td>
      <td>5.49</td>
      <td>1.78</td>
      <td>2002-10-28</td>
      <td>2019-05-06</td>
      <td>2002</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Grand Theft Auto V</td>
      <td>X360</td>
      <td>Action</td>
      <td>Rockstar Games</td>
      <td>Rockstar North</td>
      <td>7.22044</td>
      <td>15.86</td>
      <td>9.06</td>
      <td>0.06</td>
      <td>5.33</td>
      <td>1.42</td>
      <td>2013-09-17</td>
      <td>2019-05-06</td>
      <td>2013</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Call of Duty: Black Ops 3</td>
      <td>PS4</td>
      <td>Shooter</td>
      <td>Activision</td>
      <td>Treyarch</td>
      <td>8.10000</td>
      <td>15.09</td>
      <td>6.18</td>
      <td>0.41</td>
      <td>6.05</td>
      <td>2.44</td>
      <td>2015-06-11</td>
      <td>2018-01-14</td>
      <td>2015</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc[df.release_year > 2024]
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
      <th>title</th>
      <th>console</th>
      <th>genre</th>
      <th>publisher</th>
      <th>developer</th>
      <th>critic_score</th>
      <th>total_sales</th>
      <th>na_sales</th>
      <th>jp_sales</th>
      <th>pal_sales</th>
      <th>other_sales</th>
      <th>release_date</th>
      <th>last_update</th>
      <th>release_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>30650</th>
      <td>The Oregon Trail</td>
      <td>Series</td>
      <td>Education</td>
      <td>MECC</td>
      <td>MECC</td>
      <td>7.22044</td>
      <td>0.349113</td>
      <td>0.26474</td>
      <td>0.102281</td>
      <td>0.149472</td>
      <td>0.043041</td>
      <td>2071-03-12</td>
      <td>2020-02-18</td>
      <td>2071</td>
    </tr>
    <tr>
      <th>57351</th>
      <td>Trek73</td>
      <td>PC</td>
      <td>Simulation</td>
      <td>Unknown</td>
      <td>William K. Char, Perry Lee, and Dan Gee</td>
      <td>7.22044</td>
      <td>0.349113</td>
      <td>0.26474</td>
      <td>0.102281</td>
      <td>0.149472</td>
      <td>0.043041</td>
      <td>2073-08-10</td>
      <td>2019-05-06</td>
      <td>2073</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = df.drop(df[df['release_year'] == 2071].index)
df = df.drop(df[df['release_year'] == 2073].index)
```


```python
yearly_sales = df.groupby(['release_year']).total_sales.sum().sort_values(ascending = False).reset_index()

top10_years = yearly_sales.head(10)

top10_years
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
      <th>release_year</th>
      <th>total_sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2008</td>
      <td>3429.537130</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2009</td>
      <td>1405.496739</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2010</td>
      <td>1223.813444</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011</td>
      <td>1185.326444</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014</td>
      <td>1083.199317</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2007</td>
      <td>875.224630</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2013</td>
      <td>694.012137</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2002</td>
      <td>620.322702</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2003</td>
      <td>616.138744</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2006</td>
      <td>610.290081</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize = (12,10))
sns.barplot (data = top10_years, x = 'release_year', y = 'total_sales', palette = 'viridis')
plt.title("Top 10 Highest-Selling Years")
plt.xlabel("Year")
plt.ylabel("Global Sales")
plt.xticks(rotation=0)
plt.show()
```

    C:\Users\asus\AppData\Local\Temp\ipykernel_22864\1138402198.py:2: FutureWarning:
    
    
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
    
    


    
![png](output_44_1.png)
    


2008 WAS THE HIGHEST SELLING YEAR FOR GAMES, FOLLOWED BY 2009.


```python
# INDUSTRY TREND GROWTH OR DECLINE

plt.figure(figsize=(12,10))
sns.lineplot(data = yearly_sales, x='release_year', y='total_sales', marker='', color='tab:red')
plt.title("Global Video Game Sales Over Time")
plt.xlabel("Year")
plt.ylabel("Global Sales")
plt.grid(True)
plt.show()
```


    
![png](output_46_0.png)
    


THE INDUSTRY HAS SHOWN MASSIVE GROWTH SPIKE IN 2008, FOLLOWED BY A GRADUAL DECLINE.

3. Discover genre specializations across consoles.


```python
console_genre = df.groupby(['console', 'genre']).total_sales.sum().reset_index()

cg_sorted = console_genre.sort_values(by = 'total_sales', ascending = False).head(50).reset_index()

cg_sorted.head()
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
      <th>index</th>
      <th>console</th>
      <th>genre</th>
      <th>total_sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>456</td>
      <td>PC</td>
      <td>Adventure</td>
      <td>579.434784</td>
    </tr>
    <tr>
      <th>1</th>
      <td>472</td>
      <td>PC</td>
      <td>Strategy</td>
      <td>491.942109</td>
    </tr>
    <tr>
      <th>2</th>
      <td>461</td>
      <td>PC</td>
      <td>Misc</td>
      <td>466.622278</td>
    </tr>
    <tr>
      <th>3</th>
      <td>467</td>
      <td>PC</td>
      <td>Role-Playing</td>
      <td>421.048926</td>
    </tr>
    <tr>
      <th>4</th>
      <td>454</td>
      <td>PC</td>
      <td>Action</td>
      <td>406.433771</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Pivot the data for a heatmap (this will create a matrix of platforms and genres)
heatmap_data = cg_sorted.pivot(index='console', columns='genre', values='total_sales')

# Create a heatmap using Seaborn
plt.figure(figsize=(12,8))
sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='Greens', linewidths=1, cbar_kws={'label': 'Total Sales'})

plt.title("Heatmap of Total Sales by Genre and Console", fontsize=16)
plt.xlabel("Genre", fontsize=15)
plt.ylabel("Console", fontsize=15)
plt.xticks(rotation=90, ha='right')  

plt.tight_layout()
plt.show()
```


    
![png](output_50_0.png)
    


PC CONSOLE HAS THE HIGHEST SALES WITH ADVENTURE BEING THE MAIN SPECIALIZATION, FOLLOWED BY STRATEGY, ROLE-PLAYING, AND ACTION.

GENRES LIKE ACTION, SHOOTER AND SPORTS HAVE A WIDESPREAD REACH TO MULTIPLE CONSOLES. REST OF THE GENRES ARE CONSOLE SPECIFIC.

4. Analyze regional popularity to spot localized preferences or disparities.


```python
df.head()
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
      <th>title</th>
      <th>console</th>
      <th>genre</th>
      <th>publisher</th>
      <th>developer</th>
      <th>critic_score</th>
      <th>total_sales</th>
      <th>na_sales</th>
      <th>jp_sales</th>
      <th>pal_sales</th>
      <th>other_sales</th>
      <th>release_date</th>
      <th>last_update</th>
      <th>release_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Grand Theft Auto V</td>
      <td>PS3</td>
      <td>Action</td>
      <td>Rockstar Games</td>
      <td>Rockstar North</td>
      <td>9.40000</td>
      <td>20.32</td>
      <td>6.37</td>
      <td>0.99</td>
      <td>9.85</td>
      <td>3.12</td>
      <td>2013-09-17</td>
      <td>2019-05-06</td>
      <td>2013</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Grand Theft Auto V</td>
      <td>PS4</td>
      <td>Action</td>
      <td>Rockstar Games</td>
      <td>Rockstar North</td>
      <td>9.70000</td>
      <td>19.39</td>
      <td>6.06</td>
      <td>0.60</td>
      <td>9.71</td>
      <td>3.02</td>
      <td>2014-11-18</td>
      <td>2018-03-01</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Grand Theft Auto: Vice City</td>
      <td>PS2</td>
      <td>Action</td>
      <td>Rockstar Games</td>
      <td>Rockstar North</td>
      <td>9.60000</td>
      <td>16.15</td>
      <td>8.41</td>
      <td>0.47</td>
      <td>5.49</td>
      <td>1.78</td>
      <td>2002-10-28</td>
      <td>2019-05-06</td>
      <td>2002</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Grand Theft Auto V</td>
      <td>X360</td>
      <td>Action</td>
      <td>Rockstar Games</td>
      <td>Rockstar North</td>
      <td>7.22044</td>
      <td>15.86</td>
      <td>9.06</td>
      <td>0.06</td>
      <td>5.33</td>
      <td>1.42</td>
      <td>2013-09-17</td>
      <td>2019-05-06</td>
      <td>2013</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Call of Duty: Black Ops 3</td>
      <td>PS4</td>
      <td>Shooter</td>
      <td>Activision</td>
      <td>Treyarch</td>
      <td>8.10000</td>
      <td>15.09</td>
      <td>6.18</td>
      <td>0.41</td>
      <td>6.05</td>
      <td>2.44</td>
      <td>2015-06-11</td>
      <td>2018-01-14</td>
      <td>2015</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['na_ratio'] = df['na_sales'] / df['total_sales']
df['jp_ratio'] = df['jp_sales'] / df['total_sales']
df['pal_ratio'] = df['pal_sales'] / df['total_sales']
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
      <th>title</th>
      <th>console</th>
      <th>genre</th>
      <th>publisher</th>
      <th>developer</th>
      <th>critic_score</th>
      <th>total_sales</th>
      <th>na_sales</th>
      <th>jp_sales</th>
      <th>pal_sales</th>
      <th>other_sales</th>
      <th>release_date</th>
      <th>last_update</th>
      <th>release_year</th>
      <th>na_ratio</th>
      <th>jp_ratio</th>
      <th>pal_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>18669</th>
      <td>Europa Universalis: Rome Gold</td>
      <td>PC</td>
      <td>Strategy</td>
      <td>Paradox Interactive</td>
      <td>Paradox Interactive</td>
      <td>7.22044</td>
      <td>0.0</td>
      <td>0.26474</td>
      <td>0.102281</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2009-10-07</td>
      <td>2019-05-06</td>
      <td>2009</td>
      <td>inf</td>
      <td>inf</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18051</th>
      <td>Island</td>
      <td>PS4</td>
      <td>Adventure</td>
      <td>Prototype</td>
      <td>Prototype</td>
      <td>7.22044</td>
      <td>0.0</td>
      <td>0.26474</td>
      <td>0.000000</td>
      <td>0.149472</td>
      <td>0.043041</td>
      <td>2018-06-28</td>
      <td>2018-12-08</td>
      <td>2018</td>
      <td>inf</td>
      <td>NaN</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>18032</th>
      <td>Ninki Seiyuu no Tsukurikata</td>
      <td>PSV</td>
      <td>Adventure</td>
      <td>Entergram</td>
      <td>Entergram</td>
      <td>7.22044</td>
      <td>0.0</td>
      <td>0.26474</td>
      <td>0.000000</td>
      <td>0.149472</td>
      <td>0.043041</td>
      <td>2018-01-25</td>
      <td>2018-03-24</td>
      <td>2018</td>
      <td>inf</td>
      <td>NaN</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>18033</th>
      <td>Hidden Expedition: The Uncharted Islands</td>
      <td>PC</td>
      <td>Adventure</td>
      <td>Unknown</td>
      <td>Activision</td>
      <td>7.22044</td>
      <td>0.0</td>
      <td>0.26474</td>
      <td>0.102281</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2008-08-27</td>
      <td>2019-05-06</td>
      <td>2008</td>
      <td>inf</td>
      <td>inf</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18034</th>
      <td>The Lost Crown: A Ghost-hunting Adventure</td>
      <td>PC</td>
      <td>Adventure</td>
      <td>Got Game Entertainment</td>
      <td>Darkling Room</td>
      <td>7.22044</td>
      <td>0.0</td>
      <td>0.26474</td>
      <td>0.102281</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2008-03-03</td>
      <td>2019-05-06</td>
      <td>2008</td>
      <td>inf</td>
      <td>inf</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



: > 80 %  OF TOTAL SALES IN A REGION IS CONSIDERED A HIT AND < 20 % IS A FLOP


```python
# HIT IN NA - FLOP IN JP AND PAL

na_hit = df[(df.na_ratio > 0.8) & (df.jp_ratio < 0.2) & (df.pal_ratio < 0.2)].sort_values(by = 'na_sales', ascending = False)
na_top10 = na_hit.head(10)
na_top10
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
      <th>title</th>
      <th>console</th>
      <th>genre</th>
      <th>publisher</th>
      <th>developer</th>
      <th>critic_score</th>
      <th>total_sales</th>
      <th>na_sales</th>
      <th>jp_sales</th>
      <th>pal_sales</th>
      <th>other_sales</th>
      <th>release_date</th>
      <th>last_update</th>
      <th>release_year</th>
      <th>na_ratio</th>
      <th>jp_ratio</th>
      <th>pal_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>75</th>
      <td>Madden NFL 2004</td>
      <td>PS2</td>
      <td>Sports</td>
      <td>EA Sports</td>
      <td>EA Tiburon</td>
      <td>9.50000</td>
      <td>5.23</td>
      <td>4.26</td>
      <td>0.010000</td>
      <td>0.26</td>
      <td>0.71</td>
      <td>2003-12-08</td>
      <td>2019-05-06</td>
      <td>2003</td>
      <td>0.814532</td>
      <td>0.001912</td>
      <td>0.049713</td>
    </tr>
    <tr>
      <th>114</th>
      <td>Madden NFL 2005</td>
      <td>PS2</td>
      <td>Sports</td>
      <td>EA Sports</td>
      <td>EA Tiburon</td>
      <td>9.50000</td>
      <td>4.53</td>
      <td>4.18</td>
      <td>0.010000</td>
      <td>0.26</td>
      <td>0.08</td>
      <td>2004-09-08</td>
      <td>2019-05-06</td>
      <td>2004</td>
      <td>0.922737</td>
      <td>0.002208</td>
      <td>0.057395</td>
    </tr>
    <tr>
      <th>125</th>
      <td>Asteroids</td>
      <td>2600</td>
      <td>Shooter</td>
      <td>Atari</td>
      <td>Atari</td>
      <td>7.22044</td>
      <td>4.31</td>
      <td>4.00</td>
      <td>0.102281</td>
      <td>0.26</td>
      <td>0.05</td>
      <td>1981-01-01</td>
      <td>2019-05-06</td>
      <td>1981</td>
      <td>0.928074</td>
      <td>0.023731</td>
      <td>0.060325</td>
    </tr>
    <tr>
      <th>94</th>
      <td>Madden NFL 06</td>
      <td>PS2</td>
      <td>Sports</td>
      <td>EA Sports</td>
      <td>EA Tiburon</td>
      <td>9.10000</td>
      <td>4.91</td>
      <td>3.98</td>
      <td>0.010000</td>
      <td>0.26</td>
      <td>0.66</td>
      <td>2005-08-08</td>
      <td>2019-05-06</td>
      <td>2005</td>
      <td>0.810591</td>
      <td>0.002037</td>
      <td>0.052953</td>
    </tr>
    <tr>
      <th>136</th>
      <td>Frogger</td>
      <td>PS</td>
      <td>Action</td>
      <td>Hasbro Interactive</td>
      <td>Millenium Interactive</td>
      <td>7.22044</td>
      <td>4.16</td>
      <td>3.79</td>
      <td>0.102281</td>
      <td>0.27</td>
      <td>0.11</td>
      <td>1997-09-30</td>
      <td>2019-05-06</td>
      <td>1997</td>
      <td>0.911058</td>
      <td>0.024587</td>
      <td>0.064904</td>
    </tr>
    <tr>
      <th>135</th>
      <td>Teenage Mutant Ninja Turtles</td>
      <td>NES</td>
      <td>Platform</td>
      <td>Ultra Games</td>
      <td>Konami</td>
      <td>5.90000</td>
      <td>4.17</td>
      <td>3.38</td>
      <td>0.310000</td>
      <td>0.44</td>
      <td>0.04</td>
      <td>1989-01-06</td>
      <td>2019-05-06</td>
      <td>1989</td>
      <td>0.810552</td>
      <td>0.074341</td>
      <td>0.105516</td>
    </tr>
    <tr>
      <th>139</th>
      <td>Madden NFL 2003</td>
      <td>PS2</td>
      <td>Sports</td>
      <td>EA Sports</td>
      <td>EA Tiburon</td>
      <td>9.40000</td>
      <td>4.14</td>
      <td>3.36</td>
      <td>0.010000</td>
      <td>0.21</td>
      <td>0.56</td>
      <td>2002-12-08</td>
      <td>2019-05-06</td>
      <td>2002</td>
      <td>0.811594</td>
      <td>0.002415</td>
      <td>0.050725</td>
    </tr>
    <tr>
      <th>193</th>
      <td>Assassin's Creed: Brotherhood</td>
      <td>X360</td>
      <td>Action</td>
      <td>Ubisoft</td>
      <td>Ubisoft Montreal</td>
      <td>9.10000</td>
      <td>3.53</td>
      <td>2.87</td>
      <td>0.030000</td>
      <td>0.39</td>
      <td>0.25</td>
      <td>2010-11-16</td>
      <td>2019-05-06</td>
      <td>2010</td>
      <td>0.813031</td>
      <td>0.008499</td>
      <td>0.110482</td>
    </tr>
    <tr>
      <th>247</th>
      <td>NBA 2K13</td>
      <td>X360</td>
      <td>Sports</td>
      <td>2K Sports</td>
      <td>Visual Concepts</td>
      <td>8.60000</td>
      <td>3.11</td>
      <td>2.62</td>
      <td>0.010000</td>
      <td>0.21</td>
      <td>0.28</td>
      <td>2012-02-10</td>
      <td>2018-04-01</td>
      <td>2012</td>
      <td>0.842444</td>
      <td>0.003215</td>
      <td>0.067524</td>
    </tr>
    <tr>
      <th>270</th>
      <td>Madden NFL 13</td>
      <td>X360</td>
      <td>Sports</td>
      <td>EA Sports</td>
      <td>EA Tiburon</td>
      <td>8.00000</td>
      <td>2.93</td>
      <td>2.53</td>
      <td>0.102281</td>
      <td>0.16</td>
      <td>0.24</td>
      <td>2012-08-28</td>
      <td>2018-04-01</td>
      <td>2012</td>
      <td>0.863481</td>
      <td>0.034908</td>
      <td>0.054608</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig =  px.bar(na_top10, x = 'title', y = ['na_sales', 'jp_sales', 'pal_sales'], title = 'Top 10 most popular games in North America Region')
fig.update_layout(
    width = 1000,  
    height = 600,
    xaxis_tickangle = 45
)
fig.show()
```


<div>                            <div id="558bb6d9-5452-4ed4-b8b9-5b659d69fbb4" class="plotly-graph-div" style="height:600px; width:1000px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("558bb6d9-5452-4ed4-b8b9-5b659d69fbb4")) {                    Plotly.newPlot(                        "558bb6d9-5452-4ed4-b8b9-5b659d69fbb4",                        [{"alignmentgroup":"True","hovertemplate":"variable=na_sales\u003cbr\u003etitle=%{x}\u003cbr\u003evalue=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"na_sales","marker":{"color":"#636efa","pattern":{"shape":""}},"name":"na_sales","offsetgroup":"na_sales","orientation":"v","showlegend":true,"textposition":"auto","x":["Madden NFL 2004","Madden NFL 2005","Asteroids","Madden NFL 06","Frogger","Teenage Mutant Ninja Turtles","Madden NFL 2003","Assassin's Creed: Brotherhood","NBA 2K13","Madden NFL 13"],"xaxis":"x","y":[4.26,4.18,4.0,3.98,3.79,3.38,3.36,2.87,2.62,2.53],"yaxis":"y","type":"bar"},{"alignmentgroup":"True","hovertemplate":"variable=jp_sales\u003cbr\u003etitle=%{x}\u003cbr\u003evalue=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"jp_sales","marker":{"color":"#EF553B","pattern":{"shape":""}},"name":"jp_sales","offsetgroup":"jp_sales","orientation":"v","showlegend":true,"textposition":"auto","x":["Madden NFL 2004","Madden NFL 2005","Asteroids","Madden NFL 06","Frogger","Teenage Mutant Ninja Turtles","Madden NFL 2003","Assassin's Creed: Brotherhood","NBA 2K13","Madden NFL 13"],"xaxis":"x","y":[0.01,0.01,0.10228070175438597,0.01,0.10228070175438597,0.31,0.01,0.03,0.01,0.10228070175438597],"yaxis":"y","type":"bar"},{"alignmentgroup":"True","hovertemplate":"variable=pal_sales\u003cbr\u003etitle=%{x}\u003cbr\u003evalue=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"pal_sales","marker":{"color":"#00cc96","pattern":{"shape":""}},"name":"pal_sales","offsetgroup":"pal_sales","orientation":"v","showlegend":true,"textposition":"auto","x":["Madden NFL 2004","Madden NFL 2005","Asteroids","Madden NFL 06","Frogger","Teenage Mutant Ninja Turtles","Madden NFL 2003","Assassin's Creed: Brotherhood","NBA 2K13","Madden NFL 13"],"xaxis":"x","y":[0.26,0.26,0.26,0.26,0.27,0.44,0.21,0.39,0.21,0.16],"yaxis":"y","type":"bar"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"title"},"tickangle":45},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"value"}},"legend":{"title":{"text":"variable"},"tracegroupgap":0},"title":{"text":"Top 10 most popular games in North America Region"},"barmode":"relative","width":1000,"height":600},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('558bb6d9-5452-4ed4-b8b9-5b659d69fbb4');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


GAMES THAT WERE A HIT IN NA BUT FLOPPED IN JP AND PAL - MADDEN NFL 2004 WITH HIGHEST SALES IN NA


```python
# HIT IN JP - FLOP IN NA AND PAL

jp_hit = df[(df.jp_ratio > 0.8) & (df.na_ratio < 0.2) & (df.pal_ratio < 0.2)].sort_values(by = 'jp_sales', ascending = False)
jp_top10 = jp_hit.head(10)
jp_top10
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
      <th>title</th>
      <th>console</th>
      <th>genre</th>
      <th>publisher</th>
      <th>developer</th>
      <th>critic_score</th>
      <th>total_sales</th>
      <th>na_sales</th>
      <th>jp_sales</th>
      <th>pal_sales</th>
      <th>other_sales</th>
      <th>release_date</th>
      <th>last_update</th>
      <th>release_year</th>
      <th>na_ratio</th>
      <th>jp_ratio</th>
      <th>pal_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>349</th>
      <td>Hot Shots Golf</td>
      <td>PS</td>
      <td>Sports</td>
      <td>Sony Computer Entertainment</td>
      <td>Camelot Software Planning</td>
      <td>7.22044</td>
      <td>2.56</td>
      <td>0.26000</td>
      <td>2.13</td>
      <td>0.170000</td>
      <td>0.043041</td>
      <td>1998-05-05</td>
      <td>2019-05-06</td>
      <td>1998</td>
      <td>0.101562</td>
      <td>0.832031</td>
      <td>0.066406</td>
    </tr>
    <tr>
      <th>445</th>
      <td>R.B.I. Baseball</td>
      <td>NES</td>
      <td>Sports</td>
      <td>Tengen</td>
      <td>Namco</td>
      <td>7.22044</td>
      <td>2.20</td>
      <td>0.15000</td>
      <td>2.05</td>
      <td>0.149472</td>
      <td>0.043041</td>
      <td>1988-01-01</td>
      <td>2019-05-06</td>
      <td>1988</td>
      <td>0.068182</td>
      <td>0.931818</td>
      <td>0.067942</td>
    </tr>
    <tr>
      <th>499</th>
      <td>Famista '89 - Kaimaku Han!!</td>
      <td>NES</td>
      <td>Sports</td>
      <td>Namco</td>
      <td>Namco</td>
      <td>7.22044</td>
      <td>2.05</td>
      <td>0.26474</td>
      <td>2.05</td>
      <td>0.149472</td>
      <td>0.043041</td>
      <td>1989-07-28</td>
      <td>2019-05-06</td>
      <td>1989</td>
      <td>0.129141</td>
      <td>1.000000</td>
      <td>0.072913</td>
    </tr>
    <tr>
      <th>604</th>
      <td>Dragon Quest XI</td>
      <td>3DS</td>
      <td>Role-Playing</td>
      <td>Square Enix</td>
      <td>Square Enix</td>
      <td>7.22044</td>
      <td>1.82</td>
      <td>0.26474</td>
      <td>1.82</td>
      <td>0.149472</td>
      <td>0.043041</td>
      <td>2017-07-29</td>
      <td>2018-05-01</td>
      <td>2017</td>
      <td>0.145462</td>
      <td>1.000000</td>
      <td>0.082128</td>
    </tr>
    <tr>
      <th>667</th>
      <td>Super Puyo Puyo</td>
      <td>SNES</td>
      <td>Puzzle</td>
      <td>Banpresto</td>
      <td>Compile</td>
      <td>7.22044</td>
      <td>1.70</td>
      <td>0.26474</td>
      <td>1.69</td>
      <td>0.149472</td>
      <td>0.010000</td>
      <td>1993-10-12</td>
      <td>2019-05-06</td>
      <td>1993</td>
      <td>0.155729</td>
      <td>0.994118</td>
      <td>0.087925</td>
    </tr>
    <tr>
      <th>675</th>
      <td>Tomodachi Collection: New Life</td>
      <td>3DS</td>
      <td>Simulation</td>
      <td>Nintendo</td>
      <td>Nintendo</td>
      <td>7.22044</td>
      <td>1.69</td>
      <td>0.26474</td>
      <td>1.69</td>
      <td>0.149472</td>
      <td>0.043041</td>
      <td>2013-04-13</td>
      <td>2018-06-01</td>
      <td>2013</td>
      <td>0.156651</td>
      <td>1.000000</td>
      <td>0.088445</td>
    </tr>
    <tr>
      <th>815</th>
      <td>Ninja Hattori Kun: Ninja wa Shuugyou Degogiru ...</td>
      <td>NES</td>
      <td>Platform</td>
      <td>Hudson Soft</td>
      <td>Hudson Soft</td>
      <td>7.22044</td>
      <td>1.50</td>
      <td>0.26474</td>
      <td>1.50</td>
      <td>0.149472</td>
      <td>0.000000</td>
      <td>1986-05-03</td>
      <td>2019-05-06</td>
      <td>1986</td>
      <td>0.176493</td>
      <td>1.000000</td>
      <td>0.099648</td>
    </tr>
    <tr>
      <th>873</th>
      <td>Dragon Ball Z</td>
      <td>SNES</td>
      <td>Fighting</td>
      <td>Bandai</td>
      <td>TOSE</td>
      <td>7.22044</td>
      <td>1.45</td>
      <td>0.26474</td>
      <td>1.45</td>
      <td>0.149472</td>
      <td>0.000000</td>
      <td>1999-01-01</td>
      <td>2019-05-06</td>
      <td>1999</td>
      <td>0.182579</td>
      <td>1.000000</td>
      <td>0.103084</td>
    </tr>
    <tr>
      <th>872</th>
      <td>Tamagotchi</td>
      <td>GB</td>
      <td>Simulation</td>
      <td>Bandai</td>
      <td>Tom Create</td>
      <td>7.22044</td>
      <td>1.45</td>
      <td>0.26474</td>
      <td>1.44</td>
      <td>0.149472</td>
      <td>0.010000</td>
      <td>1997-06-26</td>
      <td>2019-05-06</td>
      <td>1997</td>
      <td>0.182579</td>
      <td>0.993103</td>
      <td>0.103084</td>
    </tr>
    <tr>
      <th>869</th>
      <td>Game de Hakken!! Tamagotchi 2</td>
      <td>GB</td>
      <td>Simulation</td>
      <td>Bandai</td>
      <td>Tom Create</td>
      <td>7.22044</td>
      <td>1.45</td>
      <td>0.26474</td>
      <td>1.44</td>
      <td>0.149472</td>
      <td>0.010000</td>
      <td>1997-10-17</td>
      <td>2019-05-06</td>
      <td>1997</td>
      <td>0.182579</td>
      <td>0.993103</td>
      <td>0.103084</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = px.bar(jp_top10, x = 'title', y = ['na_sales', 'jp_sales', 'pal_sales'], title = 'Top 10 most popular games in Japan Region')
fig.update_layout(
    width = 1000,  
    height = 700,
    xaxis_tickangle = 45
)
fig.show()
```


<div>                            <div id="072ac1fa-4efd-4d96-b7d0-76c3b785a3cd" class="plotly-graph-div" style="height:700px; width:1000px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("072ac1fa-4efd-4d96-b7d0-76c3b785a3cd")) {                    Plotly.newPlot(                        "072ac1fa-4efd-4d96-b7d0-76c3b785a3cd",                        [{"alignmentgroup":"True","hovertemplate":"variable=na_sales\u003cbr\u003etitle=%{x}\u003cbr\u003evalue=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"na_sales","marker":{"color":"#636efa","pattern":{"shape":""}},"name":"na_sales","offsetgroup":"na_sales","orientation":"v","showlegend":true,"textposition":"auto","x":["Hot Shots Golf","R.B.I. Baseball","Famista '89 - Kaimaku Han!!","Dragon Quest XI","Super Puyo Puyo","Tomodachi Collection: New Life","Ninja Hattori Kun: Ninja wa Shuugyou Degogiru no Maki","Dragon Ball Z","Tamagotchi","Game de Hakken!! Tamagotchi 2"],"xaxis":"x","y":[0.26,0.15,0.26474004906227744,0.26474004906227744,0.26474004906227744,0.26474004906227744,0.26474004906227744,0.26474004906227744,0.26474004906227744,0.26474004906227744],"yaxis":"y","type":"bar"},{"alignmentgroup":"True","hovertemplate":"variable=jp_sales\u003cbr\u003etitle=%{x}\u003cbr\u003evalue=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"jp_sales","marker":{"color":"#EF553B","pattern":{"shape":""}},"name":"jp_sales","offsetgroup":"jp_sales","orientation":"v","showlegend":true,"textposition":"auto","x":["Hot Shots Golf","R.B.I. Baseball","Famista '89 - Kaimaku Han!!","Dragon Quest XI","Super Puyo Puyo","Tomodachi Collection: New Life","Ninja Hattori Kun: Ninja wa Shuugyou Degogiru no Maki","Dragon Ball Z","Tamagotchi","Game de Hakken!! Tamagotchi 2"],"xaxis":"x","y":[2.13,2.05,2.05,1.82,1.69,1.69,1.5,1.45,1.44,1.44],"yaxis":"y","type":"bar"},{"alignmentgroup":"True","hovertemplate":"variable=pal_sales\u003cbr\u003etitle=%{x}\u003cbr\u003evalue=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"pal_sales","marker":{"color":"#00cc96","pattern":{"shape":""}},"name":"pal_sales","offsetgroup":"pal_sales","orientation":"v","showlegend":true,"textposition":"auto","x":["Hot Shots Golf","R.B.I. Baseball","Famista '89 - Kaimaku Han!!","Dragon Quest XI","Super Puyo Puyo","Tomodachi Collection: New Life","Ninja Hattori Kun: Ninja wa Shuugyou Degogiru no Maki","Dragon Ball Z","Tamagotchi","Game de Hakken!! Tamagotchi 2"],"xaxis":"x","y":[0.17,0.14947208359326264,0.14947208359326264,0.14947208359326264,0.14947208359326264,0.14947208359326264,0.14947208359326264,0.14947208359326264,0.14947208359326264,0.14947208359326264],"yaxis":"y","type":"bar"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"title"},"tickangle":45},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"value"}},"legend":{"title":{"text":"variable"},"tracegroupgap":0},"title":{"text":"Top 10 most popular games in Japan Region"},"barmode":"relative","width":1000,"height":700},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('072ac1fa-4efd-4d96-b7d0-76c3b785a3cd');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


GAMES THAT WERE A HIT IN JP BUT FLOPPED IN NA AND PAL - HOT SHOTS GOLF WITH HIGHEST SALES IN JP


```python
# HIT IN PAL - FLOP IN NA AND JP

pal_hit = df[(df.pal_ratio > 0.8) & (df.na_ratio < 0.2) & (df.jp_ratio < 0.2)].sort_values(by = 'pal_sales', ascending = False)
pal_top10 = pal_hit.head(10)
pal_top10
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
      <th>title</th>
      <th>console</th>
      <th>genre</th>
      <th>publisher</th>
      <th>developer</th>
      <th>critic_score</th>
      <th>total_sales</th>
      <th>na_sales</th>
      <th>jp_sales</th>
      <th>pal_sales</th>
      <th>other_sales</th>
      <th>release_date</th>
      <th>last_update</th>
      <th>release_year</th>
      <th>na_ratio</th>
      <th>jp_ratio</th>
      <th>pal_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>34</th>
      <td>The Sims 3</td>
      <td>PC</td>
      <td>Simulation</td>
      <td>Electronic Arts</td>
      <td>EA Redwood Shores</td>
      <td>8.50000</td>
      <td>7.96</td>
      <td>1.01000</td>
      <td>0.102281</td>
      <td>6.46</td>
      <td>0.500000</td>
      <td>2009-02-06</td>
      <td>2019-05-06</td>
      <td>2009</td>
      <td>0.126884</td>
      <td>0.012849</td>
      <td>0.811558</td>
    </tr>
    <tr>
      <th>280</th>
      <td>Colin McRae Rally</td>
      <td>PS</td>
      <td>Racing</td>
      <td>Sony Computer Entertainment</td>
      <td>Codemasters</td>
      <td>7.90000</td>
      <td>2.87</td>
      <td>0.09000</td>
      <td>0.120000</td>
      <td>2.43</td>
      <td>0.220000</td>
      <td>2000-01-31</td>
      <td>2019-05-06</td>
      <td>2000</td>
      <td>0.031359</td>
      <td>0.041812</td>
      <td>0.846690</td>
    </tr>
    <tr>
      <th>1007</th>
      <td>TOCA 2 Touring Car Championship</td>
      <td>PS</td>
      <td>Racing</td>
      <td>Codemasters</td>
      <td>Codemasters</td>
      <td>7.22044</td>
      <td>1.32</td>
      <td>0.03000</td>
      <td>0.020000</td>
      <td>1.16</td>
      <td>0.110000</td>
      <td>1999-10-31</td>
      <td>2019-05-06</td>
      <td>1999</td>
      <td>0.022727</td>
      <td>0.015152</td>
      <td>0.878788</td>
    </tr>
    <tr>
      <th>918</th>
      <td>Anno 2070</td>
      <td>PC</td>
      <td>Strategy</td>
      <td>Ubisoft</td>
      <td>Blue Byte Studio</td>
      <td>8.70000</td>
      <td>1.40</td>
      <td>0.26474</td>
      <td>0.102281</td>
      <td>1.14</td>
      <td>0.260000</td>
      <td>2011-11-17</td>
      <td>2019-05-06</td>
      <td>2011</td>
      <td>0.189100</td>
      <td>0.073058</td>
      <td>0.814286</td>
    </tr>
    <tr>
      <th>1077</th>
      <td>Brian Lara Cricket</td>
      <td>PS</td>
      <td>Sports</td>
      <td>Codemasters</td>
      <td>Codemasters</td>
      <td>7.22044</td>
      <td>1.26</td>
      <td>0.02000</td>
      <td>0.010000</td>
      <td>1.13</td>
      <td>0.100000</td>
      <td>1998-01-12</td>
      <td>2019-05-06</td>
      <td>1998</td>
      <td>0.015873</td>
      <td>0.007937</td>
      <td>0.896825</td>
    </tr>
    <tr>
      <th>1213</th>
      <td>Spore</td>
      <td>PC</td>
      <td>Strategy</td>
      <td>Electronic Arts</td>
      <td>Maxis</td>
      <td>7.00000</td>
      <td>1.16</td>
      <td>0.03000</td>
      <td>0.102281</td>
      <td>1.06</td>
      <td>0.070000</td>
      <td>2008-07-09</td>
      <td>2019-05-06</td>
      <td>2008</td>
      <td>0.025862</td>
      <td>0.088173</td>
      <td>0.913793</td>
    </tr>
    <tr>
      <th>1543</th>
      <td>Grand Theft Auto: San Andreas</td>
      <td>PC</td>
      <td>Action</td>
      <td>Rockstar Games</td>
      <td>Rockstar North</td>
      <td>9.40000</td>
      <td>0.97</td>
      <td>0.00000</td>
      <td>0.102281</td>
      <td>0.93</td>
      <td>0.040000</td>
      <td>2005-07-06</td>
      <td>2019-05-06</td>
      <td>2005</td>
      <td>0.000000</td>
      <td>0.105444</td>
      <td>0.958763</td>
    </tr>
    <tr>
      <th>1347</th>
      <td>Pro Evolution Soccer 2008</td>
      <td>X360</td>
      <td>Sports</td>
      <td>Konami</td>
      <td>Konami</td>
      <td>7.22044</td>
      <td>1.07</td>
      <td>0.08000</td>
      <td>0.040000</td>
      <td>0.90</td>
      <td>0.050000</td>
      <td>2008-12-03</td>
      <td>2019-05-06</td>
      <td>2008</td>
      <td>0.074766</td>
      <td>0.037383</td>
      <td>0.841121</td>
    </tr>
    <tr>
      <th>1483</th>
      <td>Winning Eleven: Pro Evolution Soccer 2007 (All...</td>
      <td>X360</td>
      <td>Sports</td>
      <td>Konami</td>
      <td>Konami Computer Entertainment Tokyo</td>
      <td>7.22044</td>
      <td>1.00</td>
      <td>0.08000</td>
      <td>0.020000</td>
      <td>0.90</td>
      <td>0.043041</td>
      <td>2007-06-02</td>
      <td>2019-05-06</td>
      <td>2007</td>
      <td>0.080000</td>
      <td>0.020000</td>
      <td>0.900000</td>
    </tr>
    <tr>
      <th>1530</th>
      <td>Fallout 3</td>
      <td>PC</td>
      <td>Role-Playing</td>
      <td>Bethesda Softworks</td>
      <td>Bethesda Game Studios</td>
      <td>9.00000</td>
      <td>0.98</td>
      <td>0.02000</td>
      <td>0.102281</td>
      <td>0.88</td>
      <td>0.070000</td>
      <td>2008-10-28</td>
      <td>2019-05-06</td>
      <td>2008</td>
      <td>0.020408</td>
      <td>0.104368</td>
      <td>0.897959</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = px.bar(pal_top10, x = 'title', y = ['na_sales', 'jp_sales', 'pal_sales'], title = 'Top 10 most popular games in PAL Region')
fig.update_layout(
    width = 1000,  
    height = 700,
    xaxis_tickangle = 45
)
fig.show()
```


<div>                            <div id="44430fe5-5d61-4636-85cf-97c7042c5326" class="plotly-graph-div" style="height:700px; width:1000px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("44430fe5-5d61-4636-85cf-97c7042c5326")) {                    Plotly.newPlot(                        "44430fe5-5d61-4636-85cf-97c7042c5326",                        [{"alignmentgroup":"True","hovertemplate":"variable=na_sales\u003cbr\u003etitle=%{x}\u003cbr\u003evalue=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"na_sales","marker":{"color":"#636efa","pattern":{"shape":""}},"name":"na_sales","offsetgroup":"na_sales","orientation":"v","showlegend":true,"textposition":"auto","x":["The Sims 3","Colin McRae Rally","TOCA 2 Touring Car Championship","Anno 2070","Brian Lara Cricket","Spore","Grand Theft Auto: San Andreas","Pro Evolution Soccer 2008","Winning Eleven: Pro Evolution Soccer 2007 (All Region sales)","Fallout 3"],"xaxis":"x","y":[1.01,0.09,0.03,0.26474004906227744,0.02,0.03,0.0,0.08,0.08,0.02],"yaxis":"y","type":"bar"},{"alignmentgroup":"True","hovertemplate":"variable=jp_sales\u003cbr\u003etitle=%{x}\u003cbr\u003evalue=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"jp_sales","marker":{"color":"#EF553B","pattern":{"shape":""}},"name":"jp_sales","offsetgroup":"jp_sales","orientation":"v","showlegend":true,"textposition":"auto","x":["The Sims 3","Colin McRae Rally","TOCA 2 Touring Car Championship","Anno 2070","Brian Lara Cricket","Spore","Grand Theft Auto: San Andreas","Pro Evolution Soccer 2008","Winning Eleven: Pro Evolution Soccer 2007 (All Region sales)","Fallout 3"],"xaxis":"x","y":[0.10228070175438597,0.12,0.02,0.10228070175438597,0.01,0.10228070175438597,0.10228070175438597,0.04,0.02,0.10228070175438597],"yaxis":"y","type":"bar"},{"alignmentgroup":"True","hovertemplate":"variable=pal_sales\u003cbr\u003etitle=%{x}\u003cbr\u003evalue=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"pal_sales","marker":{"color":"#00cc96","pattern":{"shape":""}},"name":"pal_sales","offsetgroup":"pal_sales","orientation":"v","showlegend":true,"textposition":"auto","x":["The Sims 3","Colin McRae Rally","TOCA 2 Touring Car Championship","Anno 2070","Brian Lara Cricket","Spore","Grand Theft Auto: San Andreas","Pro Evolution Soccer 2008","Winning Eleven: Pro Evolution Soccer 2007 (All Region sales)","Fallout 3"],"xaxis":"x","y":[6.46,2.43,1.16,1.14,1.13,1.06,0.93,0.9,0.9,0.88],"yaxis":"y","type":"bar"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"title"},"tickangle":45},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"value"}},"legend":{"title":{"text":"variable"},"tracegroupgap":0},"title":{"text":"Top 10 most popular games in PAL Region"},"barmode":"relative","width":1000,"height":700},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('44430fe5-5d61-4636-85cf-97c7042c5326');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


GAMES THAT WERE A HIT IN PAL BUT FLOPPED IN NA AND JP - THE SIMS 3 WITH HIGHEST SALES IN PAL

**THE END**
