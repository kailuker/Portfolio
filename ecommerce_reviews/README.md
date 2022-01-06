# Predicting Clothing Item Ratings For a Women's Clothing Company
---
# Introduction
[This kaggle dataset](https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews) includes 23486 rows of reviews from a women's e-commerce clothing company which is not identified. Each row represents a single reviewed item, and there are 10 columns of additional information about the garment and customer satisfaction.

In this project, we are investigating the roles of different factors in predicting customers' ratings of clothing items using linear regression. Most information in this dataset is self-reported, so some values may have dubious accuracy, however we can use this as a starting point for predicting item ratings.

I hypothesize that items like pants may need to be tried on to judge fit, and will be negatively correlated with ratings. As well, I predict that recommendation status will be positively correlated with ratings, as people who recommend and item will likely rate it highly.

Being able to use customer or item attributes to predict rating values can have a huge impact on sales, as it is well known that customers take reviews into account when shopping online (e.g. [Influence of consumer reviews on online purchasing decisions in older and younger adults](https://www.sciencedirect.com/science/article/pii/S0167923618300861)). This information could inform types of items carried, what products are highly advertised and to whom, as well as other business decisions.

Many of the variables included in this dataset consist of strings or binary integers. String values make linear regression difficult because they require the variables to be numerically encoded. As well, binary variables limit the amount of information included in the values, and make visualizations such as scatterplots impossible.

---
# Importing & Viewing the Data


```python
# importing libraries
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as st
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics as mt
from math import sqrt
from sklearn.model_selection import cross_val_score
```


```python
# importing data
review_data = pd.read_csv('/Users/kailuker/Documents/Data Analytics/Project3/Womens Clothing E-commerce Reviews.csv')
```


```python
# viewing the first few rows of the data to get a feel for the data structure & values
review_data.head()
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
      <th>Unnamed: 0</th>
      <th>Clothing ID</th>
      <th>Age</th>
      <th>Title</th>
      <th>Review Text</th>
      <th>Rating</th>
      <th>Recommended IND</th>
      <th>Positive Feedback Count</th>
      <th>Division Name</th>
      <th>Department Name</th>
      <th>Class Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>767</td>
      <td>33</td>
      <td>NaN</td>
      <td>Absolutely wonderful - silky and sexy and comf...</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>Initmates</td>
      <td>Intimate</td>
      <td>Intimates</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1080</td>
      <td>34</td>
      <td>NaN</td>
      <td>Love this dress!  it's sooo pretty.  i happene...</td>
      <td>5</td>
      <td>1</td>
      <td>4</td>
      <td>General</td>
      <td>Dresses</td>
      <td>Dresses</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1077</td>
      <td>60</td>
      <td>Some major design flaws</td>
      <td>I had such high hopes for this dress and reall...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>General</td>
      <td>Dresses</td>
      <td>Dresses</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1049</td>
      <td>50</td>
      <td>My favorite buy!</td>
      <td>I love, love, love this jumpsuit. it's fun, fl...</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>General Petite</td>
      <td>Bottoms</td>
      <td>Pants</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>847</td>
      <td>47</td>
      <td>Flattering shirt</td>
      <td>This shirt is very flattering to all due to th...</td>
      <td>5</td>
      <td>1</td>
      <td>6</td>
      <td>General</td>
      <td>Tops</td>
      <td>Blouses</td>
    </tr>
  </tbody>
</table>
</div>



## Column Label Glossary

### Unnamed: 0
- Index number

### Clothing ID
- Unique identifier for the specific item

### Age
- Reviewer's self-reported age

### Title
- Title of the review

### Review Text
- Body of the review

### Rating
- Integer rating from 1-5

### Recommended IND
- Binary variable
- Represents whether (1) or not (0) the customer would recommend the item to others

### Positive Feedback Count
- Number of customers that found the review helpful

### Division Name
- Highest-level item category

### Department Name
- Item department name

### Class Name
- Specific item category name

---
# Exploratory Analysis & Data Cleaning


```python
# removing spaces & capital letters from column names so that they are easier to work with
review_data.columns=['unnamed:0', 'clothing_id', 'age', 'title', 'review_text', 'rating', 'recommended_ind', 'positive_feedback_count', 'division_name', 'department_name', 'class_name']
```

## Deciding what clothing category type to use
- The category should not be too broad or too specific
- The category should not have too many categories
- department_name fits these criteria


```python
# viewing the unique values within division_name
review_data['division_name'].unique()
```




    array(['Initmates', 'General', 'General Petite', nan], dtype=object)




```python
# viewing the unique values within department_name
review_data['department_name'].unique()
```




    array(['Intimate', 'Dresses', 'Bottoms', 'Tops', 'Jackets', 'Trend', nan],
          dtype=object)




```python
# viewing the unique values within class_name
review_data['class_name'].unique()
```




    array(['Intimates', 'Dresses', 'Pants', 'Blouses', 'Knits', 'Outerwear',
           'Lounge', 'Sweaters', 'Skirts', 'Fine gauge', 'Sleep', 'Jackets',
           'Swim', 'Trend', 'Jeans', 'Legwear', 'Shorts', 'Layering',
           'Casual bottoms', nan, 'Chemises'], dtype=object)



## Investigating NaN values
- Do not want to skew the data by removing missing values that have a specific pattern
- Because most NaNs are in columns that I will not be analyzing, I will only compare the shapes of the department_name distribution with & without missing values
- The department_name column is not significantly changed when missing values are dropped, so I am comfortable dropping NaN values
- This consistency is also shown visually in the graphs below

### Counting the Number of Missing Values


```python
# counting missing values in each category
review_data.isna().sum()
```




    unnamed:0                     0
    clothing_id                   0
    age                           0
    title                      3810
    review_text                 845
    rating                        0
    recommended_ind               0
    positive_feedback_count       0
    division_name                14
    department_name              14
    class_name                   14
    dtype: int64




```python
# making a new variable to store data without missing values
no_miss_reviews = review_data.dropna()
```

### Comparing the Data With & Without Missing Values


```python
# conducting a 2-sample Kolmogorov-Smirnov test to see if shapes of the data with & without missing values is significantly different
# the shapes of the samples are not significantly different even though we have dropped missing values
ks_2samp(review_data['department_name'].value_counts(), no_miss_reviews['department_name'].value_counts())
```




    KstestResult(statistic=0.16666666666666666, pvalue=0.9999999999999998)




```python
# department_name distribution including missing values
review_data.groupby('department_name').size().plot(kind='bar', title='Item Type Distribution Including Missing Values', ylabel='Count')
```




    <AxesSubplot:title={'center':'Item Type Distribution Including Missing Values'}, xlabel='department_name', ylabel='Count'>




    
![png](for_readme_files/for_readme_18_1.png)
    



```python
# department_name distribution without missing values
no_miss_reviews.groupby('department_name').size().plot(kind='bar',title='Item Type Distribution Excluding Missing Values', ylabel='Count')
```




    <AxesSubplot:title={'center':'Item Type Distribution Excluding Missing Values'}, xlabel='department_name', ylabel='Count'>




    
![png](for_readme_files/for_readme_19_1.png)
    


## Getting Descriptive Statistics of Columns


```python
# make department_name into several numeric columns so that it can be included in the linear regression
department_name_encoded = pd.get_dummies(no_miss_reviews['department_name'])
```


```python
# adding the encoded department name columns to the rest of the data for the regression
regression_data = pd.concat([no_miss_reviews, department_name_encoded], axis=1)
```


```python
# dropping irrelevant data
regression_data.drop(columns=['unnamed:0', 'clothing_id', 'title', 'review_text', 'division_name', 'department_name', 'class_name'], inplace=True)
```


```python
# getting descriptive statistics of columns of interest
regression_data.describe()
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
      <th>age</th>
      <th>rating</th>
      <th>recommended_ind</th>
      <th>positive_feedback_count</th>
      <th>Bottoms</th>
      <th>Dresses</th>
      <th>Intimate</th>
      <th>Jackets</th>
      <th>Tops</th>
      <th>Trend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>19662.000000</td>
      <td>19662.000000</td>
      <td>19662.000000</td>
      <td>19662.000000</td>
      <td>19662.000000</td>
      <td>19662.000000</td>
      <td>19662.000000</td>
      <td>19662.000000</td>
      <td>19662.000000</td>
      <td>19662.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>43.260808</td>
      <td>4.183145</td>
      <td>0.818177</td>
      <td>2.652477</td>
      <td>0.161937</td>
      <td>0.273167</td>
      <td>0.071610</td>
      <td>0.044706</td>
      <td>0.443139</td>
      <td>0.005442</td>
    </tr>
    <tr>
      <th>std</th>
      <td>12.258122</td>
      <td>1.112224</td>
      <td>0.385708</td>
      <td>5.834285</td>
      <td>0.368402</td>
      <td>0.445597</td>
      <td>0.257848</td>
      <td>0.206662</td>
      <td>0.496769</td>
      <td>0.073571</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>34.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>41.000000</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>52.000000</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>99.000000</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>122.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Things to Note:
- With missing values removed, there are now 19662 rows
- Average rating value is 4.18, so we can conclude ratings are positively skewed
- Age has a high standard deviation, meaning age may be less reliable as a predictor variable
- Many of these columns are binary (1 or 0) and so statistics like min & max aren't useful

## Exploratory Analysis
### Ratings Distribution
- This confirms that the ratings are positively skewed
- Because of the large amount of data, this is not a cause for concern, as there are many lower ratings as well


```python
# set size of plot
plt.figure(figsize=(12, 8))
# set title of plot
plt.title(label='Ratings Distribution')
# create rating distribution
sns.histplot(data=regression_data, x='rating', hue='rating', palette='mako', discrete=True)
```




    <AxesSubplot:title={'center':'Ratings Distribution'}, xlabel='rating', ylabel='Count'>




    
![png](for_readme_files/for_readme_27_1.png)
    


### Age Distribution
- There is a large range of ages
- Most customers are 30 to 40, which is in line with the mean age
- There are outliers who are above 80 years old, contributing to the high standard deviation
- Because of an extreme number of customers at certain ages, we added a KDE to get a better idea of the population's age distribution
- We should keep an eye on age in the main analysis, as we will likely want to exclude it


```python
# set size of plot
plt.figure(figsize=(12, 8))
# set title of plot
plt.title(label='Age Distribution')
# create age distribution
sns.histplot(data=regression_data, x='age', kde=True, palette='mako', discrete=True)
```




    <AxesSubplot:title={'center':'Age Distribution'}, xlabel='age', ylabel='Count'>




    
![png](for_readme_files/for_readme_29_1.png)
    


## Creating a Correlation Heatmap
- Finding correlations between all variables included in the linear regression
- The only variables that appear to be highly correlated are recommended_ind & rating at 0.79
- This may be concerning as recommended_ind could be the only predictor of rating


```python
# set size of plot
plt.figure(figsize=(15,8))
# set title of plot
plt.title(label='Variable Correlation Heatmap')
# create heatmap
sns.heatmap(regression_data.corr(), annot=True, cmap='vlag')
```




    <AxesSubplot:title={'center':'Variable Correlation Heatmap'}>




    
![png](for_readme_files/for_readme_31_1.png)
    


---
# Main Analysis
- The main analysis for trying to predict customer ratings of clothing items
- Using a linear regression model to investigate how each independent variable contributes to the dependent variable
- Using regression modelling to test & confirm the model
- Because all of the variables in the analysis are discrete, there is no way to clearly visualize the relationships in charts


```python
# defining dependent & independent variables & adding the constant
# dependent variable is rating
dep_var = regression_data['rating']
#independent variables are all columns except rating
ind_vars = regression_data.drop('rating', axis=1)
# constant = 1
ind_vars['constant'] = 1
```

## Linear Regression Attempt 1
- R-squared is moderate, so we will try to improve the model
- Note that the coefficient of recommended_ind is much higher than any other independent variable, as we expected due to their high level of correlation


```python
# running the linear regression
lin_reg = sm.OLS(dep_var, ind_vars)
reg_results = lin_reg.fit()
print(reg_results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 rating   R-squared:                       0.630
    Model:                            OLS   Adj. R-squared:                  0.629
    Method:                 Least Squares   F-statistic:                     4175.
    Date:                Wed, 15 Dec 2021   Prob (F-statistic):               0.00
    Time:                        12:41:36   Log-Likelihood:                -20228.
    No. Observations:               19662   AIC:                         4.047e+04
    Df Residuals:                   19653   BIC:                         4.054e+04
    Df Model:                           8                                         
    Covariance Type:            nonrobust                                         
    ===========================================================================================
                                  coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------------------
    age                         0.0007      0.000      1.682      0.093      -0.000       0.001
    recommended_ind             2.2826      0.013    181.645      0.000       2.258       2.307
    positive_feedback_count    -0.0018      0.001     -2.140      0.032      -0.003      -0.000
    Bottoms                     0.3706      0.015     24.845      0.000       0.341       0.400
    Dresses                     0.3353      0.013     25.162      0.000       0.309       0.361
    Intimate                    0.3570      0.019     19.097      0.000       0.320       0.394
    Jackets                     0.3816      0.022     17.180      0.000       0.338       0.425
    Tops                        0.3395      0.013     27.058      0.000       0.315       0.364
    Trend                       0.1621      0.056      2.874      0.004       0.052       0.273
    constant                    1.9460      0.020     97.747      0.000       1.907       1.985
    ==============================================================================
    Omnibus:                     2744.372   Durbin-Watson:                   1.998
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             4226.796
    Skew:                          -0.996   Prob(JB):                         0.00
    Kurtosis:                       4.091   Cond. No.                     1.30e+17
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 2.38e-27. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.


## Linear Regression Attempt 2
- Removing age, positive_feedback_count, & Trend variables as they have low coefficients & high p-values
- This confirms what we thought, that age is a poor predictor of rating
- Removing these variables does not improve the overall model


```python
# removing positive_feedback_count, age, & Trend
ind_vars2 = ind_vars.drop(columns=['age', 'positive_feedback_count', 'Trend'])
```


```python
# running the linear regression
lin_reg2 = sm.OLS(dep_var, ind_vars2)
reg_results2 = lin_reg2.fit()
print(reg_results2.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 rating   R-squared:                       0.629
    Model:                            OLS   Adj. R-squared:                  0.629
    Method:                 Least Squares   F-statistic:                     5564.
    Date:                Wed, 15 Dec 2021   Prob (F-statistic):               0.00
    Time:                        12:41:38   Log-Likelihood:                -20231.
    No. Observations:               19662   AIC:                         4.048e+04
    Df Residuals:                   19655   BIC:                         4.053e+04
    Df Model:                           6                                         
    Covariance Type:            nonrobust                                         
    ===================================================================================
                          coef    std err          t      P>|t|      [0.025      0.975]
    -----------------------------------------------------------------------------------
    recommended_ind     2.2849      0.013    182.240      0.000       2.260       2.309
    Bottoms             0.2092      0.067      3.143      0.002       0.079       0.340
    Dresses             0.1717      0.066      2.597      0.009       0.042       0.301
    Intimate            0.1949      0.068      2.870      0.004       0.062       0.328
    Jackets             0.2197      0.069      3.168      0.002       0.084       0.356
    Tops                0.1783      0.066      2.707      0.007       0.049       0.307
    constant            2.1302      0.066     32.202      0.000       2.000       2.260
    ==============================================================================
    Omnibus:                     2745.381   Durbin-Watson:                   1.998
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             4229.430
    Skew:                          -0.996   Prob(JB):                         0.00
    Kurtosis:                       4.092   Cond. No.                         47.6
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


## Linear Regression Attempt 3: Only Rating & Recommendation
- recommended_ind appears to be accounting for most of the predictive power of the model
- Running a linear regression only between rating & recommended_ind to further investigate this relationship
- Removing all independent variables other than recommended_ind did not change the value of r-squared
- We can conclude that the relationship between rating & recommendation status is very strong, & that no other independent variables are substantially related to rating


```python
# removing all independent variables other than recommended_ind
ind_vars3 = ind_vars2.drop(columns=['Bottoms', 'Dresses', 'Intimate', 'Jackets', 'Tops'])
```


```python
# running the linear regression
lin_reg3 = sm.OLS(dep_var, ind_vars3)
reg_results3 = lin_reg3.fit()
print(reg_results3.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 rating   R-squared:                       0.629
    Model:                            OLS   Adj. R-squared:                  0.629
    Method:                 Least Squares   F-statistic:                 3.334e+04
    Date:                Wed, 15 Dec 2021   Prob (F-statistic):               0.00
    Time:                        12:41:40   Log-Likelihood:                -20240.
    No. Observations:               19662   AIC:                         4.048e+04
    Df Residuals:                   19660   BIC:                         4.050e+04
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ===================================================================================
                          coef    std err          t      P>|t|      [0.025      0.975]
    -----------------------------------------------------------------------------------
    recommended_ind     2.2871      0.013    182.601      0.000       2.263       2.312
    constant            2.3119      0.011    204.061      0.000       2.290       2.334
    ==============================================================================
    Omnibus:                     2750.775   Durbin-Watson:                   1.997
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             4242.265
    Skew:                          -0.997   Prob(JB):                         0.00
    Kurtosis:                       4.095   Cond. No.                         4.49
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


## Testing the Final Model


```python
# defining independent variables
X = ind_vars3
# defining dependent variable
y = dep_var
```


```python
# creating test & train sets for independent & dependent variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=175)
```


```python
# viewing the size & shape of the sets that were created
# each train set has 15729 rows, & each test set has 3933 rows
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
```

    (15729, 2)
    (3933, 2)
    (15729,)
    (3933,)



```python
# fitting the linear regression
lr = LinearRegression()

lr.fit(X_train, y_train)
```




    LinearRegression()




```python
# getting salary values when test values are entered
y_pred = lr.predict(X_test)
```

## Evaluating the Model
- On average, this model's prediction of ratings is off by 0.67
- The model has moderate predictive power consistent with the linear regression results


```python
# calculating RMSE
print('RMSE:',np.round(sqrt(mt.mean_squared_error(y_test,y_pred)),2))

# calculating R-Square; closer to 1 is better
print('R-Squared:',np.round(mt.r2_score(y_test,y_pred),2))
```

    RMSE: 0.67
    R-Squared: 0.62


#### Cross Validation
- Checking the model again by increasing the amount of test data
- The average cross-validation score is close to the model's R-squared value, reinforcing the medium prediction power of the model


```python
# finding the cross-validation score using 10 folds
print(np.mean(cross_val_score(lr, X_train, y_train, cv=10)))
```

    0.6315915069824405


## Relationship Between Recommendation Status & Rating
- Because of the results of Linear Regression 3, we should further investigate the relationship between recommendation status & rating
- The first chart is a bar graph showing the average rating for items that were & were not recommended
- It is clear that the average rating of those who did recommend the item is much higher than those who did not
- The second chart is a histogram showing the proportion of customers who chose each rating number & their recommendation status
- It is evident that items that were more highly rated were also recommended more often


```python
# set size for plot
plt.figure(figsize=(12, 8))
# set title of plot
plt.title(label='Mean Rating by Recommendation Status')
# comparing rating means of items that were or were not recommended
sns.barplot(data=regression_data, x='recommended_ind', y='rating', palette='mako')
```




    <AxesSubplot:title={'center':'Mean Rating by Recommendation Status'}, xlabel='recommended_ind', ylabel='rating'>




    
![png](for_readme_files/for_readme_53_1.png)
    



```python
# set size for plot
plt.figure(figsize=(12, 8))
# set title of plot
plt.title(label='Recommendation Status Totals by Rating')
# distribution of ratings with colour indicating recommendation status
sns.countplot(data=regression_data, x='rating', hue='recommended_ind', palette='mako', dodge=True)
```




    <AxesSubplot:title={'center':'Recommendation Status Totals by Rating'}, xlabel='rating', ylabel='count'>




    
![png](for_readme_files/for_readme_54_1.png)
    


---
# Conclusion
With the information provided by this dataset, we are able to predict customer ratings with moderate accuracy, however, almost all of that prediction power comes from a single variable. Though recommendation status is associated with ratings, it is essentially useless as a predictor. Similar to rating, it is a customer provided value rather than an attribute of the item or factor that can be controlled by the company. This gives the company no information on how to improve website ratings, or to improve their catalogue.

---
# Recommendations
- Include more information in the reviews that could better predict ratings (possibly rating of fit and size, selections of why they liked and did not like the item, etc.)
- Include either rating or recommendation column; they are highly correlated so both are not necessary
- Once changes have been implemented and more data is collected, use factors that correlate with rating to optimize products & push those which correlate with higher ratings
- By predicting which items will be rated higher, it is possible to influence consumer behaviour
