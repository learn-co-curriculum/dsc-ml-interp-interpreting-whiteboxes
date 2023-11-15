# Interpreting White-Boxes: Building a Decision Engine

## Introduction

In general white-box models are intrinsically interpretable. In other words, the mechanism used by the model to make the decision is more transparent and easy to explain in non-technical terms than complex tree-based models and neural networks. __Decision engines__ are pipelines that are specially programmed with __business rules__ that are derived from __domain knowledge__. Many decision engines use a combination of business rules and machine learning to achieve more reliable results. Business rules define the criteria and constraints that the decision engine must follow, while machine learning provides the ability to learn from data and identify patterns and relationships in the information that is relevant to the decision being made.

In this lesson, we will use scikit-learn to build a decision engine that we can use in an urban development scenario to classify homes as McMansions. First, we will discuss the use-case and prepare you with some domain knowledge. Then, we will discuss how to formulate the business rules that define McMansions. Next, we will code those rules in Python and apply them to our training data. Then, we will use two white-box models in scikit-learn, logistic regression and decision trees, to build a decision engine for buying McMansions. Finally, we will review some of the techniques available in scikit-learn to interpret the results of our model. Most importantly, we will explain how the pieces of the decision engine work together to achieve interpretable results.

## Objectives

You will be able to:

* Describe a decision engine and how they use business rules and machine learning to automate decisions
* Build a decision engine in scikit-learn using business rules and white-box models
* Generate metrics to interpret and evaluate results of decision engine
* Create a high level explanation based on metrics and domain knowledge. 

## Building a Decision Engine to Buy McMansions

Decision engines automate decision making by using two kinds of inputs, business rules and machine learning. The rationale for using both is that combining them can yield more interpretable results. Here are some reasons why:

* Business rules may provide a baseline decision based on predefined criteria and can act as a default decision when machine learning models are uncertain.
* Machine learning models may be used to provide additional insight and suggest modifications to the business rules, based on the data and patterns that they have learned from previous decisions.
* This combination allows decision engines to be both flexible and consistent in their decision-making, combining the strengths of both business rules and machine learning.

The general process for building the decision engine will be as follows:
1. Gather domain knowledge
2. Determine methodology
3. Select training data
4. EDA/Encode business rules
6. Preprocessing/label training data
7. Train regression model
8. Train decision tree

Let's get to work!


### 1. Gather Domain Knowledge
McMansions are large, mass-produced houses built in suburban areas with a focus on size rather than architectural design. They are often criticized for their lack of aesthetic appeal and poor construction quality. They also tend to be overpriced relative to similar homes in the same area, making them a bad investment. Additionally, their cookie-cutter design can result in declining property values and decreased demand, further reducing their investment potential.

Consider the following scenario:

McMansions became popular in the late 1980s and 1990s and saw a surge in popularity through the early 2000s, especially in suburban areas. The trend was driven by a strong housing market, low interest rates, and a desire for larger homes with more amenities. The popularity of McMansions declined after the 2008 housing crisis, as more home buyers shifted towards smaller, more sustainable, and energy-efficient homes. Since many McMansions are reaching the end of their lifespan, there is a lot of talk about what to do with all of the suburban sprawl that will need to be replaced.

Suppose you work for an urban planning firm, that is tasked with identifying potential sights for a new mixed-use walkable neighborhood district to replace a large swath of aging McMansions. You decide that the best way to do this is to train a model to evaluate real-estate listings and create a list McMansions you would like to buy. Is there a way that you can build a decision engine that can automate the task of determining if a home is a McMansion, estimating the current market value, and decide if belongs on the list?

Since we do not have labeled data to classify the homes, we will have too apply labels to our training data using __business rules__ based on our __domain knowledge__ of McMansions. Then, we can use machine learning to automate that decision on future datasets, as well as learn about what other features are common among our target. 

While we may not be able to find all of these features in the data, there are some of the quintessential features that make a house a McMansion that we can distill into simple rules:

* Single-family homes
* Built between 1980 and 2010
* 2,000+ square feet
* Sale price in the upper quartile for that neighborhood

Now that we know what a McMansion is, let's try to find some.

### 2. Determine Methodology
To build the decision engine, we need to use logic to break the task down into smaller steps. Later on, this will help us to explain the decision mechanism. Broadly, this is our approach:

1. First, we need to identify examples of our target class using domain knowledge to encode business rules apply them to the training data to create the target variable `IS_MCMANSION`.

2. Then, we need train a classifier (in this case, logistic regression) to identify McMansions in our test dataset.

3. Then, we will train a decision tree classifier to identify McMansions in our dataset.

Since we have articulated our methodology, we can select and import some training data. Then, we will tackle the first task of identifying and labeling the target class.

### 3. Select and Import Training Data
To train our model, we are going to use the __Ames Housing Dataset__. In the cell below, we will import the training data, then we will do some quick EDA and apply our business rules. For this lesson, we are keeping the EDA simple to focus more on model interpretability. In practice, you would want to do a more thorough EDA to customize your preprocessing for your use case. Building models is also an iterative process, so you will often return to tweak preprocessing steps if you find that certain choices have interfered with the results. 


```python
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
```


```python
import pandas as pd
import numpy as np

df = pd.read_csv("train.csv", na_values='?')
```

### 4. Conduct EDA and Implement Business Rules
How can we code rules to identify McMansions in the training data? 

First, we identify the features that correspond the rules we want to check. Then, we use conditional logic to create columns with binary indicators that flag rows which have the correct values. For expediency, we have located the relevant features in the data and demonstrates how to derive the new features.

For a detailed description of all the features, uncomment the cell below to read the data description.


```python
# import pprint
# pp = pprint.PrettyPrinter(indent=4)


# with open('data_description.txt') as f:
#     lines = f.readlines()

# text = []
# for line in lines:
#     line=line.strip().replace('\n', '')
#     line=line.strip().replace('\t', '')
#     if len(line) > 0:
#         text += [line]
# pp.pprint(text)
```

#### Rule 1: McMansions are single family homes
In this data, we can find that using the `BldgType` field. Let's check which value would correspond with single family homes, and get a sense of how often it appears in the data. 


```python
print(df["BldgType"].value_counts())
```

    1Fam      1220
    TwnhsE     114
    Duplex      52
    Twnhs       43
    2fmCon      31
    Name: BldgType, dtype: int64


It seems that there are a lot of single family homes, represented by the value `1Fam`. Let's create a column that applies this rule to every row.


```python
df["SINGLE_FAMILY"] = df["BldgType"].apply(lambda x: 1 if x == "1Fam" else 0)
```

#### Rule 2: McMansions are built between 1980 and 2010
In this data, we can find that using the `YearBuilt` field. Let's check which value would corresponds with our date range. 


```python
print(df["YearBuilt"].value_counts())
```

    2006    67
    2005    64
    2004    54
    2007    49
    2003    45
            ..
    1906     1
    1911     1
    1913     1
    1917     1
    1872     1
    Name: YearBuilt, Length: 112, dtype: int64


Great! Now, let's apply some logic to determine which houses were built within our date range, and turn that into a binary indicator.


```python
df["IN_DATE_RANGE"] = df["YearBuilt"].apply(lambda x: 1 if (x >= 1980 and x <=2010) else 0)
print(df["IN_DATE_RANGE"].value_counts())
```

    0    848
    1    612
    Name: IN_DATE_RANGE, dtype: int64


#### Rule 3: McMansions are Greater than 2,000 sq. ft.
In this data, we get the "above grade" (or "above ground") square footage with the `GrLivArea` field. 


```python
print(df["GrLivArea"].value_counts())
```

    864     22
    1040    14
    894     11
    848     10
    1456    10
            ..
    3447     1
    1396     1
    1395     1
    1393     1
    2054     1
    Name: GrLivArea, Length: 861, dtype: int64



```python
df["ABOVE_2K_SQFT"] = df["GrLivArea"].apply(lambda x: 1 if x >= 2000 else 0)
print(df["ABOVE_2K_SQFT"].value_counts())
```

    0    1245
    1     215
    Name: ABOVE_2K_SQFT, dtype: int64


#### Sale Price
Now, we want to identify houses that sold for a lot more than other houses.


```python
df["UPPER_QUARTILE"] = df["SalePrice"] >= df["SalePrice"].describe()["75%"]
```


```python
df["UPPER_QUARTILE"].value_counts()
```




    False    1093
    True      367
    Name: UPPER_QUARTILE, dtype: int64



#### Applying the Rules
Now that used domain knowledge to lay out the business rules and we created binary indicators to flag the rows that meet the criteria, we can create a new column that scores each row based on the number of criteria indicated.


```python
df["MCMANSION_SCORE"] = df["UPPER_QUARTILE"] + df["ABOVE_2K_SQFT"] + df["IN_DATE_RANGE"] + df["SINGLE_FAMILY"]
print(df["MCMANSION_SCORE"].value_counts())
```

    1    741
    2    287
    3    205
    4    121
    0    106
    Name: MCMANSION_SCORE, dtype: int64


If we wanted to strictly interpret our rules, we would have __121 McMansions__. However, if we want to widen the net a bit, we could use a `MCMANSION_SCORE` greater than or equal to `3` as the threshold. Let's use that to create a binary indicator `IS_MCMANSON`.


```python
df["IS_MCMANSION"] = df["MCMANSION_SCORE"] >= 3
print(df["IS_MCMANSION"].value_counts())
```

    False    1134
    True      326
    Name: IS_MCMANSION, dtype: int64


Great! Now we have applied our business rules and identified __326 McMansions__. Let's also take note that we have a class imbalance of approximately 4 non-McMansions for every McMansion. This might have an impact on our results.

Now, let's train a model to identify our McMansions.

### 5. Preprocess and Label Training Data

#### Imputing Nulls

Next, we have to do some minimal preprocessing to handle null values before passing the data through our model. This 

When imputing null values, there is no one size fits all technique. For certain use cases, it is appropriate to the mean, or the median. In other cases, it is better to indicate that the value is missing. For example `PoolSF` _should_ be zero if a house has no pool. 

In the cell below, we identify all the categorical and numerical variables. Then, we encode the categorical variables with the label encoder. We print the values so we know how many values have been imputed. 


```python
# identify all of fields that have strings as values
categorical = [col for col in df.columns if df[col].dtype=="O"]
numerical = [col for col in df.columns if col not in categorical]

# fill the NaNs in numerical features as 0
print(f"Total NaNs in Numerical Features: {sum(df[numerical].isna().sum())}")
df[numerical] = df[numerical].fillna(0)

print(f"Total NaNs in Categorical Features: {sum(df[categorical].isna().sum())}")
```

    Total NaNs in Numerical Features: 348
    Total NaNs in Categorical Features: 6617


Now, we apply the label encoder to the categorical variables.


```python
from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()

# encode the categorical variables
df[categorical] = df[categorical].fillna("NULL").apply(le.fit_transform)
```

After these two steps, all of the data in our training set has been converted to a numerical format, which will allow us to use it for training our model. 

### 6. Train Regression Model

To avoid data leakage and or overfitting, we are going to remove the indicators that were created to apply our business rules from the training data, now that it is labeled. Then, the data is split into the X (data) and y (target). 


```python
drop = ["IS_MCMANSION", "MCMANSION_SCORE", "UPPER_QUARTILE", "ABOVE_2K_SQFT", "IN_DATE_RANGE", "SINGLE_FAMILY", "Id"]
keep = [col for col in df.columns if col not in drop]
X = df[keep]
y = df['IS_MCMANSION']
```

Next, we split out data into train and test sets. 


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Finally, we fit a Logistic Regression classifier. We chose the "liblinear" solver because:
* It is efficient and scalable. This will be useful if we plan to use our model for larger datasets in the future. 
* It and can handle both L1 and L2 regularization, providing additional options for hypertuning the model. 
* It also supports both binary and multi-class classification.


```python
model = LogisticRegression(solver="liblinear", max_iter=1000).fit(X_train, y_train)
```

### Evaluation and Interpretation
With our model fit, we can make predictions on test data and evaluate our results. Evaluation is an important part of interpretation, because it provides information about how confident we can be about our results.


```python
# Predict using the model
y_pred = model.predict(X_test)

# Model accuracy
acc = model.score(X_test, y_test)
print("Accuracy: ", acc)
```

    Accuracy:  0.9623287671232876


#### Model Agnostic Evaluation Metrics
Evaluation metrics like accuracy, precision, recall, and F1 can be used to evaluate any machine learning model. They can provide important information about the behavior of the model to help inform the interpretation/explanation.


```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Compute accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(acc * 100))

# Compute precision
prec = precision_score(y_test, y_pred)
print("Precision: {:.2f}%".format(prec * 100))

# Compute recall
rec = recall_score(y_test, y_pred)
print("Recall: {:.2f}%".format(rec * 100))

# Compute F1 score
f1 = f1_score(y_test, y_pred)
print("F1 Score: {:.2f}%".format(f1 * 100))
```

    Accuracy: 96.23%
    Precision: 92.98%
    Recall: 88.33%
    F1 Score: 90.60%


##### What can we infer?
We can be fairly confident in the result, because the accuracy is pretty high. We have a higher precision than recall, which suggests that the model may be slightly more sensitive than specific. In other words, we are more likely to see a false positive than a false negative.

#### Visualize
Communicating these results can be more intuitive by using a visualization.


```python
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix
sns.heatmap(cm, annot=True, cmap="YlGnBu", fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
```


    
![png](index_files/index_48_0.png)
    


#### Model Intrinsic Metrics
To further dig into how many and white kind of errors the model made, you can also use Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE).

### Coefficients
Coefficients in a logistic regression model represent the change in the log odds of the target variable for a one unit increase in the predictor variable, holding all other predictors constant. __In other words, the coefficient of a variable answers the question: "If I were to remove this variable, what are the odds that the model would still make the same decision?"__

A positive coefficient indicates that as the predictor increases, the log odds of the target variable also increases, and vice versa. A larger magnitude coefficient indicates a stronger relationship between the predictor and target. By interpreting coefficients, we can understand the impact of individual predictors on the target and how they relate to each other.


```python
# access coefficients
coefs = model.coef_.tolist()[0]
features = keep
coef_data = list(zip(features, coefs))
coef_df = pd.DataFrame(coef_data).round(3).sort_values(by=1, ascending=False)[:10]
print(coef_df.head(10))
```

                   0      1
    18     YearBuilt  0.045
    11  Neighborhood  0.036
    22   Exterior1st  0.026
    23   Exterior2nd  0.017
    75        MoSold  0.014
    70      PoolArea  0.008
    58   GarageYrBlt  0.007
    43      2ndFlrSF  0.007
    53  TotRmsAbvGrd  0.004
    32  BsmtFinType1  0.004



```python
# plot coefficients
plt.barh(coef_df[0], coef_df[1])
plt.xlabel('Coefficents')
plt.ylabel('Features')
plt.title('Top 10 Highest Coefficients by Feature')
plt.show()
```


    
![png](index_files/index_52_0.png)
    


##### What can we infer?
Based on these coefficients, we can infer that the `YearBuilt` and `Neighborhood` were the most important features. We already knew that the year would be a defining criteria, based on our business rules. Neighborhood seems to play an important role, but it's difficult to tell if this is just a consequence of the `YearBuilt` value, as houses in new developments would all be build around the same time.

Some of the other features were not explicitly part of our business rules, but make sense given our domain knowledge of McMansions. For example:

* `TotRmsAbvGrd` or "Total Rooms Above Ground", `2ndFlrSF` or "Second Floor Square Feet" both speak to __size__ which we know is an important factor. However, this finding enhances our knowledge, because it seems to matter where that extra square footage actually is.

* `BsmtFinType1`, `GarageYrBuilt`, and `PoolArea` all indicate the presence of a special amenity, like a basement, garage, or pool. All of which are characteristic of the McMansion and contribute to size.

* `Exterior1st` and `Exterior2nd` refer to the materials used on the exterior -- we can learn more about what characterized McMansions in the dataset by digging into the distribution of values in those two fields.

Building a decision engine or any machine learning model is an iterative process. While we might tweak our regression model in the next iteration, these results look good enough to continue developing the decision engine by adding our decision tree.

### Train and Evaluate Decision Tree
Now, we will perform the same classification task and use a decision tree instead.


```python
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# fit the model
model = DecisionTreeClassifier().fit(X_train, y_train)

# get the predictions
y_pred = model.predict(X_test)

# compute accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(acc * 100))

# compute precision
prec = precision_score(y_test, y_pred)
print("Precision: {:.2f}%".format(prec * 100))

# compute recall
rec = recall_score(y_test, y_pred)
print("Recall: {:.2f}%".format(rec * 100))

# Compute F1 score
f1 = f1_score(y_test, y_pred)
print("F1 Score: {:.2f}%".format(f1 * 100))
```

    Accuracy: 98.97%
    Precision: 100.00%
    Recall: 95.00%
    F1 Score: 97.44%


#### Interpreting Decision Trees
In scikit-learn, you can use the feature_importances_ attribute of the decision tree model to determine the most important features. The feature_importances_ attribute returns an array that indicates the importance of each feature in the model. The values in the array represent the average decrease in impurity (e.g., Gini impurity) that results from splitting the data based on that feature. The larger the value, the more important the feature.


```python
# get the feature importances
importances = model.feature_importances_

# create a dataframe to store the feature importances
df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

# sort the dataframe by the feature importances
df.sort_values(by='Importance', ascending=False, inplace=True)

# print the top 5 most important features
print(df.head(5))
```

          Feature  Importance
    79  SalePrice    0.806955
    14   BldgType    0.082629
    45  GrLivArea    0.040790
    48   FullBath    0.027665
    18  YearBuilt    0.017112


#### Visualize
We can also visualize the decision mechanism using a tree plot.


```python
colors = ["red", "blue"]
classes = ["NotMcMansion", "McMansion"]
fig = plt.figure(figsize=(20, 10))
_ = plot_tree(model, filled=True, feature_names=X.columns,
              class_names=classes, label='all',
              fontsize=12, precision=2, impurity=False,
              node_ids=True, rounded=True)
```


    
![png](index_files/index_59_0.png)
    


##### What can we infer?
While the results are similar the results of our regression model, there are some differences in the features that were chosen. In addition, the decision tree displays the thresholds for key values that make a difference in classifications for a specific field. 

* Sales price is the most important feature, but there were a few McMansions that were less than the threshold of \\$213K. 

* When determining if a home was a McMansion, features pertaining to size/living area, as well as amenities (`GarageYrBlt`, `FullBath`, `Fireplaces`) were very important. 

* Interestingly, the most important feature negatively associated with our target is NOT being a single family home. 

## Explaining Our Interpretation
Let's gather the inferences that were collected to illustrate the big picture:

* We can be fairly confident in the result, because the accuracy is pretty high, because both models had a high accuracy score. __It will be important to validate these results on a hold-out dataset__.
* Both models had higher precision than recall, which suggests that the model may be slightly more specific than sensitive. In other words, we are more likely to see a false negative than a false positive. The class imbalance in the data might contribute to this.
* In both models, sales price was the dominant feature to positively identify McMansions. However, we also saw that features that would be correlated with size were also important. __We should identify if it is the size these features are adding or the features themselves that is most important__. 
* Neighborhood and sales price were identified by both of the models as important factors to determine if a home was or was not a McMansion. __This could have something to do with neighborhood development trends.__

As we mentioned earlier, building decision engines and other machine learning models is an iterative process. Our interpretation of our white-box model allows us to explore this problem from many angles. You may notice that in bold, ideas that pertain to __future work__ are highlighted. The observations that we collected from this first iteration will guide our next steps as we improve our model. 

## Summary
In this lesson, we explored a use-case for white-box models in housing development, and demonstrated how you can use white-box models like logistic regression and decision trees to build a decision engine in scikit-learn. Then we discussed which metrics can be used in scikit-learn to interpret the results. We explained how evaluation metrics can contribute to the confidence we have in our results, and how coefficients and feature importance can help us to understand the most important features in the decision mechanism. In addition, we can use visualizations like bar plots and tree plots and communicate these findings in an intuitive way.

Our decision engine did a pretty good job classifying homes in the Ames Housing Dataset as McMansions. This first iteration achieved a reasonable degree of accuracy and the results make sense based on what we already understand about McMansions. The next step is to review each of our inferences and identify any areas of uncertainty to explore them further. For example, our regression model indicated via the coefficients that the exterior material of a home is an important feature. Our next iteration might include a deep dive into the distribution of exterior materials among our McMansions.
