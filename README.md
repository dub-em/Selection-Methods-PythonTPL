# Selection Method

### Purpose of the Package
+ Selection Method is a Python module which implements a statistical learning method for selecting features (for predicting a target variable) in a given dataset.

### Features
+ Collection of Feature Selection Methods.
    - Forward Stepwise
    - Backward Stepwise

### Getting Started
The package can be found on pypi hence you can install it using pip

### Installation
```bash
pip install Selection_Method 
```

### Usage
Forward_Stepwise
```python
>>> from Selection_Method.Forward_Stepwise import forward_stepwise
>>>
>>> #initialize forward_stepwise object, and your already created regression model object.
>>> selection = forward_stepwise(linear_model)
>>>
>>> #input your already split train and test datasets into the .select_features() method, and select the optimal features using the stepwise algorithm.
>>> final_list, final_score = selection.select_features(x_train, x_test, y_train, y_test)
```

### Example
```python
>>> import pandas as pd
>>> from sklearn.linear_model import LinearRegression
>>> from Selection_Method.Forward_Stepwise import forward_stepwise
>>>
>>> #define your linear regression object
>>> linear_model = LinearRegression()
>>>
>>> #import your preferred dataset
>>> crime_xtrain = pd.read_csv('x_train.csv')
>>> crime_xtest = pd.read_csv('x_test.csv')
>>> crime_ytrain = pd.read_csv('y_train.csv')
>>> crime_ytest = pd.read_csv('y_test.csv')
>>>
>>> #initialize forward_stepwise object
>>> selection = forward_stepwise(linear_model) 
>>>
>>> #input your train and test dataset into the .select_features() method and execute.
>>> final_list, final_score = selection.select_features(x_train, x_test, y_train, y_test)
>>> print(forward_list, f_score)
['pctKids2Par', 'pctWhite', 'houseVacant', 'State', 'pctUrban', 'pctWorkMom18', 'persPoverty', 'pctRetire', 'pct1624', 'pctEmployMfg', 'ownHousLowQ', 'pct2Par', 'medOwnCostPctWO', 'numForeignBorn', 'medRentpctHousInc', 'pctEmploy', 'pctWwage', 'pctHousWOplumb', 'pctSameState5', 'otherPerCap', 'pctHousWOphone', 'pctPoverty', 'persPerOccupHous', 'persPerOwnOccup', 'persPerFam', 'rentMed', 'persHomeless', 'NAperCap'] 0.6315059907414283
```

### Contribution
This Project is open to contribution and collaboration. 
Feel free to connect.

### Author
+ Main Maintainer: Michael Dubem Igbomezie
