'''
********************************
Project: Predicting Median House Prices for investment analysis of Real Estate Company (Hypothetical Situation)

Developed By: Varun Reddy
Project Start-Date: 28-09-2020


Reference Material: Hands-On Machine Learning with Scikit-learn, Keras & Tensorflow
********************************
'''



import pandas as pd
pd.set_option('display.max_columns', 100)


url = "https://raw.githubusercontent.com/varunreddy95/handson-ml2/master/datasets/housing/housing.csv"

def load_housing_data(x = url):
    return pd.read_csv(url)

housing = load_housing_data()
print(housing.head())
print(housing.info())
print(housing.describe())

print(housing["ocean_proximity"].value_counts()) #categorial feature with 5 categories

import matplotlib.pyplot as plt
# housing.hist(bins = 50, figsize =(20,15))
# plt.show()

import numpy as np

'''
When we closely observe at some of the attributes, we can see that the ones such as median_income which can have a high
impact on target variable (median_house_value) is skewed to the right, so when we split the data into train and test with
the simple random sampling, which might result in considerable sample bias. Hence we shall try stratified sampling in splitting
the data based on median_income
'''

housing["income_cat"] = pd.cut(housing["median_income"], bins= [0.0, 1.5, 3.0, 4.5, 6, np.inf], labels = [1,2,3,4,5])
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits= 1, test_size= 0.2, random_state= 42)
# housing["income_cat"].hist()
# plt.show()

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

#Removing the income_cat attribute that we created, so that the data is back its original state after proper sampling
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis = 1, inplace = True)

'''
*********** Exploratory Data Analysis ******************
'''

housing = strat_train_set.copy() #Copy of th training set to perform EDA

# housing.plot(kind = "scatter", x = "longitude", y = "latitude" , alpha = 0.4, s = housing["population"]/100,
#              label = "population", figsize = (10, 7), c = "median_house_value", cmap = plt.get_cmap("jet"),
#              colorbar = True)
# plt.show()

corr_matrix = housing.corr()
print("\n")
print(corr_matrix["median_house_value"].sort_values(ascending = False))

from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12, 8))
# plt.show()

'''We can see from this plot that the data is capped at 500,000.
There are less obvious patters in similar fashion at various other points
We need to remove the districts which have such price cap on them'''

# housing.plot(kind = "scatter", x = "median_income", y = "median_house_value", alpha = 0.1)
# plt.show()


#A bit of feature engineering to combine some relevant attributes
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

corr_matrix = housing.corr()
print("\n", "After Feature Engineering")
print(corr_matrix["median_house_value"].sort_values(ascending= False))

'''
****************** Data Preprocessing *******************
'''

housing = strat_train_set.drop("median_house_value", axis = 1)
housing_labels = strat_train_set["median_house_value"].copy()

#replacig the missing values in total_bedrooms attribute
from sklearn.impute import SimpleImputer    #class to replace the missing values in attributes
imputer = SimpleImputer(strategy= "median")

housing_num = housing.drop("ocean_proximity", axis  = 1)       #median can only computed on numerical attributes
imputer.fit(housing_num)

#medians of all the attributes will be stored in the imputer's statistics_ instance variable
#print(imputer.statistics_)

#We just have just fit the imputer to the dataset, but have not transformed. We shall do it now
X = imputer.transform(housing_num)     #this is a numpy array containing tranformed features

#We can transform the above array to a dataframe using the follwing code
housing_transformed = pd.DataFrame(X, columns= housing_num.columns, index= housing_num.index)


#Handling text and categorical attributes
housing_cat = housing["ocean_proximity"]
print(housing_cat.value_counts()) #this shows that there are 5 categories

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
#housing_cat_1hot.toarray()   #to convert to numpy array

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    '''
    Custom Transformer class to add the computed attributes rooms_per_household, bedrooms_per_room_,
    population_per_household. Here we added add_bedrooms_per_room as a hyperparameter, so that we can see
    if it really helps the ML algorithm or not
    '''
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y= None):
        return self
    def transform(self,X, y= None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room= False)
housing_extra_attribs = attr_adder.transform(housing.values)

'''
************** Transformation Pipelines ******************
'''

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#Pipeline for numerical attributes
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy= "median")),
    ('attr_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

housing_num_tr = num_pipeline.fit_transform(housing_num)

from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
])

housing_prepared = full_pipeline.fit_transform(housing)


'''
*********************** Machine Learning Model **************
'''

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

#testing if the model works
# some_data = housing.iloc[:5]                    #Taking only first five instances
# some_labels = housing_labels.iloc[:5]
# some_data_prepared = full_pipeline.transform(some_data)
# print("Predictions: ", lin_reg.predict(some_data_prepared))
# print("Labels: ", list(some_labels))

from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_mse = np.sqrt(lin_mse)
print("\nLinear Regression Model's Generalization Error: ", lin_mse)

'''
An error of approx. 68628.19 is not so satisfying considering the most district's median_housing_value ranging between
$120,000 and $265,000. So typical prediction error of 68628.19 is not good. This means that the model is clearly
underfitting the data. So we have 3 options
1. Use more powerful model
2. Feed algorithm with better features
3. Reduce the constraints on the model (Minimize the learning rate hyperparameter)

This Model is not regularized, hence option 3 is rules out.
Lets try first more powerful Model
'''

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print("\nDecision Tree Regression Model's Generalization Error: ", tree_rmse)

''' 
A zero generalization error means, the data is much more likely to be overfitting by the model. We test this by method of 
cross-validation 
'''

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring = "neg_mean_squared_error", cv = 10)
tree_mse_scores = np.sqrt(-scores)

print("\n")

def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())

display_scores(tree_mse_scores) #Decision Tree Model seems to be working worse than linear Regression Model

#We shall now try Random Forest Regression  Model
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)

forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_mse = np.sqrt(forest_mse)
print("\nRandom Forest Regression Model error: ", forest_mse)
print("\n")
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring = "neg_mean_squared_error", cv = 10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

'''
Hyperparameter Tuning
'''

from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators': [3,10,30], 'max_features': [2,4,6,8]},
    {'bootstrap': [False], 'n_estimators': [3,10], 'max_features': [2,3,4]}
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv = 5, scoring= 'neg_mean_squared_error', return_train_score= True)
grid_search.fit(housing_prepared, housing_labels)






