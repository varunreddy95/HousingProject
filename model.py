import pandas as pd
pd.set_option('display.max_columns', 100)


url = "https://raw.githubusercontent.com/varunreddy95/handson-ml2/master/datasets/housing/housing.csv"

def load_housing_data(x = url):
    return pd.read_csv(url)

housing = load_housing_data()
print(housing.head())
print(housing.info())

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
house_labels = strat_train_set["median_house_value"].copy()

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










