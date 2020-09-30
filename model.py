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

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size= 0.2, random_state= 1)


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
Exploratory Data Analysis
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
housing.plot(kind = "scatter", x = "median_income", y = "median_house_value", alpha = 0.1)
plt.show()






