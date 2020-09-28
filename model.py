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
housing.hist(bins = 50, figsize =(20,15))
plt.show()


