import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler

# Load the data
data = pd.read_excel('housing.xlsx')

# create a test set based on income mean
data['income_cat'] = pd.cut(data['median_income'],
                                bins=[0,1.5,3.0,4.5,6,np.inf],
                                labels=[1,2,3,4,5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data['income_cat']):
    strat_train_set = data.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = data.loc[test_index].drop("income_cat", axis=1)

# working only on the training set
housing = strat_train_set.copy()


# Seperate the predictor and the labels
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)

# Separate numerical and categorical columns
num_attributes = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attributes = ["ocean_proximity"]

# Creating pipelines from the numerical and categorical attributes
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('one_hot', OneHotEncoder(handle_unknown='ignore')),
])

# Create a full pipeline that combines both numerical and categorical pipelines

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attributes),
    ("cat", cat_pipeline, cat_attributes),
])


# Transform the data
housing_prepared = full_pipeline.fit_transform(housing)

print(housing_prepared)