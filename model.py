import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
import pickle
#import the dataset
data=pd.read_csv('Data/data.csv')
#To find out the index of all categorical column
print([data.columns.get_loc(i) for i in data.columns if i in data.dtypes[data.dtypes =="object"].index])

# All columns except last
X = data.iloc[:, :-1].values
# last column 
y = data.iloc[:, -1].values

#After getting index value apply label encoder on that indexs
labelencoder_X_1 = LabelEncoder()
X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])

labelencoder_X_2 = LabelEncoder()
X[:, 3] = labelencoder_X_2.fit_transform(X[:, 3])

labelencoder_X_3 = LabelEncoder()
X[:, 8] = labelencoder_X_3.fit_transform(X[:, 8])

labelencoder_X_4 = LabelEncoder()
X[:, 9] = labelencoder_X_4.fit_transform(X[:, 9])

labelencoder_X_5 = LabelEncoder()
X[:, 13] = labelencoder_X_4.fit_transform(X[:, 13])

labelencoder_X_6 = LabelEncoder()
X[:, 17] = labelencoder_X_4.fit_transform(X[:, 17])

labelencoder_X_7 = LabelEncoder()
X[:, 18] = labelencoder_X_4.fit_transform(X[:, 18])



# Training of a model using GradientBoosting Regressor
model = ensemble.GradientBoostingRegressor()
model.fit(X, y)

# Save the trained model in pickle format
pickle.dump(model,open("model.pkl","wb"))
