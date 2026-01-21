# import libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.pipeline import Pipeline

# Data preprocessing
data=pd.read_csv("Housing.csv")
print("Dataset shape after reduction: ",data.shape)
print(data.info())
print("Duplicate rows: ",data.duplicated().sum())
print("Null value :\n ",data.isnull().sum())

# one-hot encoding
data_encoded=pd.get_dummies(
    data,
    columns=["mainroad","guestroom","basement",
             "airconditioning","prefarea","furnishingstatus"],
    drop_first=True
)

# features & target
X=data_encoded.drop(["price","hotwaterheating"],axis=1)
y=data_encoded["price"]

# train-test split
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)

# model creation with pipeline
pipeline=Pipeline([
    ("Poly",PolynomialFeatures(degree=2,include_bias=False)),
    ("scaler",StandardScaler()),
    ("model",LinearRegression())
])

# model training
pipeline.fit(X_train,y_train)

# model predition
y_pred=pipeline.predict(X_test)

# evaluation
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

print("Mean Squared Error:",mse)
print("R2 Score:",r2)


