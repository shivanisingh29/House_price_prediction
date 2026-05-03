# import libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.pipeline import Pipeline

# Data preprocessing
load=st.file.uploader("Upload your data",type=['csv'])
data=pd.read_csv(load)
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
accuracy=r2*100
print(f"Accuracy:{accuracy:.2f}%")
print("Mean Squared Error:",mse)
print("R2 Score:",r2)

# UI
st.metric("r2 Score (Accuracy)",f"{r2:.2f}%")
st.metric("Mean Squared Error",f"{mse:.2f}%")

# user input
st.subheader("Predict your house price")

# input for every feature
input_data={}
for col in X.columns:
    input_data[col]=st.number_input(col,value=0)

if st.button("Predict"):
    input_df=pd.DataFrame([input_data])
    prediction=pipeline.predict(input_df)
    st.success(f"Your house price is :{prediction[0]:,.0f}")


