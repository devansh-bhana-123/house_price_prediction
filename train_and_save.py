# train_and_save.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("house_prices.csv")
X = df[['area_sqft','bedrooms','bathrooms','location','age']]
y = df['price']

num_features = ['area_sqft','bedrooms','bathrooms','age']
cat_features = ['location']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ]
)

pipe = Pipeline([
    ('pre', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)
from math import sqrt
print("RMSE:", sqrt(mean_squared_error(y_test, preds)))
print("R2:", r2_score(y_test, preds))

joblib.dump(pipe, "model.pkl")
print("Saved model.pkl")
