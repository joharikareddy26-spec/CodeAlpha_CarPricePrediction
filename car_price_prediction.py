# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load dataset (make sure file is in same folder)
data = pd.read_csv("car_data.csv")

# Show first 5 rows
print(data.head())

# Convert text data to numbers
data['Fuel_Type'] = data['Fuel_Type'].map({'Petrol':0, 'Diesel':1, 'CNG':2})
data['Seller_Type'] = data['Seller_Type'].map({'Dealer':0, 'Individual':1})
data['Transmission'] = data['Transmission'].map({'Manual':0, 'Automatic':1})

# Input and Output
X = data.drop(['Car_Name','Selling_Price'], axis=1)
Y = data['Selling_Price']

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train model
model = LinearRegression()
model.fit(X_train, Y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
print("R2 Score:", metrics.r2_score(Y_test, predictions))

# Graph
plt.scatter(Y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Car Price Prediction")
plt.show()