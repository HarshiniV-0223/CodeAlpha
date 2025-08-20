import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("car_data.csv")
df['Brand'] = df['Car_Name'].apply(lambda x: x.split()[0])

# Rename columns to match standard naming
df.rename(columns={
    'Selling_Price': 'Selling_Price',
    'Present_Price': 'Present_Price',
    'Driven_kms': 'Driven_kms',
    'Selling_type': 'Seller_type',
    'Transmission': 'Transmission',
    'Owner': 'Owner'
}, inplace=True)

# Add Horsepower and Goodwill columns (if not already present)
np.random.seed(42)
df['Horsepower'] = np.random.randint(60, 250, df.shape[0])
df['Goodwill'] = np.random.randint(1, 11, df.shape[0])

# Encode categorical columns
df['Fuel_Type'] = df['Fuel_Type'].map({'Petrol': 1, 'Diesel': 0, 'CNG': 2})
df['Seller_type'] = df['Seller_type'].map({'Dealer': 1, 'Individual': 0})
df['Transmission'] = df['Transmission'].map({'Manual': 1, 'Automatic': 0})
brand_goodwill = {
    'Toyota': 9,
    'BMW': 10,
    'Tata': 6,
    'Maruti': 7,
    'Hyundai': 8,
    'Ford': 7,
    'Chevrolet': 6,
    'Renault': 6
}
df['Goodwill'] = np.random.randint(1, 11, df.shape[0])
required_columns = ['Year', 'Present_Price', 'Driven_kms', 'Fuel_Type',
                    'Seller_type', 'Transmission', 'Owner', 'Brand','Horsepower', 'Goodwill']
missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Define X and y
X = df[['Year', 'Present_Price', 'Driven_kms', 'Fuel_Type', 'Seller_type', 'Transmission', 'Owner', 'Goodwill','Horsepower']]
y = df['Selling_Price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)

feature_importances = model.feature_importances_
model.fit(X_train,y_train)
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by="Importance", ascending=True)

y_pred = model.predict(X_test)

# Scatter plot: Actual vs Predicted Prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='green', edgecolors='black')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel("Actual Price (Lakhs)")
plt.ylabel("Predicted Price (Lakhs)")
plt.title("Actual vs Predicted Car Prices")
plt.grid(True)
plt.tight_layout()
plt.show()


# Save the model and features
pickle.dump(model, open('car_price_model.pkl', 'wb'))
pickle.dump(X.columns.tolist(), open('model_features.pkl', 'wb'))

print("✅ Model and features saved successfully.")
features = pickle.load(open("model_features.pkl", "rb"))
print(features)
