import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# load dataset
data = pd.read_csv("Smart_Farming_Crop_Yield_2024.csv")

# drop unnecessary columns
data = data.drop([
    'farm_id', 'sensor_id', 'timestamp',
    'sowing_date', 'harvest_date'
], axis=1)

# convert categorical columns to numeric
data = pd.get_dummies(data)

# target
target = "yield_kg_per_hectare"

X = data.drop(target, axis=1)
y = data[target]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained & saved successfully!")