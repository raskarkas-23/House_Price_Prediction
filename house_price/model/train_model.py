import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("../data/house_price.csv")

# -------- Feature Selection --------
features = [
    "City","Locality","Property_Type","BHK","Size_in_SqFt",
    "Furnished_Status","Floor_No","Total_Floors","Age_of_Property",
    "Parking_Space","Facing","Nearby_Schools","Nearby_Hospitals",
    "Public_Transport_Accessibility","Amenities","Security"
]

target = "Price_in_Lakhs"

df = df[features + [target]]

# -------- Split X & y --------
X = df[features]
y = df[target]

# -------- Encoding --------
encoders = {}

for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# -------- Train Test Split --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------- Model --------
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("Model Accuracy (R2):", model.score(X_test, y_test))

# -------- Save --------
joblib.dump(model, "model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(features, "features.pkl")

print("✅ Model Saved Successfully")