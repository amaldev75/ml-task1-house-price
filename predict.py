import pickle
import pandas as pd

# load the saved model
with open('house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

# create a new, made-up house to predict on
# columns must match training data: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
new_house = pd.DataFrame([{
    'MedInc': 5.0,
    'HouseAge': 20.0,
    'AveRooms': 6.0,
    'AveBedrms': 1.0,
    'Population': 1000.0,
    'AveOccup': 3.0,
    'Latitude': 34.0,
    'Longitude': -118.0
}])

prediction = model.predict(new_house)
print(f"Predicted median house value: ${prediction[0] * 100000:.2f}")