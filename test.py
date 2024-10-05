from keras import models
from train import extract_data
import numpy as np
import os

model = models.load_model("NASA-Space-Apps\\Space-Apps\\model1.keras")

test_data_path = "space_apps_2024_seismic_detection\\data\\lunar\\test\\data\\S12_GradeB"

data = extract_data(test_data_path, 2153, 159) # Format: (vel_windows, temp_features, non_temp_features)

vel_array = np.array([data[label][0] for label in data])
feature_array = np.array([data[label][1] for label in data])
print(vel_array)
print(feature_array)
predictions = model.predict_on_batch([vel_array, feature_array])

print(predictions)
