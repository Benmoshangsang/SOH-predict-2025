import pandas as pd

# Path to the original dataset
original_path = r'C:\Users\LiaoGuanfu\Downloads\RUL_prediction-main (1)\RUL_prediction-main\data\NASA\discharge\total\B0018_discharge.csv'

# Load the dataset into a DataFrame
data = pd.read_csv(original_path)

# Modify the 'voltage_battery' column by scaling its values to half
data['capacity'] = data['capacity'] * 0.5

# Path for the modified dataset
new_path = r'C:\Users\LiaoGuanfu\Downloads\RUL_prediction-main (1)\RUL_prediction-main\data\NASA\discharge\total\B0018_discharge_modified.csv'

# Save the modified DataFrame to a new file
data.to_csv(new_path, index=False)