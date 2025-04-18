import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file
file_path = r'C:\Users\LiaoGuanfu\Downloads\RUL_prediction-main (1)\RUL_prediction-main\data\NASA\charge\total\B0005_charge.csv'
data = pd.read_csv(file_path)

# Filter data for cycles between 0 and 175
filtered_data = data[(data['cycle'] >= 0) & (data['cycle'] <= 600)]

# Set font and font size
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 8

# Define figure parameters
fig_width = 7.5 / 2.54  # Convert cm to inches
fig_height = 7 / 2.54
dpi = 300

# Adjust subplot margins to maximize plot area
margin = 0.1

# Plot voltage vs cycle
plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
plt.plot(filtered_data['cycle'], filtered_data['voltage_battery'], color='navy', linewidth=1.5)
plt.xlabel('Cycle')
plt.ylabel('Voltage (V)')
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.subplots_adjust(left=margin, right=1-margin, top=1-margin, bottom=margin)
plt.savefig('voltage_cycle_0_175.tiff', format='tiff', bbox_inches='tight')
plt.savefig('voltage_cycle_0_175.pdf', format='pdf', bbox_inches='tight')
plt.close()

# Plot current vs cycle
plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
plt.plot(filtered_data['cycle'], filtered_data['current_battery'], color='darkgreen', linewidth=1.5)
plt.xlabel('Cycle')
plt.ylabel('Current (A)')
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.subplots_adjust(left=margin, right=1-margin, top=1-margin, bottom=margin)
plt.savefig('current_cycle_0_175.tiff', format='tiff', bbox_inches='tight')
plt.savefig('current_cycle_0_175.pdf', format='pdf', bbox_inches='tight')
plt.close()

# Plot temperature vs cycle
plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
plt.plot(filtered_data['cycle'], filtered_data['temp_battery'], color='brown', linewidth=1.5)
plt.xlabel('Cycle')
plt.ylabel('Temperature (°C)')
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.subplots_adjust(left=margin, right=1-margin, top=1-margin, bottom=margin)
plt.savefig('temperature_cycle_0_175.tiff', format='tiff', bbox_inches='tight')
plt.savefig('temperature_cycle_0_175.pdf', format='pdf', bbox_inches='tight')
plt.close()
