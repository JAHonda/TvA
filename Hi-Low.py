import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

# Define the filter parameters
lowcut = 100.0  # Hz
highcut = 1000.0  # Hz
fs = 10000.0  # Hz
order = 5

# load the dataset
og_df = read_csv('5-2.csv', header='infer')

# Remove Duplicate Data
og_df = og_df.groupby(' Angle', as_index=False).mean()

og_x = og_df[' Angle']
og_y = og_df[' Torque']

plt.scatter(og_df[' Angle'], og_df[' Torque'], linewidths=0.5, marker=".", color='blue')

# Delete all points before first clickpoint and after last clickpoint
rm_cp = plt.ginput(2)

rm_cpx1, rm_cpy1 = rm_cp[0]
rm_cpx2, rm_cpy2 = rm_cp[1]

print('You clicked:', rm_cp)

# Plot og_df
plt.xlim(0, max(og_x)*1.05)
plt.ylim(0, max(og_y)*1.05)
plt.title('Remove excess Data')
plt.xlabel('Angle')
plt.ylabel('Torque')
plt.show()

# Remove points before first clickpoint and after last clickpoint
rm_df = og_df[og_df[' Angle'] >= rm_cpx1]
rm_df = rm_df[rm_df[' Angle'] <= rm_cpx2]

# Define the filter coefficients
nyq = 0.5 * fs
low = lowcut / nyq
high = highcut / nyq
b, a = butter(order, [low, high], btype='band')

# Apply the filter to the data
rm_df['filtered_data'] = filtfilt(b, a, rm_df[' Torque'])

# Remove spikes in Data
def relative_madness(x):
    x = rm_df
    return abs(x['filtered_data'] - np.median(x['filtered_data'])) - np.median(abs(x['filtered_data'] - np.median(x['filtered_data'])))

rm_df['Madness'] #= rm_df['filtered_data'].rolling(3, center=True).apply(relative_madness)

# plot rm_df
plt.scatter(rm_df[' Angle'], rm_df[' Torque'], linewidths=0.5, marker= ".", color='red')
plt.scatter(rm_df[' Angle'], rm_df['filtered_data'], linewidths=0.5, marker=".", color='blue')
#plt.scatter(rm_df[' Angle'], rm_df['Madness'], linewidths=0.5, marker=".", color='Red')
plt.xlim(0, max(og_x)*1.05)
plt.ylim(0, max(og_y)*1.05)
plt.show()
