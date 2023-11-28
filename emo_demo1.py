import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD

# 输入数据
data = np.array([
    [2793, 77.45, 0.664, 9.9, 11.915, 4.6, 1, 1, 3],
    [2793.5, 78.26, 0.661, 14.2, 12.565, 4.1, 1, 0.979, 3],
    [2794, 79.05, 0.658, 14.8, 13.05, 3.6, 1, 0.957, 3],
    [2794.5, 86.1, 0.655, 13.9, 13.115, 3.5, 1, 0.936, 3],
    [2795, 74.58, 0.647, 13.5, 13.3, 3.4, 1, 0.915, 3],
    [2795.5, 73.97, 0.636, 14, 13.385, 3.6, 1, 0.894, 3],
    [2796, 73.72, 0.63, 15.6, 13.93, 3.7, 1, 0.872, 3],
    [2796.5, 75.65, 0.625, 16.5, 13.92, 3.5, 1, 0.83, 3],
    [2797, 73.79, 0.624, 16.2, 13.98, 3.4, 1, 0.809, 3],
    [2797.5, 76.89, 0.615, 16.9, 14.22, 3.5, 1, 0.787, 3],
    [2798, 76.11, 0.6, 14.8, 13.375, 3.6, 1, 0.766, 3],
    [2798.5, 74.95, 0.583, 13.3, 12.69, 3.7, 1, 0.745, 3],
    [2799, 71.87, 0.561, 11.3, 12.475, 3.5, 1, 0.723, 3],
    [2799.5, 83.42, 0.537, 13.3, 14.93, 3.4, 1, 0.702, 3],
    [2800, 90.1, 0.519, 14.3, 16.555, 3.2, 1, 0.681, 2],
    [2800.5, 78.15, 0.467, 11.8, 15.96, 3.1, 1, 0.638, 2],
    [2801, 69.3, 0.438, 9.5, 15.12, 3.1, 1, 0.617, 2],
    [2801.5, 63.54, 0.418, 8.8, 15.19, 3, 1, 0.596, 2],
    [2802, 63.87, 0.401, 7.2, 15.39, 2.9, 1, 0.574, 2]
])

time = data[:, 0]
data_to_analyze = data[:, 1:]
data_to_analyze = data_to_analyze.squeeze()
# Create an instance of the EMD class
emd = EMD()

# Perform Empirical Mode Decomposition
IMFs = emd(data_to_analyze)

# Calculate the residue manually
residue = data_to_analyze.sum(axis=1) - IMFs.sum(axis=0)

# Visualize the IMFs
for i, imf in enumerate(IMFs):
    plt.figure()
    plt.plot(time, imf)
    plt.title(f'IMF {i+1}')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)

# Plot the residue
plt.figure()
plt.plot(time, residue)
plt.title('Residue')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

# Show all plots
plt.show()