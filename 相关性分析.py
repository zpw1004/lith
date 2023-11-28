import pandas as pd
import numpy as np
from PyEMD import EMD
import matplotlib.pyplot as plt

# Load your dataset (replace 'your_data.csv' with the actual file path)
data = pd.read_csv("../dataset/American/train_data.csv")

# Extract the signal you want to decompose (e.g., 'GR' column)
signal = data['GR'].values
# Create an EMD object
emd = EMD()

# Decompose the signal into IMFs
IMFs = emd(signal)

# Get the number of IMFs
num_imfs = len(IMFs)

print(IMFs)


# Set the overall figure size
plt.figure(figsize=(12, num_imfs * 2.5))  # Adjust the multiplier (2.5) to control the height

# Plot the original signal
plt.subplot(num_imfs + 1, 1, 1)
plt.plot(signal, 'b', label='Original Signal')

# Plot each IMF with a specified height
for i, imf in enumerate(IMFs):
    plt.subplot(num_imfs + 1, 1, i + 2)
    plt.plot(imf, 'r', label=f'IMF {i + 1}')

    # Adjust the ylim to set the desired height
    plt.ylim([-50, 50])  # Set the desired height range for each IMF

plt.tight_layout()
plt.show()
