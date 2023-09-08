import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Get a list of all subdirectories in the current directory
folder_paths = [folder for folder in os.listdir() if os.path.isdir(folder)]

data_list = []

for folder_path in folder_paths:
    file_path = os.path.join(folder_path, 'result_data.json')
    with open(file_path, 'r') as f:
        data = json.load(f)
        data_list.append(data)

time_limit = 5000
time_low = 500
smoothing_window = 15  # Adjust the window size for smoothing

plt.figure(figsize=(10, 5))  # Create a figure with a larger width to accommodate two subplots
plt.suptitle(f'Smoothed Accuracy and Loss for EMNIST [smooth window={smoothing_window}]')

for i, data in enumerate(data_list):
    filtered_data = [(d['time'] - data[0]['time'], d['acc'], d['loss']) for d in data if
                     d['time'] - data[0]['time'] <= time_limit and d['time'] - data[0]['time'] >= time_low]

    x = [d[0] for d in filtered_data]
    # x = list(range(len(filtered_data)))
    y_acc = [d[1] for d in filtered_data]
    y_loss = [d[2] for d in filtered_data]

    # Apply moving average for smoothing
    smoothed_y_acc = np.convolve(y_acc, np.ones(smoothing_window) / smoothing_window, mode='valid')
    smoothed_y_loss = np.convolve(y_loss, np.ones(smoothing_window) / smoothing_window, mode='valid')

    plt.subplot(1, 2, 1)  # Create the left subplot for smoothed accuracy
    plt.plot(x[smoothing_window - 1:], smoothed_y_acc, label=folder_paths[i][20:-18])
    plt.xlabel('time')
    plt.ylabel('accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)  # Create the right subplot for smoothed loss
    plt.plot(x[smoothing_window - 1:], smoothed_y_loss, label=folder_paths[i][20:-18])
    plt.xlabel('time')
    plt.ylabel('loss')
    plt.legend()

plt.tight_layout()  # Adjust layout for better spacing
plt.show()
