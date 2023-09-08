import os
import json
import matplotlib.pyplot as plt

# Get a list of all subdirectories in the current directory
folder_paths = [folder for folder in os.listdir() if os.path.isdir(folder)]

data_list = []

for folder_path in folder_paths:
    file_path = os.path.join(folder_path, 'result_data.json')
    with open(file_path, 'r') as f:
        data = json.load(f)
        data_list.append(data)

plt.figure(figsize=(10, 5))  # Create a figure with a larger width to accommodate two subplots
plt.suptitle('Accuracy and Loss')

time_limit = 5000
for i, data in enumerate(data_list):
    filtered_data = [(d['time'] - data[0]['time'], d['acc'], d['loss']) for d in data if d['time'] - data[0]['time'] <= time_limit]

    x = [d[0] for d in filtered_data]
    y_acc = [d[1] for d in filtered_data]
    y_loss = [d[2] for d in filtered_data]

    plt.subplot(1, 2, 1)  # Create the left subplot for accuracy
    plt.plot(x, y_acc, label=folder_paths[i][20:-18])
    plt.xlabel('time')
    plt.ylabel('accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)  # Create the right subplot for loss
    plt.plot(x, y_loss, label=folder_paths[i][20:-18])
    plt.xlabel('time')
    plt.ylabel('loss')
    plt.legend()

plt.tight_layout()  # Adjust layout for better spacing
plt.show()
