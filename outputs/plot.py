import csv
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

# Read the data

data = {"SerialTime": [], "AverageWorkerTime": [], "MasterTime": []}

dimensions = [1, 10, 100, 1000]
head_nums = [1, 2, 4, 8, 16]
for i in range(len(dimensions)):
    data["SerialTime"].append([])
    data["AverageWorkerTime"].append([])
    data["MasterTime"].append([])
    for j in range(len(head_nums)):
        data["SerialTime"][i].append(0)
        data["AverageWorkerTime"][i].append(0)
        data["MasterTime"][i].append(0)

# Read the data
csv_serialTime = "SerialTime.csv"
with open(csv_serialTime) as f:
    reader = csv.reader(f)
    next(reader)
    for dim_idx, row in enumerate(reader):
        for head_idx, cell in enumerate(row[1:]):
            data["SerialTime"][dim_idx][head_idx] = float(cell)

csv_workerTime = "AverageWorkerTime.csv"
with open(csv_workerTime) as f:
    reader = csv.reader(f)
    next(reader)
    for dim_idx, row in enumerate(reader):
        for head_idx, cell in enumerate(row[1:]):
            data["AverageWorkerTime"][dim_idx][head_idx] = float(cell)

csv_masterTime = "MasterProcessingTime.csv"
with open(csv_masterTime) as f:
    reader = csv.reader(f)
    next(reader)
    for dim_idx, row in enumerate(reader):
        for head_idx, cell in enumerate(row[1:]):
            data["MasterTime"][dim_idx][head_idx] = float(cell)

print(data)
colors = {
    "SerialTime": "cornflowerblue",
    "AverageWorkerTime": "green",
    "MasterTime": "pink",
}

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")
head_nums = np.array(head_nums)
dimensions = np.array(dimensions)
for model, model_data in data.items():
    X, Y = np.meshgrid(list(range(len(head_nums))), list(range(len(dimensions))))
    Z = np.array(model_data)
    ax.plot_surface(X, Y, Z, label=model, alpha=0.7, color=colors[model])
    ax.scatter(X, Y, Z, label=model, s=50, color=colors[model])

ax.set_xlabel("Number of Heads")
ax.set_ylabel("Dimension scale")
# set z to be log scale
# ax.set_zscale("log")
ax.set_zlabel("Time(ms)")
# ax.set_title("Model Performance Comparison")
# ax.legend(["Serial", "AverageWorker", "Master"])
legend_elements = [
    Line2D([0], [0], color=colors["SerialTime"], lw=4, label="SerialTime"),
    Line2D(
        [0], [0], color=colors["AverageWorkerTime"], lw=4, label="AverageWorkerTime"
    ),
    Line2D([0], [0], color=colors["MasterTime"], lw=4, label="MasterTime"),
]
ax.legend(handles=legend_elements)
# Manually set the ticks and labels for dimensions axis
ax.set_xticks(np.arange(len(head_nums)))
ax.set_xticklabels(head_nums)
ax.set_yticks(np.arange(len(dimensions)))
ax.set_yticklabels(dimensions)


# Manually set the ticks and labels for num_heads axis


plt.show()
