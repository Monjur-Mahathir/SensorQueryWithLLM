import os 
import numpy as np 

log_file = "logs.txt"
feature_name_file = "./../UCI HAR Dataset/features.txt"

best_feature_indices = []
best_feature_names = []
best_acc = 0

with open(log_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        data = line.split()
        acc = float(data[-1])
        data = data[:-1]
        data = [int(i) for i in data]
        if acc > best_acc:
            best_acc = acc
            best_feature_indices = data
f.close()

with open(feature_name_file, 'r') as f:
    lines = f.readlines()
    for idx in best_feature_indices:
        feature_name = lines[idx].split("\n")[0].split(" ")[1]
        best_feature_names.append(feature_name)
f.close()   

out_str = ""
for i in range(len(best_feature_names)):
    out_str += f"Feature {i+1}: {best_feature_names[i]}, "

with open("best_features.txt", "a") as f:
    f.writelines(out_str)
f.close()

