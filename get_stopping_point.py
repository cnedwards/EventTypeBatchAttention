
import numpy as np
import json

import os

save_path = 'model_output/'
key = 'silhouette_CLS_cosine 23'

vals = {}

window = 5

for file in os.listdir(save_path):
    if file.endswith(".log"):
        vals[int(file[:-4])] = float(eval(json.load(open(save_path + file)))[key])

#print(vals)


running_avg = {}

for i in vals:
    tmp = []

    for j in range(i-window//2, i+window//2 + 1):
        if j in vals: tmp.append(vals[j])

    running_avg[i] = np.mean(tmp)

#print(running_avg)
print('Best Iteration at:', max(running_avg, key=running_avg.get))



