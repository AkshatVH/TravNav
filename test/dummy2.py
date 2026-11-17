import pickle
import numpy as np

path = "./train/vint_train/data/go_stanford/no1vc_7_0/traj_data.pkl"
with open(path, "rb") as f:
    d = pickle.load(f)
for k, v in d.items():
    a = np.asarray(v)
    print(k, a.shape, a.dtype)