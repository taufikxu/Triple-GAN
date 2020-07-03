import glob
import pickle

model_path = "/home/kunxu/Workspace/Triple-GAN/allresults/tune_svhn_lesslabel/*n_labels_500*/summary/Model_stats.pkl"
stat_paths = glob.glob(model_path)
for p in stat_paths:
    with open(p, "rb") as f:
        dat = pickle.load(f)
        print(p, dat["testing"]["accuracy_t"][-1])
