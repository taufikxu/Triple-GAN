import glob
import pickle
import os

basename = (
    "/home/kunxu/Workspace/Triple-GAN/allresults/ResNet_Last"
)
model_path = "/home/kunxu/Workspace/Triple-GAN/allresults/ResNet_Last/*/summary/Model_stats.pkl"
stat_paths = glob.glob(model_path)

for p in stat_paths:
    ckpt_path = os.path.join(os.path.dirname(p), "../source/configs_dict.pkl")
    with open(ckpt_path, "rb") as f:
        config = pickle.load(f)
    with open(p, "rb") as f:
        dat = pickle.load(f)
        # print(p, dat["training_pre"]["loss"][-1])
        test_dat = dat["testing"]["accuracy_t"]
        plist = []
        for itr, v in test_dat:
            if itr == 50000:
                plist.append(v)
        plist.append(test_dat[-1])

        print(p[len(basename) :], config["translate"], plist)
