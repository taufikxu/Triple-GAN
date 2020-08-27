import glob
import pickle
import os
import numpy as np


basename = "/home/kunxu/Workspace/Triple-GAN/allresults/RE_VN/*/summary"
# basename = "/home/kunxu/Workspace/Triple-GAN/allresults/RE_VN/*/summary"
# # basename = "/home/kunxu/Workspace/Triple-GAN/allresults/819_D_VN/*/summary"
# basename = "/home/kunxu/Workspace/Triple-GAN/allresults/RE_VN/*/summary"
# model_path = (
#     "/home/kunxu/Workspace/Triple-GAN/allresults/RE_VN/*08-22**(tra_0)*/summary/*.pkl"
# )
# basename = "/home/kunxu/Workspace/Triple-GAN/allresults/819_D_TI/*/summary"
# model_path = (
#     "/home/kunxu/Workspace/Triple-GAN/allresults/819_D_TI/*(tra_2)*/summary/*.pkl"
# )
model_path = (
    "/home/kunxu/Workspace/Triple-GAN/allresults/RE_VN/*08-22*(tra_0)*/summary/*.pkl"
)
# model_path = (
#     "/home/kunxu/Workspace/Triple-GAN/allresults/819_D_VN/*(tra_2)*/summary/*.pkl"
# )
# model_path = (
#     "/home/kunxu/Workspace/Triple-GAN/allresults/RE_VN/*/summary/*.pkl"
# )
# model_path = "/home/kunxu/Workspace/Triple-GAN/allresults/ELR_SVHN/*(n_labels_500)*(translate_2)*/summary/Model*.pkl"
stat_paths = glob.glob(model_path)

all_results = []
for p in stat_paths:
    ckpt_path = os.path.join(os.path.dirname(p), "../source/configs_dict.pkl")
    with open(ckpt_path, "rb") as f:
        config = pickle.load(f)
    with open(p, "rb") as f:
        dat = pickle.load(f)
        # print(p, dat["training_pre"]["loss"][-1])
        # test_dat = dat["testing"]["accuracy"]
        test_dat = dat["testing"]["accuracy_t"]

        plist = test_dat[-10:]
        acc_list = [x[1] for x in plist]
        plist = []
        for itr, v in test_dat:
            if itr > 40000 and itr % 10000 == 0:
                plist.append((itr, v))

        # print(p[len(basename) :], config["translate"], plist)
        all_results.append([p, plist, np.mean(acc_list), plist[-1]])

all_results.sort(key=lambda x: x[2])
for x in all_results:
    print(x[0][len(basename) :])
    print(x[1], " && ",  x[2], " && ",  x[3])
    print("")
