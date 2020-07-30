from matplotlib import pyplot as plt
import pickle

stats_ = "/home/kunxu/Workspace/Triple-GAN/allresults/AverageBaseline_resnet/(train_classifier.py)_(cifar10)_(2020-07-15-16-36-43)_((ssl_seed_1001)(n_labels_4000))_(NotValid_Signature)/summary/00149999.pkl"
with open(stats_, "rb") as f:
    stat = pickle.load(f)
test_dat = stat["testing"]["accuracy_t"]

iter_list, acc_list = [], []
for tup in test_dat:
    iter_list.append(tup[0])
    acc_list.append(tup[1])

figure = plt.figure()
plt.plot(iter_list, acc_list, "-")
figure.savefig("learning_curve.png")

i = 100
while i < len(acc_list):
    titer, tacc = 0, 0
    for j in range(i, i + 6):
        titer += iter_list[j]
        tacc += acc_list[j]
    print(titer / 6, tacc / 6)
    i = i + 6

