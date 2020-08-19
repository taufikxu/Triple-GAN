import copy
import os
import pickle

import torch
import torch.nn as nn
import numpy as np

from library import inputs, eval_inception_score
from Utils.checkpoints import save_context, Logger
from Utils import flags
from Utils import config

import Torture
from library.chainer_evaluation.evaluation import calc_inception, calc_FID, FID
import PIL.Image

FLAGS = flags.FLAGS
KEY_ARGUMENTS = config.load_config(FLAGS.config_file)
dirname = model = FLAGS.old_model
config_path = os.path.join(dirname, "..", "source", "configs_dict.pkl")
summary_path = os.path.join(dirname, "..", "summary")
with open(config_path, "rb") as f:
    new_dict = pickle.load(f)
new_dict["gpu"] = FLAGS.gpu
FLAGS.set_dict(new_dict)
FLAGS.old_model = "loaded"
text_logger, MODELS_FOLDER, SUMMARIES_FOLDER = save_context(__file__, KEY_ARGUMENTS)

torch.manual_seed(1234)
torch.cuda.manual_seed(1235)
np.random.seed(1236)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

itr = inputs.get_data_iter(batch_size=100, subset=1000)
itr_u = inputs.get_data_iter(batch_size=100, infinity=False)
netG, optim_G = inputs.get_generator_optimizer()
netD, optim_D = inputs.get_discriminator_optimizer()
netC, optim_c = inputs.get_classifier_optimizer()
netC_T, _ = inputs.get_classifier_optimizer()

netG, netD, netC = netG.to(device), netD.to(device), netC.to(device)
netG = nn.DataParallel(netG)
netC = nn.DataParallel(netC)
netC_T = nn.DataParallel(netC_T)

checkpoint_io = Torture.utils.checkpoint.CheckpointIO(checkpoint_dir=MODELS_FOLDER)
checkpoint_io.register_modules(
    netG=netG, netC=netC, netC_T=netC_T, optim_G=optim_G, optim_c=optim_c,
)
logger = Logger(log_dir=SUMMARIES_FOLDER)

# dat1 = np.load("cifar10_total.npz")
# dat2 = np.load("cifar10_total_original.npz")
# fid = FID(dat1["mean"], dat1["cov"], dat2["mean"], dat2["cov"])
# print(fid)
# exit()
# x_list = []
# for _ in range(500):
#     data, label = itr_u.__next__()
#     x_list.append(data.data.cpu().numpy() * 0.5 + 0.5)
# x_list = np.concatenate(x_list) * 255
# print(np.min(x_list), np.max(x_list))
# print(x_list.shape)
# calc_FID(x_list)
# exit()

# x_list, y_list = [], []
# for _ in range(500):
#     data, label = itr_u.__next__()
#     x_list.append(data.data.cpu().numpy() * 0.5 + 0.5)
#     y_list.append(label.data.cpu().numpy())
# x_list = np.concatenate(x_list) * 255
# y_list = np.concatenate(y_list)
# for lid in range(10):
#     tmp_x = x_list[y_list == lid]
#     calc_FID(tmp_x, 100, "./cifar10_label_{}.npz".format(lid))
# print(np.min(x_list), np.max(x_list))
# print(x_list.shape)
# exit()


# max_ism, max_id = 0, 0
# for iters in range(50000, 250001, 2000):
#     torch.manual_seed(1235)
#     torch.cuda.manual_seed(1235)
#     np.random.seed(1235)

#     model_path = os.path.join(dirname, "model{}.pt".format(iters))
#     if os.path.exists(model_path) is False:
#         continue
#     checkpoint_io.load_file(model_path)
#     # # # # Inception score
#     with torch.no_grad():
#         netG.eval()
#         img_list = []
#         for _ in range(10):
#             sample_z = torch.randn(100, FLAGS.g_z_dim).to(device)
#             data, label = itr.__next__()
#             x_fake = netG(sample_z.to(device), label.to(device))
#             img_list.append(x_fake.data.cpu().numpy() * 0.5 + 0.5)
#         img_list = np.concatenate(img_list, axis=0) * 255
#         # img_list = (np.transpose(img_list, [0, 2, 3, 1]) ).astype(np.uint8)
#         # new_img_list = []
#         # for i in range(img_list.shape[0]):
#         # new_img_list.append(img_list[i])
#         # ism, isvar = eval_inception_score.get_inception_score(new_img_list, 1, 100)
#         ism, isvar = calc_inception(img_list, 100, 5000, 1)
#         if ism > max_ism:
#             max_ism = ism
#             max_id = iters
#         text_logger.info(str((iters, ism, max_id, max_ism)))

max_id = 72000
model_path = os.path.join(dirname, "model{}.pt".format(max_id))
checkpoint_io.load_file(model_path)
# # # # Inception score
with torch.no_grad():
    torch.manual_seed(1235)
    torch.cuda.manual_seed(1235)
    np.random.seed(1235)

    netG.eval()
    img_list, label_list = [], []
    for _ in range(500):
        sample_z = torch.randn(100, FLAGS.g_z_dim).to(device)
        data, label = itr.__next__()
        x_fake = netG(sample_z.to(device), label.to(device))
        img_list.append(x_fake.data.cpu().numpy() * 0.5 + 0.5)
        label_list.append(label.data.cpu().numpy())
    img_list = np.concatenate(img_list, axis=0) * 255
    label_list = np.concatenate(label_list)
    for iid, img in enumerate(img_list):
        img = PIL.Image.fromarray(img.transpose([1, 2, 0]).astype(np.uint8))
        img.save("/home/kunxu/cifar10_tmp/{}.png".format(iid))
    # img_list = (np.transpose(img_list, [0, 2, 3, 1])).astype(np.uint8)
    # print(img_list.shape)
    # new_img_list = []
    # for i in range(img_list.shape[0]):
    #     new_img_list.append(img_list[i])
    # ism, isvar = calc_inception(img_list, 100, 50000, 10)
    # text_logger.info(str((max_id, ism, isvar)))

    fid = calc_FID(img_list, 100, "./cifar10_total_original.npz")
    text_logger.info(str((max_id, fid)))
