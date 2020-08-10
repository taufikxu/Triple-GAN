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
itr_u = inputs.get_data_iter(batch_size=100)
netG, optim_G = inputs.get_generator_optimizer()
netD, optim_D = inputs.get_discriminator_optimizer()
netC, optim_c = inputs.get_classifier_optimizer()
netC_T, _ = inputs.get_classifier_optimizer()

netG, netD, netC = netG.to(device), netD.to(device), netC.to(device)
netG = nn.DataParallel(netG)
netD = nn.DataParallel(netD)
netC = nn.DataParallel(netC)
netC_T = nn.DataParallel(netC_T)

checkpoint_io = Torture.utils.checkpoint.CheckpointIO(checkpoint_dir=MODELS_FOLDER)
checkpoint_io.register_modules(
    netG=netG,
    netD=netD,
    netC=netC,
    netC_T=netC_T,
    optim_G=optim_G,
    optim_D=optim_D,
    optim_c=optim_c,
)
logger = Logger(log_dir=SUMMARIES_FOLDER)

max_ism, max_id = 0, 0
for iters in range(50000, 250001, 2000):
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1235)
    np.random.seed(1236)

    model_path = os.path.join(dirname, "model{}.pt".format(iters))
    if os.path.exists(model_path) is False:
        continue
    checkpoint_io.load_file(model_path)
    # # # # Inception score
    with torch.no_grad():
        netG.eval()
        img_list = []
        for _ in range(50):
            sample_z = torch.randn(100, FLAGS.g_z_dim).to(device)
            data, label = itr.__next__()
            x_fake = netG(sample_z.to(device), label.to(device))
            img_list.append(x_fake.data.cpu().numpy() * 0.5 + 0.5)
        img_list = np.concatenate(img_list, axis=0)
        img_list = (np.transpose(img_list, [0, 2, 3, 1]) * 255).astype(np.uint8)
        new_img_list = []
        for i in range(img_list.shape[0]):
            new_img_list.append(img_list[i])
        ism, isvar = eval_inception_score.get_inception_score(new_img_list, 1, 100)
        if ism > max_ism:
            max_ism = ism
            max_id = iters
        text_logger.info(str((iters, ism, max_id, max_ism)))

max_id = 52000
model_path = os.path.join(dirname, "model{}.pt".format(max_id))
checkpoint_io.load_file(model_path)
# # # # Inception score
with torch.no_grad():
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1235)
    np.random.seed(1236)

    netG.eval()
    img_list = []
    for _ in range(50):
        sample_z = torch.randn(100, FLAGS.g_z_dim).to(device)
        data, label = itr.__next__()
        x_fake = netG(sample_z.to(device), label.to(device))
        img_list.append(x_fake.data.cpu().numpy() * 0.5 + 0.5)
    img_list = np.concatenate(img_list, axis=0)
    img_list = (np.transpose(img_list, [0, 2, 3, 1]) * 255).astype(np.uint8)
    print(img_list.shape)
    new_img_list = []
    for i in range(img_list.shape[0]):
        new_img_list.append(img_list[i])
    ism, isvar = eval_inception_score.get_inception_score(new_img_list, 1, 100)
    text_logger.info(str((max_id, ism, isvar)))
