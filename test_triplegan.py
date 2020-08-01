import copy
import os
import pickle

import torch
import torch.nn as nn
import numpy as np

from library import inputs
from Utils.checkpoints import save_context, Logger
from Utils import flags
from Utils import config

import Torture

FLAGS = flags.FLAGS
KEY_ARGUMENTS = config.load_config(FLAGS.config_file)
model = FLAGS.old_model
dirname = os.path.dirname(model)
basename = os.path.basename(model)
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
checkpoint_io.load_file(model)
logger = Logger(log_dir=SUMMARIES_FOLDER)

# with torch.no_grad():
#     netG.eval()
#     data, label = itr.__next__()
#     sample_z = torch.randn(FLAGS.bs_g, FLAGS.g_z_dim).to(device)
#     tlabel = label[: FLAGS.bs_g // 10]
#     tlabel = torch.cat([tlabel for _ in range(10)], 0)
#     x_fake = netG(sample_z, tlabel)
#     logger.add_imgs(x_fake, "imgtest", nrow=FLAGS.bs_g // 10)

itr_test = inputs.get_data_iter(batch_size=100, train=False, infinity=False)
netC_T.eval()
total, correct = 0, 0
for images, labels in itr_test:
    images, labels = images.to(device), labels.to(device)
    outputs = netC_T(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
print(total, correct, correct / total)
