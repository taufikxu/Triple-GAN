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
from library import loss_triplegan, loss_classifier, evaluation

FLAGS = flags.FLAGS
KEY_ARGUMENTS = config.load_config(FLAGS.config_file)
text_logger, MODELS_FOLDER, SUMMARIES_FOLDER = save_context(__file__, KEY_ARGUMENTS)

torch.manual_seed(1234)
torch.cuda.manual_seed(1235)
np.random.seed(1236)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FLAGS.old_model
dirname = os.path.dirname(model)
basename = os.path.basename(model)
config_path = os.path.join(dirname, "..", "source", "configs_dict.pkl")
summary_path = os.path.join(dirname, "..", "summary")
with open(config_path, "rb") as f:
    new_dict = pickle.load(f)
FLAGS.set_dict(new_dict)

itr = inputs.get_data_iter(batch_size=FLAGS.bs_c, subset=1000)
itr_u = inputs.get_data_iter(batch_size=FLAGS.bs_c)
# itr_t = inputs.get_data_iter_twice(subset=1000)
# itr_ut = inputs.get_data_iter_twice()
netG, optim_G = inputs.get_generator_optimizer()
netD, optim_D = inputs.get_discriminator_optimizer()
netC, optim_c = inputs.get_classifier_optimizer()

netG, netD, netC = netG.to(device), netD.to(device), netC.to(device)
netG = nn.DataParallel(netG)
netD = nn.DataParallel(netD)
netC = nn.DataParallel(netC)

checkpoint_io = Torture.utils.checkpoint.CheckpointIO(checkpoint_dir=MODELS_FOLDER)
checkpoint_io.register_modules(
    netG=netG, netD=netD, netC=netC, optim_G=optim_G, optim_D=optim_D, optim_c=optim_c
)
checkpoint_io.load_file(FLAGS.old_model)
logger = Logger(log_dir=SUMMARIES_FOLDER)

with torch.no_grad():
    netG.eval()
    data, label = itr.__next__()
    sample_z = torch.randn(FLAGS.bs_g, FLAGS.g_z_dim).to(device)
    tlabel = label[: FLAGS.bs_g // 10]
    tlabel = torch.cat([tlabel for _ in range(10)], 0)
    x_fake = netG(sample_z, tlabel)
    logger.add_imgs(x_fake, "imgtest", nrow=FLAGS.bs_g // 10)
