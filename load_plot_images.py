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

# from library.chainer_evaluation.evaluation import calc_inception, calc_FID, FID
import PIL.Image
import torchvision

FLAGS = flags.FLAGS
KEY_ARGUMENTS = config.load_config(FLAGS.config_file)
model = FLAGS.old_model
dirname = os.path.dirname(model)
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

# itr = inputs.get_data_iter(batch_size=100, subset=1000)
# itr_u = inputs.get_data_iter(batch_size=100, infinity=False)
netG, optim_G = inputs.get_generator_optimizer()
# netD, optim_D = inputs.get_discriminator_optimizer()
# netC, optim_c = inputs.get_classifier_optimizer()
# netC_T, _ = inputs.get_classifier_optimizer()

# netG, netD, netC = netG.to(device), netD.to(device), netC.to(device)
netG = nn.DataParallel(netG.to(device))
# netC = nn.DataParallel(netC)
# netC_T = nn.DataParallel(netC_T)

checkpoint_io = Torture.utils.checkpoint.CheckpointIO(checkpoint_dir=MODELS_FOLDER)
checkpoint_io.register_modules(netG=netG, optim_G=optim_G)
logger = Logger(log_dir=SUMMARIES_FOLDER)


torch.manual_seed(4567)
torch.cuda.manual_seed(6666)
np.random.seed(8888)


checkpoint_io.load_file(model)
# # # # Inception score
with torch.no_grad():
    netG.eval()
    tlabel = torch.from_numpy(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])).to(device)
    tlabel = torch.cat([tlabel for _ in range(10)], 0)
    tlabel = tlabel.reshape(10, 10).transpose(0, 1).reshape(100)
    sample_z = torch.randn(10, FLAGS.g_z_dim).to(device)
    sample_z = torch.cat([sample_z for _ in range(10)], 0)
    x_fake = netG(sample_z, tlabel)
    torchvision.utils.save_image(x_fake * 0.5 + 0.5, model + ".png", nrow=10)
