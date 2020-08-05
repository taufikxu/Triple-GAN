import copy
import os
import pickle

import torch
import torch.nn as nn
import numpy as np

from matplotlib import pyplot as plt

from library import inputs
from Utils.checkpoints import save_context, Logger
from Utils import flags
from Utils import config

import Torture
from library import evaluation, eval_inception_score

FLAGS = flags.FLAGS
KEY_ARGUMENTS = config.load_config(FLAGS.config_file)
FLAGS.subfolder = "Test"
old_c = FLAGS.old_model_c
old_gan = FLAGS.old_model_gan
FLAGS.old_model_c = "loaded"
FLAGS.old_model_gan = "loaded"
# text_logger, MODELS_FOLDER, SUMMARIES_FOLDER = save_context(__file__, KEY_ARGUMENTS)

torch.manual_seed(1234)
torch.cuda.manual_seed(1235)
np.random.seed(1236)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLAGS.device = device

# model = FLAGS.old_model_c
# dirname = os.path.dirname(model)
# basename = os.path.basename(model)
# config_path = os.path.join(dirname, "..", "source", "configs_dict.pkl")
# summary_path = os.path.join(dirname, "..", "summary")
# with open(config_path, "rb") as f:
#     new_dict = pickle.load(f)
# FLAGS.set_dict(new_dict)

itr = inputs.get_data_iter(batch_size=FLAGS.bs_c, subset=1000)
itr_u = inputs.get_data_iter(batch_size=FLAGS.bs_c)
itr_test = inputs.get_data_iter(batch_size=FLAGS.bs_c, train=False)
# itr_t = inputs.get_data_iter_twice(subset=1000)
# itr_ut = inputs.get_data_iter_twice()
netG, optim_G = inputs.get_generator_optimizer()
netD, optim_D = inputs.get_discriminator_optimizer()
netC, optim_c = inputs.get_classifier_optimizer()
netC_T, _ = inputs.get_classifier_optimizer()

netG, netD, netC = netG.to(device), netD.to(device), netC.to(device)
netG = nn.DataParallel(netG)
netD = nn.DataParallel(netD)
netC = nn.DataParallel(netC)
netC_T = nn.DataParallel(netC_T)

checkpoint_io_gan = Torture.utils.checkpoint.CheckpointIO(checkpoint_dir=MODELS_FOLDER)
checkpoint_io_gan.register_modules(netG=netG, netD=netD)
checkpoint_io_c = Torture.utils.checkpoint.CheckpointIO(checkpoint_dir=MODELS_FOLDER)
checkpoint_io_c.register_modules(netC=netC, netC_T=netC_T)

# checkpoint_io_gan.load_file(FLAGS.old_model_gan)
checkpoint_io_c.load_file(old_c)
logger = Logger(log_dir=SUMMARIES_FOLDER)

# total, correct, _ = evaluation.test_classifier(netC_T)
# print(total, correct, (100 * (correct / total)))

# with torch.no_grad():
#     netG.eval()
#     data, label = itr.__next__()
#     sample_z = torch.randn(FLAGS.bs_g, FLAGS.g_z_dim).to(device)
#     tlabel = label[: FLAGS.bs_g // 10]
#     tlabel = torch.cat([tlabel for _ in range(10)], 0)
#     x_fake = netG(sample_z, tlabel)
#     logger.add_imgs(x_fake, "imgtest", nrow=FLAGS.bs_g // 10)

# # # # Inception score
with torch.no_grad():
    netG.eval()
    img_list = []
    for _ in range(100):
        sample_z = torch.randn(FLAGS.bs_g, FLAGS.g_z_dim).to(device)
        data, label = itr.__next__()
        x_fake = netG(sample_z.to(device), label.to(device))
        img_list.append(x_fake.data.cpu().numpy() * 0.5 + 0.5)
    img_list = np.concatenate(img_list, axis=0)
    img_list = np.transpose(img_list, [0, 2, 3, 1])
    print(img_list.shape)
    print(eval_inception_score.get_inception_score(img_list))

with torch.no_grad():
    netG.eval()
    confidence_sum = 0
    total, correct = 0, 0
    softmax = nn.Softmax(dim=1)
    conf_cor_list, conf_wor_list = [], []
    for _ in range(100):
        sample_z = torch.randn(FLAGS.bs_g, FLAGS.g_z_dim).to(device)
        data, label = itr_test.__next__()
        label = label.to(device)
        # x_fake = netG(sample_z.to(device), label.to(device))
        x_fake = data.to(device)

        outputs = netC_T(x_fake)
        confidence, predicted = torch.max(softmax(outputs).data, 1)
        confidence_sum += confidence.sum()
        total += label.size(0)
        correct += (predicted == label).sum().item()

        confidence_cor = confidence[predicted == label]
        confidence_wor = confidence[predicted != label]
        conf_cor_list.append(confidence_cor.data.cpu().numpy())
        conf_wor_list.append(confidence_wor.data.cpu().numpy())
    print(total, correct, (100 * (correct / total)), confidence_sum / total)
    conf_cor_list = np.concatenate(conf_cor_list)
    conf_wor_list = np.concatenate(conf_wor_list)
    print(conf_cor_list.shape)
    bins = []
    for i in range(10):
        bins.append(0.9 + 0.01 * i)
    figure = plt.figure()
    plt.hist([conf_cor_list, conf_wor_list], bins, label=["c", "w"])
    plt.legend()
    figure.savefig("confidence.pdf")
