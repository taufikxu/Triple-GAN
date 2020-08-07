import torch
import torch.nn as nn
import numpy as np

from library import inputs
from Utils.checkpoints import save_context, Logger
from Utils import flags
from Utils import config

import Torture
from library import evaluation, loss_gan

FLAGS = flags.FLAGS
KEY_ARGUMENTS = config.load_config(FLAGS.config_file)
text_logger, MODELS_FOLDER, SUMMARIES_FOLDER = save_context(__file__, KEY_ARGUMENTS)

torch.manual_seed(1234)
torch.cuda.manual_seed(1235)
np.random.seed(1236)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

itr = inputs.get_data_iter()
netG, optim_G = inputs.get_generator_optimizer()
netD, optim_D = inputs.get_discriminator_optimizer()

netG, netD = netG.to(device), netD.to(device)
netG = nn.DataParallel(netG)
netD = nn.DataParallel(netD)

checkpoint_io = Torture.utils.checkpoint.CheckpointIO(checkpoint_dir=MODELS_FOLDER)
checkpoint_io.register_modules(netG=netG, netD=netD, optim_G=optim_G, optim_D=optim_D)
logger = Logger(log_dir=SUMMARIES_FOLDER)

# train
print_interval = 200
image_interval = 2000
max_iter = FLAGS.n_iter
batch_size = FLAGS.batch_size
loss_func_g = loss_gan.g_loss_dict[FLAGS.gan_type]
loss_func_d = loss_gan.d_loss_dict[FLAGS.gan_type]

logger_prefix = "Itera {}/{} ({:.0f}%)"
for i in range(max_iter):
    x_real, label = itr.__next__()
    x_real, label = x_real.to(device), label.to(device)

    sample_z = torch.randn(batch_size, FLAGS.g_z_dim).to(device)
    loss_d, dreal, dfake = loss_func_d(netD, netG, x_real, sample_z, label)
    optim_D.zero_grad()
    loss_d.backward()
    if FLAGS.clip_value > 0:
        torch.nn.utils.clip_grad_norm_(netD.parameters(), FLAGS.clip_value)
    optim_D.step()

    logger.add("training_d", "loss", loss_d.item(), i + 1)
    logger.add("training_d", "dreal", dreal.item(), i + 1)
    logger.add("training_d", "dfake", dfake.item(), i + 1)

    if (i + 1) % FLAGS.n_iter_d:
        sample_z = torch.randn(batch_size, FLAGS.g_z_dim).to(device)
        loss_g, dfake_g = loss_func_g(netD, netG, sample_z, label)
        optim_G.zero_grad()
        loss_g.backward()
        if FLAGS.clip_value > 0:
            torch.nn.utils.clip_grad_norm_(netG.parameters(), FLAGS.clip_value)
        optim_G.step()

        logger.add("training_g", "loss", loss_g.item(), i + 1)
        logger.add("training_g", "dfake", dfake_g.item(), i + 1)

    if (i + 1) % print_interval == 0:
        prefix = logger_prefix.format(i + 1, max_iter, (100 * i + 1) / max_iter)
        cats = ["training_d", "training_g"]
        logger.log_info(prefix, text_logger.info, cats=cats)

    if (i + 1) % image_interval == 0:
        with torch.no_grad():
            sample_z = torch.randn(100, FLAGS.g_z_dim).to(device)
            tlabel = label[:10]
            tlabel = torch.cat([tlabel for _ in range(10)], 0)
            x_fake = netG(sample_z, tlabel)
            logger.add_imgs(x_fake, "img{:08d}".format(i + 1), nrow=10)

    if (i + 1) % FLAGS.save_every == 0:
        logger.save_stats("ModelStat.pkl")
        file_name = "model" + str(i + 1) + ".pt"
        checkpoint_io.save(file_name)
