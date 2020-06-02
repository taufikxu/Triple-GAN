import torch
import torch.nn as nn
import numpy as np

from library import inputs
from Utils.checkpoints import save_context, Logger
from Utils import flags
from Utils import config

import Torture
from library import evaluation, gan_loss

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
print_interval = 50
image_interval = 500
max_iter = FLAGS.n_iter
batch_size = FLAGS.batch_size
loss_func_g = gan_loss.g_loss_dict[FLAGS.gan_type]
loss_func_d = gan_loss.d_loss_dict[FLAGS.gan_type]

for i in range(max_iter):
    x_real, label = itr.__next__()
    x_real, label = x_real.to(device), label.to(device)

    sample_z = torch.randn(batch_size, FLAGS.g_z_dim).to(device)
    x_fake = netG(sample_z, label).detach()
    loss_d, dreal, dfake = loss_func_d(netD, x_real, x_fake, label)
    optim_D.zero_grad()
    loss_d.backward()
    if FLAGS.clip_value > 0:
        torch.nn.utils.clip_grad_norm_(netD.parameters(), FLAGS.clip_value)
    optim_D.step()

    logger.add("training_d", "loss", loss_d.item(), i + 1)
    logger.add("training_d", "dreal", dreal.item(), i + 1)
    logger.add("training_d", "dfake", dfake.item(), i + 1)

    sample_z = torch.randn(batch_size, FLAGS.g_z_dim).to(device)
    x_fake = netG(sample_z, label)
    loss_g, dfake_g = loss_func_g(netD, x_fake, label)
    optim_G.zero_grad()
    loss_g.backward()
    if FLAGS.clip_value > 0:
        torch.nn.utils.clip_grad_norm_(netG.parameters(), FLAGS.clip_value)
    optim_G.step()

    logger.add("training_g", "loss", loss_g.item(), i + 1)
    logger.add("training_g", "dfake", dfake_g.item(), i + 1)

    if (i + 1) % print_interval == 0:
        str_meg = "Iteration {}/{} ({:.0f}%), D: loss {:.5f}, real/fake {:.5f}/{:.5f}"
        str_meg += " G: loss {:.5f}, dfake {:.5f}"
        str_meg = str_meg.format(
            i + 1,
            max_iter,
            100 * ((i + 1) / max_iter),
            loss_d.item(),
            dreal.item(),
            dfake.item(),
            loss_g.item(),
            dfake_g.item(),
        )
        text_logger.info(str_meg)

    if (i + 1) % image_interval == 0:
        with torch.no_grad():
            sample_z = torch.randn(batch_size, FLAGS.g_z_dim).to(device)
            x_fake = netG(sample_z, label)
            logger.add_imgs(x_fake, "img{:08d}".format(i + 1))

    if (i + 1) % FLAGS.save_every == 0:
        logger.save_stats("{:08d}.pkl".format(i))
        file_name = "model" + str(i + 1) + ".pt"
        checkpoint_io.save(file_name)
