import torch
import torch.nn as nn
import numpy as np

from TripleGAN import inputs
from Utils.checkpoints import save_context, Logger
from Utils import flags
from Utils import config

import Torture
from TripleGAN import evaluation

FLAGS = flags.FLAGS
KEY_ARGUMENTS = config.load_config(FLAGS.config_file)
FILES_TO_BE_SAVED = ["./", "./configs", "./TripleGAN", "./TripleGAN/mean_teacher"]
CONFIG = {"FILES_TO_BE_SAVED": FILES_TO_BE_SAVED, "KEY_ARGUMENTS": KEY_ARGUMENTS}
text_logger, MODELS_FOLDER, SUMMARIES_FOLDER = save_context(__file__, CONFIG)

torch.manual_seed(1234)
# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(1235)
torch.cuda.manual_seed(1236)
device = Torture.device
logger = Logger(log_dir=SUMMARIES_FOLDER)

itr = inputs.get_data_iter()
netC, optim_c = inputs.get_classifier_optimizer()
netC = netC.to(device)
netEnetC = nn.DataParallel(netC)

checkpoint_io = Torture.utils.checkpoint.CheckpointIO(checkpoint_dir=MODELS_FOLDER)
checkpoint_io.register_modules(netC=netC)

# train
print_interval = 50
test_interval = 500
max_iter = FLAGS.n_iter
loss_func = nn.CrossEntropyLoss()

for i in range(max_iter):
    data, label = itr.__next__()
    data, label = data.to(device), label.to(device)

    label_pre = netC(data)
    optim_c.zero_grad()
    tloss = loss_func(label_pre, label)
    tloss.backward()
    if FLAGS.clip_value > 0:
        torch.nn.utils.clip_grad_norm_(netC.parameters(), FLAGS.clip_value)
    optim_c.step()

    logger.add("training", "loss", tloss.item(), i + 1)

    if (i + 1) % test_interval == 0:
        total_t, correct_t, loss_t = evaluation.test_classifier(netC, loss_func)
        str_meg = "Iteration {}/{} ({:.0f}%), train loss {:.5f}, test_loss {:.5f},"
        str_meg += " test: total tested {:05d}, corrected {:05d}, accuracy {:.5f}"
        text_logger.info(
            str_meg.format(
                i + 1,
                max_iter,
                100 * ((i + 1) / max_iter),
                tloss.item(),
                loss_t,
                total_t,
                correct_t,
                100 * (correct_t / total_t),
            )
        )
        logger.add("testing", "loss", loss_t.item(), i + 1)
        logger.add("testing", "accuracy", 100 * (correct_t / total_t), i + 1)
    elif (i + 1) % print_interval == 0:
        str_meg = "Iteration {}/{} ({:.0f}%), train loss {:.5f}"
        text_logger.info(
            str_meg.format(i + 1, max_iter, 100 * ((i + 1) / max_iter), tloss.item())
        )

    if (i + 1) % FLAGS.save_every == 0:
        logger.save_stats("{:08d}.pkl".format(i))
        file_name = "model" + str(i + 1) + ".pt"
        checkpoint_io.save(file_name)
