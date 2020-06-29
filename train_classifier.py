import torch
import torch.nn as nn
import numpy as np
import copy

from library import inputs
from Utils.checkpoints import save_context, Logger
from Utils import flags
from Utils import config

import Torture
from library import evaluation
from library import loss_cla as loss_classifier

# from mean_teacher import ramps

FLAGS = flags.FLAGS
KEY_ARGUMENTS = config.load_config(FLAGS.config_file)
# KEY_ARGUMENTS = ["translate"] + KEY_ARGUMENTS
text_logger, MODELS_FOLDER, SUMMARIES_FOLDER = save_context(__file__, KEY_ARGUMENTS)

torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(1235)
torch.cuda.manual_seed(1236)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

itr = inputs.get_data_iter(subset=FLAGS.n_labels)
itr_u = inputs.get_data_iter()
netC, optim_c = inputs.get_classifier_optimizer()
netC = netC.to(device)
netC = nn.DataParallel(netC)
netC_T, _ = inputs.get_classifier_optimizer()
netC_T = netC_T.to(device)
netC_T = nn.DataParallel(netC_T)
netC.train()
netC_T.train()
Torture.update_average(netC_T, netC, 0)
for p in netC_T.parameters():
    p.requires_grad_(False)


checkpoint_io = Torture.utils.checkpoint.CheckpointIO(checkpoint_dir=MODELS_FOLDER)
checkpoint_io.register_modules(netC=netC, netC_T=netC_T)
logger = Logger(log_dir=SUMMARIES_FOLDER)
# train
print_interval = 50
test_interval = 500
max_iter = FLAGS.n_iter
loss_func = loss_classifier.c_loss_dict[FLAGS.c_loss]
step_func = loss_classifier.c_step_func[FLAGS.c_step]

logger_prefix = "Itera {}/{} ({:.0f}%)"
for i in range(max_iter):
    tloss, l_loss, u_loss = loss_func(netC, netC_T, i, itr, itr_u, device)
    step_func(optim_c, netC, netC_T, i, tloss)

    logger.add("training", "loss", tloss.item(), i + 1)
    logger.add("training", "l_loss", l_loss.item(), i + 1)
    logger.add("training", "u_loss", u_loss.item(), i + 1)

    if (i + 1) % print_interval == 0:
        prefix = logger_prefix.format(i + 1, max_iter, (100 * i + 1) / max_iter)
        cats = ["training"]
        logger.log_info(prefix, text_logger.info, cats=cats)

    if (i + 1) % test_interval == 0:
        netC.eval()
        netC_T.eval()
        total_t, correct_t, loss_t = evaluation.test_classifier(netC_T)
        logger.add("testing", "loss_t", loss_t.item(), i + 1)
        logger.add("testing", "accuracy_t", 100 * (correct_t / total_t), i + 1)

        total_t, correct_t, loss_t = evaluation.test_classifier(netC)
        logger.add("testing", "loss", loss_t.item(), i + 1)
        logger.add("testing", "accuracy", 100 * (correct_t / total_t), i + 1)

        prefix = logger_prefix.format(i + 1, max_iter, (100 * i + 1) / max_iter)
        cats = ["testing"]
        logger.log_info(prefix, text_logger.info, cats=cats)
        netC.train()
        netC_T.train()

    if (i + 1) % FLAGS.save_every == 0:
        logger.save_stats("{:08d}.pkl".format(i))
        file_name = "model" + str(i + 1) + ".pt"
        checkpoint_io.save(file_name)
