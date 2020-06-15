import torch
import torch.nn as nn
import numpy as np

from library import inputs
from Utils.checkpoints import save_context, Logger
from Utils import flags
from Utils import config

import Torture
from library import evaluation
from library import loss_classifier
from mean_teacher import ramps

FLAGS = flags.FLAGS
KEY_ARGUMENTS = config.load_config(FLAGS.config_file)
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

checkpoint_io = Torture.utils.checkpoint.CheckpointIO(checkpoint_dir=MODELS_FOLDER)
checkpoint_io.register_modules(netC=netC)
logger = Logger(log_dir=SUMMARIES_FOLDER)
# train
print_interval = 50
test_interval = 500
max_iter = FLAGS.n_iter
loss_func = loss_classifier.loss_dict[FLAGS.c_loss]


# torch implementation 
# from mean_teacher import ramps
# def adjust_learning_rate(lr, epoch, step_in_epoch, total_steps_in_epoch):
#     epoch = epoch + step_in_epoch / total_steps_in_epoch

#     # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
#     lr = ramps.linear_rampup(epoch, FLAGS.lr_rampup) * (FLAGS.lr - FLAGS.initial_lr) + FLAGS.initial_lr

#     # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
#     if FLAGS.lr_rampdown_epochs:
#         assert FLAGS.lr_rampdown_epochs >= FLAGS.epochs
#         lr *= ramps.cosine_rampdown(epoch, FLAGS.lr_rampdown_epochs)
#     return lr

# def get_current_consistency_weight(epoch):
#     # Consistency ramp-up from https://arxiv.org/abs/1610.02242
#     return FLAGS.consistency * ramps.sigmoid_rampup(epoch, FLAGS.consistency_rampup)

# tf implementation
# model['rampdown_length'] = 0
# model['rampup_length'] = 5000
# model['training_length'] = 40000
# model['max_consistency_cost'] = 50.0

# DEFAULT_HYPERPARAMS = {
#         # Consistency hyperparameters
#         'ema_consistency': True,
#         'apply_consistency_to_labeled': True,
#         'max_consistency_cost': 100.0,
#         'ema_decay_during_rampup': 0.99,
#         'ema_decay_after_rampup': 0.999,
#         'consistency_trust': 0.0,
#         'num_logits': 1, # Either 1 or 2
#         'logit_distance_cost': 0.0, # Matters only with 2 outputs

#         # Optimizer hyperparameters
#         'max_learning_rate': 0.003,
#         'adam_beta_1_before_rampdown': 0.9,
#         'adam_beta_1_after_rampdown': 0.5,
#         'adam_beta_2_during_rampup': 0.99,
#         'adam_beta_2_after_rampup': 0.999,
#         'adam_epsilon': 1e-8,

#         # Architecture hyperparameters
#         'input_noise': 0.15,
#         'student_dropout_probability': 0.5,
#         'teacher_dropout_probability': 0.5,

#         # Training schedule
#         'rampup_length': 40000,
#         'rampdown_length': 25000,
#         'training_length': 150000,

#         # Input augmentation
#         'flip_horizontally': False,
#         'translate': True,

#         # Whether to scale each input image to mean=0 and std=1 per channel
#         # Use False if input is already normalized in some other way
#         'normalize_input': True,

#         # Output schedule
#         'print_span': 20,
#         'evaluation_span': 500,
#     }

# def step_rampup(global_step, rampup_length):
#     result = tf.cond(global_step < rampup_length,
#                      lambda: tf.constant(0.0),
#                      lambda: tf.constant(1.0))
#     return tf.identity(result, name="step_rampup")

# def sigmoid_rampup(global_step, rampup_length):
#     global_step = tf.to_float(global_step)
#     rampup_length = tf.to_float(rampup_length)
#     def ramp():
#         phase = 1.0 - tf.maximum(0.0, global_step) / rampup_length
#         return tf.exp(-5.0 * phase * phase)

#     result = tf.cond(global_step < rampup_length, ramp, lambda: tf.constant(1.0))
#     return tf.identity(result, name="sigmoid_rampup")

# def sigmoid_rampdown(global_step, rampdown_length, training_length):
#     global_step = tf.to_float(global_step)
#     rampdown_length = tf.to_float(rampdown_length)
#     training_length = tf.to_float(training_length)
#     def ramp():
#         phase = 1.0 - tf.maximum(0.0, training_length - global_step) / rampdown_length
#         return tf.exp(-12.5 * phase * phase)

#     result = tf.cond(global_step >= training_length - rampdown_length,
#                      ramp,
#                      lambda: tf.constant(1.0))
#     return tf.identity(result, name="sigmoid_rampdown")

# sigmoid_rampup_value = sigmoid_rampup(self.global_step, self.hyper['rampup_length'])
# sigmoid_rampdown_value = sigmoid_rampdown(self.global_step,
#                                           self.hyper['rampdown_length'],
#                                           self.hyper['training_length'])
# self.learning_rate = tf.multiply(sigmoid_rampup_value * sigmoid_rampdown_value,
#                                  self.hyper['max_learning_rate'],
#                                  name='learning_rate')
# self.adam_beta_1 = tf.add(sigmoid_rampdown_value * self.hyper['adam_beta_1_before_rampdown'],
#                           (1 - sigmoid_rampdown_value) * self.hyper['adam_beta_1_after_rampdown'],
#                           name='adam_beta_1')
# self.cons_coefficient = tf.multiply(sigmoid_rampup_value,
#                                     self.hyper['max_consistency_cost'],
#                                     name='consistency_coefficient')

# step_rampup_value = step_rampup(self.global_step, self.hyper['rampup_length'])
# self.adam_beta_2 = tf.add((1 - step_rampup_value) * self.hyper['adam_beta_2_during_rampup'],
#                           step_rampup_value * self.hyper['adam_beta_2_after_rampup'],
#                           name='adam_beta_2')
# self.ema_decay = tf.add((1 - step_rampup_value) * self.hyper['ema_decay_during_rampup'],
#                         step_rampup_value * self.hyper['ema_decay_after_rampup'],
#                         name='ema_decay')


for i in range(max_iter):
    data, label = itr.__next__()
    data, label = data.to(device), label.to(device)
    data_u, _ = itr_u.__next__()
    data_u = data_u.to(device)

    tloss = loss_func(netC, data, label, data_u)
    optim_c.zero_grad()
    tloss.backward()
    if FLAGS.clip_value > 0:
        torch.nn.utils.clip_grad_norm_(netC.parameters(), FLAGS.clip_value)
    optim_c.step()

    logger.add("training", "loss", tloss.item(), i + 1)

    if (i + 1) % test_interval == 0:
        total_t, correct_t, loss_t = evaluation.test_classifier(netC)
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
