import os
import multiprocessing
import time
import itertools

# Args
# gpu_list = [0, 1, 2, 3]
# args_fortune = {
#     "config_file": [
#         "python train_classifier.py ./configs/classifier_svhn_mt_aug.yaml",
#         "python train_classifier.py ./configs/classifier_svhn_mt_noaug.yaml",
#         "python train_triplegan.py ./configs/triple_gan_svhn_mt_aug_sngan.yaml",
#         "python train_triplegan.py ./configs/triple_gan_svhn_mt_noaug_sngan.yaml",
#     ],
#     "alpha_c_pdl": [3.0],
#     "ssl_seed": [1001],
#     "n_labels": [100],
#     "num_label_per_batch": [1],
#     "subfolder": ["SVHN_FINAL100"],

gpu_list = [0, 1, 2]
args_fortune = {
    "config_file": [
        # "python train_classifier.py ./configs/classifier_svhn_mt_aug.yaml",
        # "python train_classifier.py ./configs/classifier_svhn_mt_noaug.yaml",
        # "python train_triplegan.py ./configs/triple_gan_svhn_mt_aug_sngan.yaml",
        "python train_triplegan_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml",
    ],
    "ssl_seed": [1001],
    "n_labels": [500, 800, 1000],
    "subfolder": ["svhn_elr"],
}
command_template = ""
key_sequence = []
for k in args_fortune:
    key_sequence.append(k)
    if k == "config_file":
        command_template += " {}"
    else:
        command_template += " -" + k + " {}"
print(command_template, key_sequence)


possible_value = []
for k in key_sequence:
    possible_value.append(args_fortune[k])
commands = []
for args in itertools.product(*possible_value):
    commands.append(command_template.format(*args))

print("# experiments = {}".format(len(commands)))
gpus = multiprocessing.Manager().list(gpu_list)
proc_to_gpu_map = multiprocessing.Manager().dict()


def exp_runner(cmd):
    process_id = multiprocessing.current_process().name
    if process_id not in proc_to_gpu_map:
        proc_to_gpu_map[process_id] = gpus.pop()
        print("assign gpu {} to {}".format(proc_to_gpu_map[process_id], process_id))
    gpuid = proc_to_gpu_map[process_id]
    return os.system(cmd + " -gpu {} -key {}".format(gpuid, gpuid))


p = multiprocessing.Pool(processes=len(gpus))
rets = p.map(exp_runner, commands)
print(rets)
