import os
import multiprocessing
import time
import itertools

# Args
args_fortune = {
    "config_file": [
        "./configs/triple_gan_cifar10_mt_aug_sngan.yaml",
        # "./configs/triple_gan_cifar10_mt_noaug_sngan.yaml",
    ],
    "n_iter_pretrain": [20000],
    "n_labels": [4000],
    "ssl_seed": [1001,1002],
    "teach_for_d": ["true"],
    "alpha_c_adv": [0.03],
    "alpha_c_pdl": [0.3, 3],
    "subfolder": ["tune_teacher"],
}
command_template = "python train_triplegan.py"
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
gpus = multiprocessing.Manager().list([0, 1, 2, 3])
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
