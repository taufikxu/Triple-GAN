import os
import multiprocessing
import time
import itertools

# Args
args_fortune = {
    "config_file": [
        "./configs/classifier_cifar10_mt_aug.yaml",
        "./configs/classifier_cifar10_mt_noaug.yaml",
        "./configs/classifier_svhn_mt_aug.yaml",
        "./configs/classifier_svhn_mt_noaug.yaml",
    ],
    # "alpha_c_pdl": [0.03, 0.1, 0.3],
    # "psl_iters": [150000],
    # "alpha_c_adv": [0.003, 0.01],
    # "adv_iters": [150000, 9999999],
    "subfolder": ["FinalBaseline"],
}
command_template = "python train_classifier.py"
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
gpus = multiprocessing.Manager().list([0, 1, 2, 3, 4, 5, 6, 7])
proc_to_gpu_map = multiprocessing.Manager().dict()


def exp_runner(cmd):
    process_id = multiprocessing.current_process().name
    if process_id not in proc_to_gpu_map:
        proc_to_gpu_map[process_id] = gpus.pop()
        print("assign gpu {} to {}".format(proc_to_gpu_map[process_id], process_id))
    return os.system(cmd + " -gpu {}".format(proc_to_gpu_map[process_id]))


p = multiprocessing.Pool(processes=len(gpus))
rets = p.map(exp_runner, commands)
print(rets)
