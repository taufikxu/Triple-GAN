import os
import multiprocessing
import time
import itertools
import subprocess

WORK_SPACE = "~/Workspace/Triple-GAN"
PYTHON_PATH = "/home/kunxu/ENV/envs/torch/bin/python"
jungpus = [10, 12, 13, 21, 23]
# Args
args_fortune = {
    "config_file": [
        "./configs/classifier_cifar10_mt_aug.yaml",
        "./configs/classifier_cifar10_mt_noaug.yaml",
    ],
    "ssl_seed": [1001, 1002, 1003],
    "subfolder": ["AverageBaseline"],
    "TUNNER_groups": [
        "-n_labels 4000 -num_label_per_batch 8",
        "-n_labels 2000 -num_label_per_batch 4",
        "-n_labels 1000 -num_label_per_batch 2",
    ],
}
command_template = "cd {};{} train_classifier.py".format(WORK_SPACE, PYTHON_PATH)
key_sequence = []
for k in args_fortune:
    key_sequence.append(k)
    if k == "config_file" or "TUNNER_groups" in k:
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


gpus = []
get_gpu_command = '"import gpustat; q = gpustat.GPUStatCollection.new_query(); usable = []; [usable.append(i) for i in range(len(q)) if len(q[i].processes) == 0]; print(usable)"'
for server in jungpus:
    tcmd = "ssh g{} '{} -c {}'".format(server, PYTHON_PATH, get_gpu_command)
    usable = eval(subprocess.getoutput(tcmd))
    for g in usable:
        gpus.append((server, g))


print("# experiments = {}, # GPUS = {}".format(len(commands), len(gpus)))
print(gpus)
assert len(commands) <= len(gpus)
gpus = multiprocessing.Manager().list(gpus)
proc_to_gpu_map = multiprocessing.Manager().dict()


def exp_runner(cmd):
    # print(cmd)
    process_id = multiprocessing.current_process().name
    if process_id not in proc_to_gpu_map:
        proc_to_gpu_map[process_id] = gpus.pop()
        print("assign gpu {} to {}".format(proc_to_gpu_map[process_id], process_id))
    server, gpuid = proc_to_gpu_map[process_id]
    return os.system(
        "ssh g{} '".format(server)
        + cmd
        + " -gpu {} -key {}'".format(gpuid, str(gpuid) + str(server))
    )


p = multiprocessing.Pool(processes=len(gpus))
rets = p.map(exp_runner, commands)
print(rets)

