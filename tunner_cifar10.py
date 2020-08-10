import os
import multiprocessing
import time
import itertools

# Args
gpu_list = [0, 1, 2, 3]
args_fortune = {
    "TUNNER_groups": [
        "python train_classifier_elr.py ./configs/classifier_cifar10_mt_aug_elr.yaml -subfolder elr_baseline -n_labels 4000 -ssl_seed 1001 -translate 2 -flip_horizontal true -c_loss mtssl",
        "python train_classifier_elr.py ./configs/classifier_cifar10_mt_aug_elr.yaml -subfolder elr_baseline -n_labels 4000 -ssl_seed 1002 -translate 2 -flip_horizontal true -c_loss mtssl",
        "python train_classifier_elr.py ./configs/classifier_cifar10_mt_aug_elr.yaml -subfolder elr_baseline -n_labels 4000 -ssl_seed 1003 -translate 2 -flip_horizontal true -c_loss mtssl",
        "python train_classifier_elr.py ./configs/classifier_cifar10_mt_aug_elr.yaml -subfolder elr_baseline -n_labels 4000 -ssl_seed 1001 -translate 0 -flip_horizontal false -c_loss mtssl",
        "python train_classifier_elr.py ./configs/classifier_cifar10_mt_aug_elr.yaml -subfolder elr_baseline -n_labels 4000 -ssl_seed 1002 -translate 0 -flip_horizontal false -c_loss mtssl",
        "python train_classifier_elr.py ./configs/classifier_cifar10_mt_aug_elr.yaml -subfolder elr_baseline -n_labels 4000 -ssl_seed 1003 -translate 0 -flip_horizontal false -c_loss mtssl",
        "python train_classifier_elr.py ./configs/classifier_cifar10_mt_aug_elr.yaml -subfolder elr_baseline -n_labels 4000 -ssl_seed 1001 -translate 2 -flip_horizontal true -c_loss loss_elr_wrap",
        "python train_classifier_elr.py ./configs/classifier_cifar10_mt_aug_elr.yaml -subfolder elr_baseline -n_labels 4000 -ssl_seed 1002 -translate 2 -flip_horizontal true -c_loss loss_elr_wrap",
        "python train_classifier_elr.py ./configs/classifier_cifar10_mt_aug_elr.yaml -subfolder elr_baseline -n_labels 4000 -ssl_seed 1003 -translate 2 -flip_horizontal true -c_loss loss_elr_wrap ",
        "python train_classifier_elr.py ./configs/classifier_cifar10_mt_aug_elr.yaml -subfolder elr_baseline -n_labels 4000 -ssl_seed 1001 -translate 0 -flip_horizontal false -c_loss loss_elr_wrap ",
        "python train_classifier_elr.py ./configs/classifier_cifar10_mt_aug_elr.yaml -subfolder elr_baseline -n_labels 4000 -ssl_seed 1002 -translate 0 -flip_horizontal false -c_loss loss_elr_wrap ",
        "python train_classifier_elr.py ./configs/classifier_cifar10_mt_aug_elr.yaml -subfolder elr_baseline -n_labels 4000 -ssl_seed 1003 -translate 0 -flip_horizontal false -c_loss loss_elr_wrap ",
        "python train_triplegan_elr.py ./configs/triple_gan_cifar10_noaug_elr.yaml -subfolder elr_tgan -n_labels 4000 -ssl_seed 1001 -translate 2 -flip_horizontal true",
        "python train_triplegan_elr.py ./configs/triple_gan_cifar10_noaug_elr.yaml -subfolder elr_tgan -n_labels 4000 -ssl_seed 1002 -translate 2 -flip_horizontal true ",
        "python train_triplegan_elr.py ./configs/triple_gan_cifar10_noaug_elr.yaml -subfolder elr_tgan -n_labels 4000 -ssl_seed 1003 -translate 2 -flip_horizontal true ",
        "python train_triplegan_elr.py ./configs/triple_gan_cifar10_noaug_elr.yaml -subfolder elr_tgan -n_labels 4000 -ssl_seed 1001 -translate 0 -flip_horizontal false ",
        "python train_triplegan_elr.py ./configs/triple_gan_cifar10_noaug_elr.yaml -subfolder elr_tgan -n_labels 4000 -ssl_seed 1002 -translate 0 -flip_horizontal false ",
        "python train_triplegan_elr.py ./configs/triple_gan_cifar10_noaug_elr.yaml -subfolder elr_tgan -n_labels 4000 -ssl_seed 1003 -translate 0 -flip_horizontal false ",
    ],
    "subfolder": ["ELR_CIFAR10"],
}
command_template = ""
key_sequence = []
for k in args_fortune:
    key_sequence.append(k)
    if k == "config_file" or "TUNNER" in k:
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
