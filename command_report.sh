# xuan: g6&g8
python train_triplegan_final_elr.py ./configs/triple_gan_cifar10_noaug_elr.yaml -subfolder RE_CI -ssl_seed 1001 -translate 0 -flip_horizontal false -gpu 0
python train_triplegan_final_elr.py ./configs/triple_gan_cifar10_noaug_elr.yaml -subfolder RE_CI -ssl_seed 1002 -translate 0 -flip_horizontal false -gpu 1
python train_triplegan_final_elr.py ./configs/triple_gan_cifar10_noaug_elr.yaml -subfolder RE_CI -ssl_seed 1003 -translate 0 -flip_horizontal false -gpu 2
python train_triplegan_final_elr.py ./configs/triple_gan_cifar10_noaug_elr.yaml -subfolder RE_CI -ssl_seed 1001 -translate 2 -flip_horizontal true -gpu 3
python train_triplegan_final_elr.py ./configs/triple_gan_cifar10_noaug_elr.yaml -subfolder RE_CI -ssl_seed 1002 -translate 2 -flip_horizontal true -gpu 1
python train_triplegan_final_elr.py ./configs/triple_gan_cifar10_noaug_elr.yaml -subfolder RE_CI -ssl_seed 1003 -translate 2 -flip_horizontal true -gpu 0

