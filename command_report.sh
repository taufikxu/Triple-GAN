# ablation
# generation 

# clean up and report

# RE_VI and TI 100000 collect
# SVHN 4000, 500 collect

# TINY 1000 collect
# zca
# 30000-50000
# 50000-70000

# alpha 0.01 / 0.3

# rerun ssl

# tiny baseline
# python train_classifier_elr.py ./configs/classifier_tinyimagenet_elr.yaml -subfolder RE_TI_1000 -n_labels 1000 -translate 0 -rampup_length 15000 -rampup_length_lr 15000 -flip_horizontal false -ssl_seed 1001 -gpu 0
# python train_classifier_elr.py ./configs/classifier_tinyimagenet_elr.yaml -subfolder RE_TI_1000 -n_labels 1000 -translate 0 -rampup_length 15000 -rampup_length_lr 15000 -flip_horizontal false -ssl_seed 1002 -gpu 1
# python train_classifier_elr.py ./configs/classifier_tinyimagenet_elr.yaml -subfolder RE_TI_1000 -n_labels 1000 -translate 0 -rampup_length 15000 -rampup_length_lr 15000 -flip_horizontal false -ssl_seed 1003 -gpu 0
# python train_classifier_elr.py ./configs/classifier_tinyimagenet_elr.yaml -subfolder RE_TI_1000 -n_labels 1000 -translate 2 -rampup_length 15000 -rampup_length_lr 15000 -flip_horizontal true -ssl_seed 1001 -gpu 1
# python train_classifier_elr.py ./configs/classifier_tinyimagenet_elr.yaml -subfolder RE_TI_1000 -n_labels 1000 -translate 2 -rampup_length 15000 -rampup_length_lr 15000 -flip_horizontal true -ssl_seed 1002 -gpu 1
# python train_classifier_elr.py ./configs/classifier_tinyimagenet_elr.yaml -subfolder RE_TI_1000 -n_labels 1000 -translate 2 -rampup_length 15000 -rampup_length_lr 15000 -flip_horizontal true -ssl_seed 1003 -gpu 1


# python train_classifier_elr.py ./configs/classifier_tinyimagenet_elr.yaml -subfolder RE_TI_1000 -n_labels 1000 -zca true -translate 0 -rampup_length 15000 -rampup_length_lr 15000 -flip_horizontal false -ssl_seed 1001 -gpu 0
# python train_classifier_elr.py ./configs/classifier_tinyimagenet_elr.yaml -subfolder RE_TI_1000 -n_labels 1000 -zca true -translate 0 -rampup_length 15000 -rampup_length_lr 15000 -flip_horizontal false -ssl_seed 1002 -gpu 1
# python train_classifier_elr.py ./configs/classifier_tinyimagenet_elr.yaml -subfolder RE_TI_1000 -n_labels 1000 -zca true -translate 0 -rampup_length 15000 -rampup_length_lr 15000 -flip_horizontal false -ssl_seed 1003 -gpu 0
# python train_classifier_elr.py ./configs/classifier_tinyimagenet_elr.yaml -subfolder RE_TI_1000 -n_labels 1000 -zca true -translate 2 -rampup_length 15000 -rampup_length_lr 15000 -flip_horizontal true -ssl_seed 1001 -gpu 1
# python train_classifier_elr.py ./configs/classifier_tinyimagenet_elr.yaml -subfolder RE_TI_1000 -n_labels 1000 -zca true -translate 2 -rampup_length 15000 -rampup_length_lr 15000 -flip_horizontal true -ssl_seed 1002 -gpu 1
# python train_classifier_elr.py ./configs/classifier_tinyimagenet_elr.yaml -subfolder RE_TI_1000 -n_labels 1000 -zca true -translate 2 -rampup_length 15000 -rampup_length_lr 15000 -flip_horizontal true -ssl_seed 1003 -gpu 1


# g12 - g25
# svhn -alpha_adv 0.03 
# tiny 
# python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder RE_TI_1000 -translate 2 -flip_horizontal true -zca false -alpha_c_pdl 0.01 -pdl_ramp_start 30000 -pdl_ramp_end 50000 -adv_ramp_start 30000 -adv_ramp_end 50000 -rampup_length 15000 -rampup_length_lr 15000 -n_labels 1000 -ssl_seed 1001 -gpu 0
# python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder RE_TI_1000 -translate 2 -flip_horizontal true -zca false -alpha_c_pdl 0.03 -pdl_ramp_start 50000 -pdl_ramp_end 70000 -adv_ramp_start 50000 -adv_ramp_end 70000 -rampup_length 15000 -rampup_length_lr 15000 -n_labels 1000 -ssl_seed 1001 -gpu 1
# python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder RE_TI_1000 -translate 2 -flip_horizontal true -zca false -alpha_c_pdl 0.01 -pdl_ramp_start 30000 -pdl_ramp_end 50000 -adv_ramp_start 30000 -adv_ramp_end 50000 -rampup_length 15000 -rampup_length_lr 15000 -n_labels 1000 -ssl_seed 1001 -gpu 2
# python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder RE_TI_1000 -translate 2 -flip_horizontal true -zca false -alpha_c_pdl 0.03 -pdl_ramp_start 50000 -pdl_ramp_end 70000 -adv_ramp_start 50000 -adv_ramp_end 70000 -rampup_length 15000 -rampup_length_lr 15000 -n_labels 1000 -ssl_seed 1001 -gpu 3


# python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder RE_TI_1000 -translate 2 -flip_horizontal true -zca false -alpha_c_pdl 0.01 -alpha_c_adv 0.01 -pdl_ramp_start 30000 -pdl_ramp_end 50000 -adv_ramp_start 30000 -adv_ramp_end 50000 -rampup_length 15000 -rampup_length_lr 15000 -n_labels 1000 -ssl_seed 1001 -gpu 2
# python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder RE_TI_1000 -translate 2 -flip_horizontal true -zca false -alpha_c_pdl 0.03 -alpha_c_adv 0.01 -pdl_ramp_start 50000 -pdl_ramp_end 70000 -adv_ramp_start 50000 -adv_ramp_end 70000 -rampup_length 15000 -rampup_length_lr 15000 -n_labels 1000 -ssl_seed 1001 -gpu 3
# python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder RE_TI_1000 -translate 2 -flip_horizontal true -zca false -alpha_c_pdl 0.01 -alpha_c_adv 0.01 -pdl_ramp_start 30000 -pdl_ramp_end 50000 -adv_ramp_start 30000 -adv_ramp_end 50000 -rampup_length 15000 -rampup_length_lr 15000 -n_labels 1000 -ssl_seed 1001 -gpu 4
# python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder RE_TI_1000 -translate 2 -flip_horizontal true -zca false -alpha_c_pdl 0.03 -alpha_c_adv 0.01 -pdl_ramp_start 50000 -pdl_ramp_end 70000 -adv_ramp_start 50000 -adv_ramp_end 70000 -rampup_length 15000 -rampup_length_lr 15000 -n_labels 1000 -ssl_seed 1001 -gpu 6


# python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder RE_TI_1000 -translate 2 -flip_horizontal true -zca true  -alpha_c_pdl  0.1 -pdl_ramp_start 30000 -pdl_ramp_end 50000 -adv_ramp_start 30000 -adv_ramp_end 50000 -rampup_length 15000 -rampup_length_lr 15000 -n_labels 1000 -ssl_seed 1001 -gpu 7
# python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder RE_TI_1000 -translate 2 -flip_horizontal true -zca true  -alpha_c_pdl  0.3 -pdl_ramp_start 50000 -pdl_ramp_end 70000 -adv_ramp_start 50000 -adv_ramp_end 70000 -rampup_length 15000 -rampup_length_lr 15000 -n_labels 1000 -ssl_seed 1001 -gpu 0
# python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder RE_TI_1000 -translate 2 -flip_horizontal true -zca true  -alpha_c_pdl  0.1 -pdl_ramp_start 30000 -pdl_ramp_end 50000 -adv_ramp_start 30000 -adv_ramp_end 50000 -rampup_length 15000 -rampup_length_lr 15000 -n_labels 1000 -ssl_seed 1001 -gpu 1
# python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder RE_TI_1000 -translate 2 -flip_horizontal true -zca true  -alpha_c_pdl  0.3 -pdl_ramp_start 50000 -pdl_ramp_end 70000 -adv_ramp_start 50000 -adv_ramp_end 70000 -rampup_length 15000 -rampup_length_lr 15000 -n_labels 1000 -ssl_seed 1001 -gpu 2

# python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder RE_TI_1000 -translate 2 -flip_horizontal true -zca false -alpha_c_pdl 0.01 -pdl_ramp_start 30000 -pdl_ramp_end 50000 -adv_ramp_start 30000 -adv_ramp_end 50000 -rampup_length 15000 -rampup_length_lr 15000 -n_labels 1000 -ssl_seed 1002 -gpu 3
# python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder RE_TI_1000 -translate 2 -flip_horizontal true -zca false -alpha_c_pdl 0.03 -pdl_ramp_start 50000 -pdl_ramp_end 70000 -adv_ramp_start 50000 -adv_ramp_end 70000 -rampup_length 15000 -rampup_length_lr 15000 -n_labels 1000 -ssl_seed 1002 -gpu 4
# python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder RE_TI_1000 -translate 2 -flip_horizontal true -zca false -alpha_c_pdl 0.01 -pdl_ramp_start 30000 -pdl_ramp_end 50000 -adv_ramp_start 30000 -adv_ramp_end 50000 -rampup_length 15000 -rampup_length_lr 15000 -n_labels 1000 -ssl_seed 1002 -gpu 5
# python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder RE_TI_1000 -translate 2 -flip_horizontal true -zca false -alpha_c_pdl 0.03 -pdl_ramp_start 50000 -pdl_ramp_end 70000 -adv_ramp_start 50000 -adv_ramp_end 70000 -rampup_length 15000 -rampup_length_lr 15000 -n_labels 1000 -ssl_seed 1002 -gpu 6
# python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder RE_TI_1000 -translate 2 -flip_horizontal true -zca true  -alpha_c_pdl  0.1 -pdl_ramp_start 30000 -pdl_ramp_end 50000 -adv_ramp_start 30000 -adv_ramp_end 50000 -rampup_length 15000 -rampup_length_lr 15000 -n_labels 1000 -ssl_seed 1002 -gpu 7
# python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder RE_TI_1000 -translate 2 -flip_horizontal true -zca true  -alpha_c_pdl  0.3 -pdl_ramp_start 50000 -pdl_ramp_end 70000 -adv_ramp_start 50000 -adv_ramp_end 70000 -rampup_length 15000 -rampup_length_lr 15000 -n_labels 1000 -ssl_seed 1002 -gpu 0
# python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder RE_TI_1000 -translate 2 -flip_horizontal true -zca true  -alpha_c_pdl  0.1 -pdl_ramp_start 30000 -pdl_ramp_end 50000 -adv_ramp_start 30000 -adv_ramp_end 50000 -rampup_length 15000 -rampup_length_lr 15000 -n_labels 1000 -ssl_seed 1002 -gpu 1
# python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder RE_TI_1000 -translate 2 -flip_horizontal true -zca true  -alpha_c_pdl  0.3 -pdl_ramp_start 50000 -pdl_ramp_end 70000 -adv_ramp_start 50000 -adv_ramp_end 70000 -rampup_length 15000 -rampup_length_lr 15000 -n_labels 1000 -ssl_seed 1002 -gpu 2

# python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder RE_TI_1000 -translate 2 -flip_horizontal true -zca false -alpha_c_pdl 0.01 -pdl_ramp_start 30000 -pdl_ramp_end 50000 -adv_ramp_start 30000 -adv_ramp_end 50000 -rampup_length 15000 -rampup_length_lr 15000 -n_labels 1000 -ssl_seed 1003 -gpu 3
# python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder RE_TI_1000 -translate 2 -flip_horizontal true -zca false -alpha_c_pdl 0.03 -pdl_ramp_start 50000 -pdl_ramp_end 70000 -adv_ramp_start 50000 -adv_ramp_end 70000 -rampup_length 15000 -rampup_length_lr 15000 -n_labels 1000 -ssl_seed 1003 -gpu 4
# python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder RE_TI_1000 -translate 2 -flip_horizontal true -zca false -alpha_c_pdl 0.01 -pdl_ramp_start 30000 -pdl_ramp_end 50000 -adv_ramp_start 30000 -adv_ramp_end 50000 -rampup_length 15000 -rampup_length_lr 15000 -n_labels 1000 -ssl_seed 1003 -gpu 5
# python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder RE_TI_1000 -translate 2 -flip_horizontal true -zca false -alpha_c_pdl 0.03 -pdl_ramp_start 50000 -pdl_ramp_end 70000 -adv_ramp_start 50000 -adv_ramp_end 70000 -rampup_length 15000 -rampup_length_lr 15000 -n_labels 1000 -ssl_seed 1003 -gpu 6
# python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder RE_TI_1000 -translate 2 -flip_horizontal true -zca true  -alpha_c_pdl  0.1 -pdl_ramp_start 30000 -pdl_ramp_end 50000 -adv_ramp_start 30000 -adv_ramp_end 50000 -rampup_length 15000 -rampup_length_lr 15000 -n_labels 1000 -ssl_seed 1003 -gpu 7
# python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder RE_TI_1000 -translate 2 -flip_horizontal true -zca true  -alpha_c_pdl  0.3 -pdl_ramp_start 50000 -pdl_ramp_end 70000 -adv_ramp_start 50000 -adv_ramp_end 70000 -rampup_length 15000 -rampup_length_lr 15000 -n_labels 1000 -ssl_seed 1003 -gpu 0
# python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder RE_TI_1000 -translate 2 -flip_horizontal true -zca true  -alpha_c_pdl  0.1 -pdl_ramp_start 30000 -pdl_ramp_end 50000 -adv_ramp_start 30000 -adv_ramp_end 50000 -rampup_length 15000 -rampup_length_lr 15000 -n_labels 1000 -ssl_seed 1003 -gpu 1
# python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder RE_TI_1000 -translate 2 -flip_horizontal true -zca true  -alpha_c_pdl  0.3 -pdl_ramp_start 50000 -pdl_ramp_end 70000 -adv_ramp_start 50000 -adv_ramp_end 70000 -rampup_length 15000 -rampup_length_lr 15000 -n_labels 1000 -ssl_seed 1003 -gpu 2


# report svhn 500
# tgan svhn
# python train_triplegan_final_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder RE_VN_500 -translate 2 -n_labels 500 -ssl_seed 1001 -gpu 2
# python train_triplegan_final_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder RE_VN_500 -translate 2 -n_labels 500 -ssl_seed 1002 -gpu 3
# python train_triplegan_final_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder RE_VN_500 -translate 2 -n_labels 500 -ssl_seed 1003 -gpu 2

# python train_triplegan_final_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder RE_VN_500 -translate 0 -n_labels 500 -ssl_seed 1001 -gpu 5
# python train_triplegan_final_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder RE_VN_500 -translate 0 -n_labels 500 -ssl_seed 1002 -gpu 6
# python train_triplegan_final_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder RE_VN_500 -translate 0 -n_labels 500 -ssl_seed 1003 -gpu 7

# baseline svhn
# python train_classifier_elr.py ./configs/classifier_svhn_elr.yaml -subfolder RE_VN_500 -n_labels 500 -translate 2 -ssl_seed 1001 -gpu 0
# python train_classifier_elr.py ./configs/classifier_svhn_elr.yaml -subfolder RE_VN_500 -n_labels 500 -translate 2 -ssl_seed 1002 -gpu 1
# python train_classifier_elr.py ./configs/classifier_svhn_elr.yaml -subfolder RE_VN_500 -n_labels 500 -translate 2 -ssl_seed 1003 -gpu 2

# python train_classifier_elr.py ./configs/classifier_svhn_elr.yaml -subfolder RE_VN_500 -n_labels 500 -translate 0 -ssl_seed 1001 -gpu 3
# python train_classifier_elr.py ./configs/classifier_svhn_elr.yaml -subfolder RE_VN_500 -n_labels 500 -translate 0 -ssl_seed 1002 -gpu 5
# python train_classifier_elr.py ./configs/classifier_svhn_elr.yaml -subfolder RE_VN_500 -n_labels 500 -translate 0 -ssl_seed 1003 -gpu 0


# ablation kun g35&g26; xuan g4
python train_classifier.py ./configs/classifier_ablation_cifar10.yaml -subfolder ABLATION -n_labels 4000 -ssl_seed 1001 -translate 2 -flip_horizontal true -c_step regular -c_loss loss_elr_wrap -gpu 1
python train_classifier.py ./configs/classifier_ablation_cifar10.yaml -subfolder ABLATION -n_labels 4000 -ssl_seed 1001 -translate 2 -flip_horizontal true -c_step regular -c_loss entropyssl -gpu 0

python train_classifier.py ./configs/classifier_ablation_cifar10.yaml -subfolder ABLATION -n_labels 4000 -ssl_seed 1001 -translate 0 -flip_horizontal false -c_step regular -c_loss loss_elr_wrap -gpu 1
python train_classifier.py ./configs/classifier_ablation_cifar10.yaml -subfolder ABLATION -n_labels 4000 -ssl_seed 1001 -translate 0 -flip_horizontal false -c_step regular -c_loss entropyssl -gpu 0


python train_triplegan.py ./configs/triple_gan_cifar10_ablation.yaml -subfolder ABLATION -n_labels 4000 -ssl_seed 1001 -translate 2 -flip_horizontal true -alpha_c_pdl 3.0 -c_step regular -c_loss entropyssl -gpu 1
python train_triplegan.py ./configs/triple_gan_cifar10_ablation.yaml -subfolder ABLATION -n_labels 4000 -ssl_seed 1001 -translate 0 -flip_horizontal false -alpha_c_pdl 3.0 -c_step regular -c_loss entropyssl -gpu 2

python train_triplegan.py ./configs/triple_gan_cifar10_ablation.yaml -subfolder ABLATION -n_labels 4000 -ssl_seed 1001 -translate 2 -flip_horizontal true -alpha_c_pdl 3.0 -c_step regular -c_loss loss_elr_wrap -gpu 3
python train_triplegan.py ./configs/triple_gan_cifar10_ablation.yaml -subfolder ABLATION -n_labels 4000 -ssl_seed 1001 -translate 0 -flip_horizontal false -alpha_c_pdl 3.0 -c_step regular -c_loss loss_elr_wrap -gpu 4


python train_triplegan.py ./configs/triple_gan_cifar10_ablation.yaml -subfolder ABLATION -n_labels 4000 -ssl_seed 1001 -translate 2 -flip_horizontal true -alpha_c_pdl 1.0 -gpu 5
python train_triplegan.py ./configs/triple_gan_cifar10_ablation.yaml -subfolder ABLATION -n_labels 4000 -ssl_seed 1001 -translate 0 -flip_horizontal false -alpha_c_pdl 1.0 -gpu 6

python train_triplegan.py ./configs/triple_gan_cifar10_ablation.yaml -subfolder ABLATION -n_labels 4000 -ssl_seed 1001 -translate 2 -flip_horizontal true -alpha_c_pdl 10.0 -gpu 7
python train_triplegan.py ./configs/triple_gan_cifar10_ablation.yaml -subfolder ABLATION -n_labels 4000 -ssl_seed 1001 -translate 0 -flip_horizontal false -alpha_c_pdl 10.0 -gpu 3

python train_triplegan.py ./configs/triple_gan_cifar10_ablation.yaml -subfolder ABLATION -n_labels 4000 -ssl_seed 1001 -translate 2 -flip_horizontal true -alpha_c_pdl 0.0 -gpu 0
python train_triplegan.py ./configs/triple_gan_cifar10_ablation.yaml -subfolder ABLATION -n_labels 4000 -ssl_seed 1001 -translate 0 -flip_horizontal false -alpha_c_pdl 0.0 -gpu 0












# report svhn
# tgan svhn
python train_triplegan_final_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder RE_VN -translate 2 -alpha_c_pdl 0.01 -n_labels 1000 -ssl_seed 1002 -gpu 1
python train_triplegan_final_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder RE_VN -translate 2 -alpha_c_pdl 0.01 -n_labels 1000 -ssl_seed 1003 -gpu 2

python train_triplegan_final_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder RE_VN -translate 0 -alpha_c_pdl 0.01 -n_labels 1000 -ssl_seed 1002 -gpu 3
python train_triplegan_final_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder RE_VN -translate 0 -alpha_c_pdl 0.01 -n_labels 1000 -ssl_seed 1003 -gpu 4

# baseline svhn 
# xuan g4&11
python train_classifier_elr.py ./configs/classifier_svhn_elr.yaml -subfolder 819_D_VN -n_labels 1000 -translate 2 -ssl_seed 1001 -gpu 0
python train_classifier_elr.py ./configs/classifier_svhn_elr.yaml -subfolder 819_D_VN -n_labels 1000 -translate 2 -ssl_seed 1002 -gpu 1
python train_classifier_elr.py ./configs/classifier_svhn_elr.yaml -subfolder 819_D_VN -n_labels 1000 -translate 2 -ssl_seed 1003 -gpu 2
python train_classifier_elr.py ./configs/classifier_svhn_elr.yaml -subfolder 819_D_VN -n_labels 1000 -translate 0 -ssl_seed 1001 -gpu 3
python train_classifier_elr.py ./configs/classifier_svhn_elr.yaml -subfolder 819_D_VN -n_labels 1000 -translate 0 -ssl_seed 1002 -gpu 0
python train_classifier_elr.py ./configs/classifier_svhn_elr.yaml -subfolder 819_D_VN -n_labels 1000 -translate 0 -ssl_seed 1003 -gpu 0

# cifar10
# baseline
python train_classifier_elr.py ./configs/classifier_cifar10_elr.yaml -subfolder RE_CI -n_labels 4000 -translate 2 -flip_horizontal true -ssl_seed 1001 -gpu 0
python train_classifier_elr.py ./configs/classifier_cifar10_elr.yaml -subfolder RE_CI -n_labels 4000 -translate 2 -flip_horizontal true -ssl_seed 1002 -gpu 1
python train_classifier_elr.py ./configs/classifier_cifar10_elr.yaml -subfolder RE_CI -n_labels 4000 -translate 2 -flip_horizontal true -ssl_seed 1003 -gpu 0
python train_classifier_elr.py ./configs/classifier_cifar10_elr.yaml -subfolder RE_CI -n_labels 4000 -translate 0 -flip_horizontal false -ssl_seed 1001 -gpu 1
python train_classifier_elr.py ./configs/classifier_cifar10_elr.yaml -subfolder RE_CI -n_labels 4000 -translate 0 -flip_horizontal false -ssl_seed 1002 -gpu 2
python train_classifier_elr.py ./configs/classifier_cifar10_elr.yaml -subfolder RE_CI -n_labels 4000 -translate 0 -flip_horizontal false -ssl_seed 1003 -gpu 3
# tgan
python train_triplegan_final_elr.py ./configs/triple_gan_cifar10_noaug_elr.yaml -subfolder RE_CI -ssl_seed 1001 -translate 0 -flip_horizontal false -gpu 0
python train_triplegan_final_elr.py ./configs/triple_gan_cifar10_noaug_elr.yaml -subfolder RE_CI -ssl_seed 1002 -translate 0 -flip_horizontal false -gpu 1
python train_triplegan_final_elr.py ./configs/triple_gan_cifar10_noaug_elr.yaml -subfolder RE_CI -ssl_seed 1003 -translate 0 -flip_horizontal false -gpu 2
python train_triplegan_final_elr.py ./configs/triple_gan_cifar10_noaug_elr.yaml -subfolder RE_CI -ssl_seed 1001 -translate 2 -flip_horizontal true -gpu 3
python train_triplegan_final_elr.py ./configs/triple_gan_cifar10_noaug_elr.yaml -subfolder RE_CI -ssl_seed 1002 -translate 2 -flip_horizontal true -gpu 1
python train_triplegan_final_elr.py ./configs/triple_gan_cifar10_noaug_elr.yaml -subfolder RE_CI -ssl_seed 1003 -translate 2 -flip_horizontal true -gpu 0