# todo 
# svhn and cifar10 generation
# augmentation for c
# augmentation for d
# tune parameters
# nets
# gan archs

# generation
python train_triplegan.py ./configs/triple_gan_cifar10_mt_aug_sngan.yaml -subfolder revision_2021_3_1 -ssl_seed 1234 -gpu 1

# Baseline
python train_classifier.py ./configs/classifier_stl10_mt_noaug.yaml -n_labels 1000 -num_label_per_batch 2 -c_model_name stl10_cnn -subfolder revision_2021_3_1/baseline -ssl_seed 1001 -gpu 0

# wide: 
python train_classifier.py ./configs/classifier_stl10_mt_noaug.yaml -n_labels 1000 -num_label_per_batch 2 -c_model_name stl10_cnn_wide -subfolder revision_2021_3_1/baseline -ssl_seed 1001 -gpu 3

# tune parameters
python train_classifier.py ./configs/classifier_stl10_mt_noaug.yaml -n_labels 1000 -num_label_per_batch 8 -c_model_name stl10_cnn -subfolder revision_2021_3_1/baseline -ssl_seed 1001 -gpu 4

python train_classifier.py ./configs/classifier_stl10_mt_noaug.yaml -n_labels 1000 -num_label_per_batch 20 -c_model_name stl10_cnn -subfolder revision_2021_3_1/baseline -ssl_seed 1001 -gpu 5

python train_classifier.py ./configs/classifier_stl10_mt_noaug.yaml -n_labels 1000 -num_label_per_batch 8 -c_model_name stl10_cnn_wide -subfolder revision_2021_3_1/baseline -ssl_seed 1001 -gpu 6

python train_classifier.py ./configs/classifier_stl10_mt_noaug.yaml -n_labels 1000 -num_label_per_batch 20 -c_model_name stl10_cnn_wide -subfolder revision_2021_3_1/baseline -ssl_seed 1001 -gpu 7

# augmentation

python train_classifier.py ./configs/classifier_stl10_mt_aug.yaml -n_labels 1000 -num_label_per_batch 8 -c_model_name stl10_cnn -translate 6 -subfolder revision_2021_3_1/baseline_aug -ssl_seed 1001 -gpu 0

python train_classifier.py ./configs/classifier_stl10_mt_aug.yaml -n_labels 1000 -num_label_per_batch 8 -c_model_name stl10_cnn -translate 12 -subfolder revision_2021_3_1/baseline_aug -ssl_seed 1001 -gpu 0


# 5000