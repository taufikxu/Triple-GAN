#  todo 
# augmentation for c
# augmentation for d
# tune parameters
# nets
# gan archs


# generation
python train_triplegan.py ./configs/triple_gan_cifar10_mt_aug_sngan.yaml -subfolder revision_2021_3_1 -ssl_seed 1234 -gpu 1

# Baseline
python train_classifier.py ./configs/classifier_stl10_mt_noaug.yaml -max_consistency_cost 50.0 -n_labels 1000 -num_label_per_batch 2 -c_model_name stl10_cnn -key noaug -subfolder revision_2021_3_1/baseline -ssl_seed 1001 -gpu 0

# no zca
python train_classifier.py ./configs/classifier_stl10_mt_noaug.yaml -max_consistency_cost 50.0 -n_labels 1000 -num_label_per_batch 2 -c_model_name stl10_cnn -key noaug -subfolder revision_2021_3_1/baseline -ssl_seed 1001 -zca false -gpu 2

# wide: 
python train_classifier.py ./configs/classifier_stl10_mt_noaug.yaml -max_consistency_cost 50.0 -n_labels 1000 -num_label_per_batch 2 -c_model_name stl10_cnn_wide -key noaug -subfolder revision_2021_3_1/baseline -ssl_seed 1001 -zca false -gpu 3

# augmentation

# tune parameters

# 5000