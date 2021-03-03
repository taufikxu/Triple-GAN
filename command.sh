# todo 
# nets
# tune ramp up and max coefficient
# gan archs

# vgg: 
python train_classifier.py ./configs/classifier_stl10_mt_noaug.yaml -c_model_name stl10_cnn_vgg -subfolder revision_2021_3_1/baseline -ssl_seed 1001 -gpu 1

# tune model and lr

python train_classifier.py ./configs/classifier_stl10_mt_noaug.yaml -c_model_name stl10_cnn_vgg -subfolder revision_2021_3_1/baseline -ssl_seed 1001 -c_lr 0.0003 -gpu 2

python train_classifier.py ./configs/classifier_stl10_mt_noaug.yaml -c_model_name stl10_cnn_vgg -subfolder revision_2021_3_1/baseline -ssl_seed 1001 -c_lr 0.001 -gpu 3

python train_classifier.py ./configs/classifier_stl10_mt_noaug.yaml -c_model_name stl10_cnn -subfolder revision_2021_3_1/baseline -ssl_seed 1001 -c_lr 0.0003 -gpu 0

python train_classifier.py ./configs/classifier_stl10_mt_noaug.yaml -c_model_name stl10_cnn -subfolder revision_2021_3_1/baseline -ssl_seed 1001 -c_lr 0.001 -gpu 1

# tune rampup and max 

python train_classifier.py ./configs/classifier_stl10_mt_noaug.yaml -c_model_name stl10_cnn -subfolder revision_2021_3_1/baseline -ssl_seed 1001 -max_consistency_cost 100 -gpu 3


# augmentation

python train_classifier.py ./configs/classifier_stl10_mt_aug.yaml -c_model_name stl10_cnn_vgg -subfolder revision_2021_3_1/baseline_aug -ssl_seed 1001 -c_lr 0.0003 -gpu 2

python train_classifier.py ./configs/classifier_stl10_mt_aug.yaml -c_model_name stl10_cnn_vgg -subfolder revision_2021_3_1/baseline_aug -ssl_seed 1001 -c_lr 0.001 -gpu 3

python train_classifier.py ./configs/classifier_stl10_mt_aug.yaml -c_model_name stl10_cnn_vgg -subfolder revision_2021_3_1/baseline_aug -ssl_seed 1001 -c_lr 0.003 -gpu 0

python train_classifier.py ./configs/classifier_stl10_mt_aug.yaml -c_model_name stl10_cnn -subfolder revision_2021_3_1/baseline_aug -ssl_seed 1001 -c_lr 0.001 -gpu 1

python train_classifier.py ./configs/classifier_stl10_mt_aug.yaml -c_model_name stl10_cnn -subfolder revision_2021_3_1/baseline_aug -ssl_seed 1001 -c_lr 0.0003 -gpu 2


# 5000