# baseline 1000
python train_classifier.py ./configs/classifier_stl10_mt_noaug.yaml -subfolder revision_2021_3_1/baseline -ssl_seed 1001 -gpu 0

python train_classifier.py ./configs/classifier_stl10_mt_noaug.yaml -subfolder revision_2021_3_1/baseline -ssl_seed 1002 -gpu 1

python train_classifier.py ./configs/classifier_stl10_mt_noaug.yaml -subfolder revision_2021_3_1/baseline -ssl_seed 1003 -gpu 2

python train_classifier.py ./configs/classifier_stl10_mt_aug.yaml -subfolder revision_2021_3_1/baseline_aug -ssl_seed 1001 -gpu 0

python train_classifier.py ./configs/classifier_stl10_mt_aug.yaml -subfolder revision_2021_3_1/baseline_aug -ssl_seed 1002 -gpu 0

python train_classifier.py ./configs/classifier_stl10_mt_aug.yaml -subfolder revision_2021_3_1/baseline_aug -ssl_seed 1003 -gpu 3

# baseline 5000
python train_classifier.py ./configs/classifier_stl10_mt_noaug.yaml -subfolder revision_2021_3_1/baseline -n_labels 5000 -ssl_seed 1001 -gpu 0

python train_classifier.py ./configs/classifier_stl10_mt_noaug.yaml -subfolder revision_2021_3_1/baseline -n_labels 5000 -ssl_seed 1002 -gpu 1

python train_classifier.py ./configs/classifier_stl10_mt_noaug.yaml -subfolder revision_2021_3_1/baseline -n_labels 5000 -ssl_seed 1003 -gpu 2

python train_classifier.py ./configs/classifier_stl10_mt_aug.yaml -subfolder revision_2021_3_1/baseline_aug -n_labels 5000 -ssl_seed 1001 -gpu 0

python train_classifier.py ./configs/classifier_stl10_mt_aug.yaml -subfolder revision_2021_3_1/baseline_aug -n_labels 5000 -ssl_seed 1002 -gpu 1

python train_classifier.py ./configs/classifier_stl10_mt_aug.yaml -subfolder revision_2021_3_1/baseline_aug -n_labels 5000 -ssl_seed 1003 -gpu 3

# triple-gan


# gan_type: hinge and g_model_name: resnet_sngan96
python train_triplegan.py ./configs/triple_gan_stl10_mt_aug_sngan.yaml -subfolder revision_2021_3_1/tgan_aug -n_labels 1000 -ssl_seed 1001 -gpu 2

python train_triplegan.py ./configs/triple_gan_stl10_mt_aug_sngan.yaml -subfolder revision_2021_3_1/tgan_aug -n_labels 1000 -gan_type hinge -ssl_seed 1001 -gpu 3

python train_triplegan.py ./configs/triple_gan_stl10_mt_aug_sngan.yaml -subfolder revision_2021_3_1/tgan_aug -n_labels 1000 -gan_type hinge -g_model_name resnet_sngan96 -d_model_name resnet_sngan96 -gan_type bcr -ssl_seed 1001 -gpu 4

python train_triplegan.py ./configs/triple_gan_stl10_mt_aug_sngan.yaml -subfolder revision_2021_3_1/tgan_aug -n_labels 1000 -gan_type hinge -g_model_name resnet_sngan96 -d_model_name resnet_sngan96 -gan_type hinge -ssl_seed 1001 -gpu 5


# no aug
python train_triplegan.py ./configs/triple_gan_stl10_mt_noaug_sngan.yaml -subfolder revision_2021_3_1/tgan_noaug -n_labels 1000 -ssl_seed 1001 -gpu 6

python train_triplegan.py ./configs/triple_gan_stl10_mt_noaug_sngan.yaml -subfolder revision_2021_3_1/tgan_noaug -n_labels 1000 -gan_type hinge -ssl_seed 1001 -gpu 7

python train_triplegan.py ./configs/triple_gan_stl10_mt_noaug_sngan.yaml -subfolder revision_2021_3_1/tgan_noaug -n_labels 1000 -gan_type hinge -g_model_name resnet_sngan96 -d_model_name resnet_sngan96 -gan_type bcr -ssl_seed 1001 -gpu 2

python train_triplegan.py ./configs/triple_gan_stl10_mt_noaug_sngan.yaml -subfolder revision_2021_3_1/tgan_noaug -n_labels 1000 -gan_type hinge -g_model_name resnet_sngan96 -d_model_name resnet_sngan96 -gan_type hinge -ssl_seed 1001 -gpu 3



# other params follow tiny. may change to cifar10



