# Baseline
python train_classifier.py ./configs/classifier_cifar10_mt_aug.yaml -subfolder newbaseline -key aug -gpu 0 
python train_classifier.py ./configs/classifier_cifar10_mt_noaug.yaml -subfolder newbaseline -key noaug -gpu 1
python train_classifier.py ./configs/classifier_svhn_mt_aug.yaml -subfolder newbaseline -key aug -gpu 2
python train_classifier.py ./configs/classifier_svhn_mt_noaug.yaml -subfolder newbaseline -key noaug -gpu 3


python train_classifier.py ./configs/classifier_cifar10_mt_noaug.yaml -subfolder tune_ramp -rampup_length_lr -1 -c_lr 0.001 -key noaug -gpu 2
python train_classifier.py ./configs/classifier_cifar10_mt_noaug.yaml -subfolder tune_ramp -rampup_length_lr -1 -c_lr 0.0003 -key noaug -gpu 3
python train_classifier.py ./configs/classifier_cifar10_mt_noaug.yaml -subfolder tune_ramp -rampup_length_lr -1 -c_lr 0.0001 -key noaug -gpu 4

python train_classifier.py ./configs/classifier_svhn_mt_noaug.yaml -subfolder FinalBaseline -n_labels 250 -gpu 3
python train_classifier.py ./configs/classifier_svhn_mt_noaug.yaml -subfolder FinalBaseline -n_labels 500 -gpu 1
python train_classifier.py ./configs/classifier_svhn_mt_noaug.yaml -subfolder FinalBaseline -n_labels 1000 -gpu 2

python train_classifier.py ./configs/classifier_cifar10_mt_noaug.yaml -subfolder FinalBaseline -num_label_per_batch 2 -n_labels 1000 -gpu 1
python train_classifier.py ./configs/classifier_cifar10_mt_noaug.yaml -subfolder FinalBaseline -num_label_per_batch 4 -n_labels 2000 -gpu 2
python train_classifier.py ./configs/classifier_cifar10_mt_noaug.yaml -subfolder FinalBaseline -num_label_per_batch 8 -n_labels 4000 -gpu 3


python train_triplegan.py ./configs/triple_gan_cifar10_mt_aug.yaml -subfolder test -model_name resnet_sngan -gpu 0
python train_triplegan.py ./configs/triple_gan_svhn_load.yaml -subfolder loadC -gpu 1

# python train_classifier.py ./configs/classifier_cifar10_mt_aug.yaml -subfolder newbaseline -c_model_name cifar10_cnn_gaussian -key nogau_test -gpu 0
# python train_classifier.py ./configs/classifier_cifar10_mt_noaug.yaml -subfolder newbaseline -key noaug -c_model_name cifar10_cnn_gaussian -key nogau_test -gpu 1
# python train_classifier.py ./configs/classifier_svhn_mt_aug.yaml -subfolder newbaseline -c_model_name cifar10_cnn_gaussian -key nogau_test -gpu 2
# python train_classifier.py ./configs/classifier_svhn_mt_noaug.yaml -subfolder newbaseline -key noaug -c_model_name cifar10_cnn_gaussian -key nogau_test -gpu 3

# python train_classifier.py ./configs/classifier_cifar10_vat_aug.yaml -subfolder vat -gpu 0
# python train_classifier.py ./configs/classifier_cifar10_vat_noaug.yaml -subfolder vat -gpu 1
# python train_classifier.py ./configs/classifier_svhn_vat_aug.yaml -subfolder vat  -gpu 2
# python train_classifier.py ./configs/classifier_svhn_vat_noaug.yaml -subfolder vat -gpu 3

# python train_triplegan.py ./configs/triple_gan_cifar10_vat_aug.yaml -subfolder vat -gpu 4
# python train_triplegan.py ./configs/triple_gan_cifar10_vat_noaug.yaml -subfolder vat -gpu 5
# python train_triplegan.py ./configs/triple_gan_svhn_vat_aug.yaml -subfolder vat  -gpu 6
# python train_triplegan.py ./configs/triple_gan_svhn_vat_noaug.yaml -subfolder vat -gpu 7

# python train_triplegan.py ./configs/triple_gan_cifar10_vat_noaug_sngan.yaml -subfolder vat -gpu 2

python train_gan.py ./configs/gan.yaml -dataset svhn -d_model_name resnet_reggan -g_model_name resnet_reggan -subfolder GAN -gpu 0
python train_gan.py ./configs/gan.yaml -dataset svhn -model_name resnet_sngan  -subfolder GAN -gpu 1
python train_gan.py ./configs/gan.yaml -dataset cifar10 -d_model_name resnet_reggan -g_model_name resnet_reggan -subfolder GAN -gpu 2
python train_gan.py ./configs/gan.yaml -dataset cifar10 -model_name resnet_sngan  -subfolder GAN -gpu 3



python train_triplegan.py ./configs/triple_gan_svhn.yaml -dataset svhn -key fix_aug.addcgen.testlog -gpu 5
python train_triplegan.py ./configs/triple_gan_svhn.yaml -dataset svhn -key cford_int -gpu 1

# resnet
python train_classifier.py ./configs/classifier_cifar10_mt_resnet_aug.yaml -subfolder resnet -c_lr 0.2 -gpu 0,1,2,3
python train_classifier.py ./configs/classifier_cifar10_mt_resnet_aug.yaml -subfolder resnet -c_lr 0.1 -gpu 4,5,6,7


# Triple-gan
python train_triplegan.py ./configs/triple_gan_svhn_mt_aug.yaml -subfolder tganmt -translate 2 -c_model_name cifar10_cnn_gaussian -gpu 0
python train_triplegan.py ./configs/triple_gan_svhn_mt_noaug.yaml -subfolder tganmt -translate 0 -c_model_name cifar10_cnn_gaussian -gpu 1
python train_triplegan.py ./configs/triple_gan_cifar10_mt_aug.yaml -subfolder tganmt -key aug -c_model_name cifar10_cnn_gaussian -gpu 2


python test_triplegan.py ./configs/triple_gan_svhn.yaml -dataset svhn -key test -old_model "/home/kunxu/Workspace/Triple-GAN/allresults/(train_triplegan.py)_(svhn)_(2020-06-12-11-24-49)_()_(cford_int)/models/model209000.pt" -gpu 1