# baseline
python train_classifier.py ./configs/classifier.yaml -dataset svhn -subfolder baseline -gpu 1
python train_classifier_ssl.py ./configs/classifier_ssl.yaml -dataset svhn -subfolder baseline -gpu 2
python train_classifier_mt.py ./configs/classifier_mt_svhn.yaml -dataset svhn -subfolder baseline -translate 2 -gpu 0
python train_classifier_mt.py ./configs/classifier_mt_svhn.yaml -dataset svhn -subfolder baseline -translate 0 -gpu 1
python train_classifier_mt.py ./configs/classifier_mt_svhn.yaml -dataset svhn -subfolder baseline -translate 0 -ssl_seed 2 -gpu 5
python train_classifier_mt.py ./configs/classifier_mt_svhn.yaml -dataset svhn -subfolder baseline -translate 2 -ssl_seed 2 -gpu 6
python train_classifier_mt.py ./configs/classifier_mt_cifar10.yaml -dataset cifar10 -subfolder baseline -translate 2 -flip_horizontal true -gpu 2
python train_classifier_mt.py ./configs/classifier_mt_cifar10.yaml -dataset cifar10 -subfolder baseline -translate 0 -flip_horizontal false -gpu 3
python train_classifier_mt.py ./configs/classifier_mt_cifar10.yaml -dataset cifar10 -subfolder baseline -zca false -translate 2 -flip_horizontal true -gpu 4


python train_gan.py ./configs/gan.yaml -dataset cifar10 -key test -gpu 1
python train_triplegan.py ./configs/triple_gan_svhn.yaml -dataset svhn -key fix_aug.addcgen.testlog -gpu 5
python train_triplegan.py ./configs/triple_gan_svhn.yaml -dataset svhn -key cford_int -gpu 1


python test_triplegan.py ./configs/triple_gan_svhn.yaml -dataset svhn -key test -old_model "/home/kunxu/Workspace/Triple-GAN/allresults/(train_triplegan.py)_(svhn)_(2020-06-12-11-24-49)_()_(cford_int)/models/model209000.pt" -gpu 1