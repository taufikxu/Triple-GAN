# baseline
python train_classifier.py ./configs/classifier.yaml -dataset svhn -subfolder baseline -gpu 1
python train_classifier_ssl.py ./configs/classifier_ssl.yaml -dataset svhn -subfolder baseline -gpu 2

python train_gan.py ./configs/gan.yaml -dataset cifar10 -key test -gpu 1
python train_triplegan.py ./configs/triple_gan_svhn.yaml -dataset svhn -key test -gpu 4