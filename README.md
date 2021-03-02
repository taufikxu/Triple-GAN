# Triple-GAN

PyTorch implementation of Triple-GAN built upon a mean teacher classifier and a projection discriminator with spectral normalization. The code reproduces the main results in https://arxiv.org/abs/1912.09784 


## Envoronment settings and libs we used in our experiments

pip install -r requirement.txt 

## Run examples

python train_classifier.py ./configs/classifier_cifar10_mt_noaug.yaml -subfolder cifar10 -key noaug -gpu 0

python train_triplegan.py ./configs/triple_gan_cifar10_mt_noaug_sngan.yaml -subfolder cifar10 -key noaug -gpu 1


## Results

### Semi-supervised classification

Using a commonly adopted 13-layer CNN classifier, Triple-GAN-V2 outperforms extensive semi-supervised learning methods substantially on more than 10 benchmarks no matter data augmentation is applied or not.

### Semi-supervised generation

Besides, with only 8% labels, Triple-GAN-V2 achieves comparable Inception Score (IS) and Frechet Inception Distance (FID) to CGAN-PD trained with full labels on the CIFAR10 dataset. It also significantly outperforms SNGAN trained with fully unlabeled data and CGAN-PD trained on a subset of labeled data.

### Extreme low data regime

We also provide preliminary results in the extreme low data regime where only a small subset of labeled data is available.

