# - test bcr on ssl tiny tgan (to generate images and check bcr for bug and check elr for bug)
    
    python train_triplegan.py ./configs/triple_gan_tinyimagenet_mt_aug_sngan.yaml -subfolder Tiny32_SSL_TGAN_BCR -gan_type bcr -n_labels 2000 -gpu 0
    python train_triplegan.py ./configs/triple_gan_tinyimagenet_mt_noaug_sngan.yaml -subfolder Tiny32_SSL_TGAN_BCR -gan_type bcr -n_labels 2000 -gpu 1

    python train_triplegan.py ./configs/triple_gan_tinyimagenet_mt_aug_sngan.yaml -subfolder Tiny32_SSL_TGAN_BCR -gan_type bcr -n_labels 1000 -gpu 2
    python train_triplegan.py ./configs/triple_gan_tinyimagenet_mt_noaug_sngan.yaml -subfolder Tiny32_SSL_TGAN_BCR -gan_type bcr -n_labels 1000 -gpu 3

# - bcr + mt (original setting cifar10 and tiny. just no improvement or marginal)
    # should compare with losses in ssl settings to debug or improve

    # svhn 
    # xuan: python train_triplegan.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder New_elr_svhn -alpha_c_pdl 0.3 -n_labels 1000 -translate 2 -gpu 0
    python train_triplegan.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder New_elr_svhn -alpha_c_pdl 1.0 -n_labels 1000 -translate 2 -gpu 0

    python train_triplegan.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder New_elr_svhn -alpha_c_pdl 1.0 -consist_pdl true -n_labels 1000 -translate 2 -gpu 1

    python train_triplegan.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder New_elr_svhn -alpha_c_pdl 0.3 -n_labels 1000 -translate 0 -gpu 2
    python train_triplegan.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder New_elr_svhn -alpha_c_pdl 1.0 -n_labels 1000 -translate 0 -gpu 3

    # cifar10