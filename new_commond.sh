# - test bcr on ssl tiny tgan (to generate images and check bcr for bug and check elr for bug)
    
    # xuan 
    python train_triplegan.py ./configs/triple_gan_tinyimagenet_mt_aug_sngan.yaml -subfolder Tiny32_SSL_TGAN_BCR -gan_type bcr -n_labels 2000 -gpu 0
    python train_triplegan.py ./configs/triple_gan_tinyimagenet_mt_noaug_sngan.yaml -subfolder Tiny32_SSL_TGAN_BCR -gan_type bcr -n_labels 2000 -gpu 1

    python train_triplegan.py ./configs/triple_gan_tinyimagenet_mt_aug_sngan.yaml -subfolder Tiny32_SSL_TGAN_BCR -gan_type bcr -n_labels 1000 -gpu 2
    python train_triplegan.py ./configs/triple_gan_tinyimagenet_mt_noaug_sngan.yaml -subfolder Tiny32_SSL_TGAN_BCR -gan_type bcr -n_labels 1000 -gpu 3

# - bcr + mt (original setting cifar10 and tiny. just no improvement or marginal)
    # should compare with losses in ssl settings to debug or improve

    # NEW_SVHN g36
    python train_triplegan_final_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder NEW_SVHN -alpha_c_pdl 0.3 -n_labels 1000 -ssl_seed 1001 -translate 2 -gpu 1
    python train_triplegan_final_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder NEW_SVHN -alpha_c_pdl 1.0 -n_labels 1000 -ssl_seed 1001 -translate 2 -gpu 2
    python train_triplegan_final_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder NEW_SVHN -alpha_c_pdl 1.0 -n_labels 1000 -ssl_seed 1001 -pdl_ramp_start 40000 -pdl_ramp_end 60000 -translate 2 -gpu 3
    python train_triplegan_final_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder NEW_SVHN -alpha_c_pdl 1.0 -n_labels 1000 -ssl_seed 1001 -translate 0 -gpu 4
    python train_triplegan_final_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder NEW_SVHN -alpha_c_pdl 3.0 -n_labels 1000 -ssl_seed 1001 -translate 2 -gpu 7

    # NEW_SVHN 500 800 g26 g34
    python train_triplegan_final_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder NEW_SVHN -alpha_c_pdl 0.3 -n_labels 800 -ssl_seed 1001 -translate 2 -gpu 7
    python train_triplegan_final_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder NEW_SVHN -alpha_c_pdl 1.0 -n_labels 800 -ssl_seed 1001 -translate 2 -gpu 4
    python train_triplegan_final_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder NEW_SVHN -alpha_c_pdl 1.0 -n_labels 800 -ssl_seed 1001 -translate 0 -gpu 7
    
    python train_triplegan_final_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder NEW_SVHN -alpha_c_pdl 0.3 -n_labels 500 -ssl_seed 1001 -translate 2 -gpu 6
    python train_triplegan_final_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder NEW_SVHN -alpha_c_pdl 1.0 -n_labels 500 -ssl_seed 1001 -translate 2 -gpu 5
    python train_triplegan_final_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder NEW_SVHN -alpha_c_pdl 1.0 -n_labels 500 -ssl_seed 1001 -translate 0 -gpu 4

    # NEW_CIFAR10 g32
    python train_triplegan_final_elr.py ./configs/triple_gan_cifar10_noaug_elr.yaml -subfolder NEW_CIFAR10 -alpha_c_pdl 3.0 -n_labels 4000 -ssl_seed 1001 -translate 2 -flip_horizontal true -gpu 0
    python train_triplegan_final_elr.py ./configs/triple_gan_cifar10_noaug_elr.yaml -subfolder NEW_CIFAR10 -alpha_c_pdl 1.0 -n_labels 4000 -ssl_seed 1001 -translate 2 -flip_horizontal true -gpu 1
    python train_triplegan_final_elr.py ./configs/triple_gan_cifar10_noaug_elr.yaml -subfolder NEW_CIFAR10 -alpha_c_pdl 3.0 -n_labels 4000 -ssl_seed 1001 -translate 0 -flip_horizontal false -gpu 2

    # NEW_TINY g28 g26
    python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder NEW_TINY -alpha_c_pdl 3.0 -n_labels 2000 -ssl_seed 1001 -translate 2 -flip_horizontal true -gpu 0
    python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder NEW_TINY -alpha_c_pdl 1.0 -n_labels 2000 -ssl_seed 1001 -translate 2 -flip_horizontal true -gpu 1
    python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder NEW_TINY -alpha_c_pdl 3.0 -n_labels 2000 -ssl_seed 1001 -translate 0 -flip_horizontal false -gpu 2

    python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder NEW_TINY -alpha_c_pdl 3.0 -n_labels 1000 -ssl_seed 1001 -translate 2 -flip_horizontal true -gpu 3
    python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder NEW_TINY -alpha_c_pdl 1.0 -n_labels 1000 -ssl_seed 1001 -translate 2 -flip_horizontal true -gpu 2
    python train_triplegan_final_elr.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder NEW_TINY -alpha_c_pdl 3.0 -n_labels 1000 -ssl_seed 1001 -translate 0 -flip_horizontal false -gpu 3
