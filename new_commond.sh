# 写好程序写好命令，在一个机器上确性能跑，然后不断杀。

# 杀完清理下文件

# 留一些机器给xuan跑。这样随时可以看。

# 其实就是elr的结果。只是三个数据一起调整比较麻烦。

# 前往别忘记 push 和 pull

# - test bcr on ssl tiny tgan (to generate images and check bcr for bug and check elr for bug)
    
    # xuan 
    python train_triplegan.py ./configs/triple_gan_tinyimagenet_mt_aug_sngan.yaml -subfolder Tiny32_SSL_TGAN_BCR -gan_type bcr -n_labels 2000 -gpu 0
    python train_triplegan.py ./configs/triple_gan_tinyimagenet_mt_noaug_sngan.yaml -subfolder Tiny32_SSL_TGAN_BCR -gan_type bcr -n_labels 2000 -gpu 1

    python train_triplegan.py ./configs/triple_gan_tinyimagenet_mt_aug_sngan.yaml -subfolder Tiny32_SSL_TGAN_BCR -gan_type bcr -n_labels 1000 -gpu 2
    python train_triplegan.py ./configs/triple_gan_tinyimagenet_mt_noaug_sngan.yaml -subfolder Tiny32_SSL_TGAN_BCR -gan_type bcr -n_labels 1000 -gpu 3

# - bcr + mt (original setting cifar10 and tiny. just no improvement or marginal)
    # should compare with losses in ssl settings to debug or improve

    # NEW_SVHN g36
    python train_triplegan.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder NEW_SVHN -alpha_c_pdl 0.3 -n_labels 1000 -translate 2 -gpu 0
    python train_triplegan.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder NEW_SVHN -alpha_c_pdl 1.0 -n_labels 1000 -translate 2 -gpu 0
    python train_triplegan.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder NEW_SVHN -alpha_c_pdl 1.0 -n_labels 1000 -pdl_ramp_start 40000 -pdl_ramp_end 60000 -translate 2 -gpu 3
    python train_triplegan.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder NEW_SVHN -alpha_c_pdl 1.0 -n_labels 1000 -translate 0 -gpu 3
    

    # cifar10