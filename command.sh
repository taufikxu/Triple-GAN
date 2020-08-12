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

python train_gan.py ./configs/gan_cifar10.yaml -subfolder GAN_IS -gpu 0
python train_gan.py ./configs/gan_cifar10.yaml -subfolder GAN_IS -n_labels 4000 -gpu 1
python train_gan.py ./configs/gan_cifar10_un.yaml -subfolder GAN_IS -gpu 2


python train_triplegan.py ./configs/triple_gan_svhn.yaml -dataset svhn -key fix_aug.addcgen.testlog -gpu 5
python train_triplegan.py ./configs/triple_gan_svhn.yaml -dataset svhn -key cford_int -gpu 1

# resnet for report baseline
python train_classifier.py ./configs/classifier_cifar10_mt_resnet_aug.yaml -subfolder AverageBaseline_resnet_256 -ssl_seed 1001 -n_labels 4000 -gpu 0,1,2,3
python train_classifier.py ./configs/classifier_cifar10_mt_resnet_aug.yaml -subfolder AverageBaseline_resnet_256 -ssl_seed 1002 -n_labels 4000 -gpu 0,1,2,3
python train_classifier.py ./configs/classifier_cifar10_mt_resnet_aug.yaml -subfolder AverageBaseline_resnet_256 -ssl_seed 1003 -n_labels 4000 -gpu 0,1,4,5

python train_classifier.py ./configs/classifier_cifar10_mt_resnet_aug.yaml -subfolder AverageBaseline_resnet_256 -ssl_seed 1001 -n_labels 1000 -gpu 3,4,5,7
python train_classifier.py ./configs/classifier_cifar10_mt_resnet_aug.yaml -subfolder AverageBaseline_resnet_256 -ssl_seed 1002 -n_labels 1000 -gpu 0,1,2,3
python train_classifier.py ./configs/classifier_cifar10_mt_resnet_aug.yaml -subfolder AverageBaseline_resnet_256 -ssl_seed 1003 -n_labels 1000 -gpu 0,1,2,3

 # resnet consist pdl and masked pdl: last chance
# python train_triplegan.py ./configs/triple_gan_cifar10_mt_aug_sngan_resnet.yaml -subfolder ResNet_Last -consist_pdl true -masked_pdl true -n_labels 1000 -gpu 0,1,2,3
# python train_triplegan.py ./configs/triple_gan_cifar10_mt_aug_sngan_resnet.yaml -subfolder ResNet_Last -consist_pdl true -masked_pdl true -n_labels 4000 -gpu 4,5,6,7 

# python train_triplegan.py ./configs/triple_gan_cifar10_mt_aug_sngan_resnet.yaml -subfolder ResNet_Last -consist_pdl false -masked_pdl false -n_labels 1000 -gpu 0,1,2,3
# python train_triplegan.py ./configs/triple_gan_cifar10_mt_aug_sngan_resnet.yaml -subfolder TMP -consist_pdl false -masked_pdl false -n_labels 4000 -gpu 4,5,6,7 -alpha_c_pdl 1.0

python train_triplegan.py ./configs/triple_gan_cifar10_mt_aug_sngan_resnet.yaml -subfolder ResNet_Final -consist_pdl true -masked_pdl false -n_labels 1000 -gpu 4,5,6,7 -alpha_c_pdl 0.3 -ssl_seed 1003
python train_triplegan.py ./configs/triple_gan_cifar10_mt_aug_sngan_resnet.yaml -subfolder ResNet_Final -consist_pdl true -masked_pdl false -n_labels 4000 -gpu 0,1,2,3 -alpha_c_pdl 0.3 -ssl_seed 1001 -key save_model -save_every 2000

python train_triplegan.py ./configs/triple_gan_cifar10_mt_aug_sngan_resnet.yaml -subfolder ResNet_Tune -alpha_c_adv 0.03 -alpha_c_pdl 3.0 -n_labels 4000 -teach_for_d false -gpu 0,1,2,3
python train_triplegan.py ./configs/triple_gan_cifar10_mt_aug_sngan_resnet.yaml -subfolder ResNet_Tune -alpha_c_adv 0.03 -alpha_c_pdl 0.3 -n_labels 1000 -gpu 0,1,2,3
python train_triplegan.py ./configs/triple_gan_cifar10_mt_aug_sngan_resnet.yaml -subfolder ResNet_Tune -alpha_c_adv 0.03 -alpha_c_pdl 0.1 -n_labels 1000 -gpu 4,5,6,7
python train_triplegan.py ./configs/triple_gan_cifar10_mt_aug_sngan_resnet.yaml -subfolder ResNet_Tune -alpha_c_adv 0.03 -alpha_c_pdl 3.0 -n_labels 1000 -gpu 0,1,2,3

python train_triplegan.py ./configs/triple_gan_cifar10_mt_aug_sngan_resnet.yaml -subfolder TMP -alpha_c_adv 0.03 -alpha_c_pdl 0.3 -gpu 0,1,2,3
python train_triplegan.py ./configs/triple_gan_cifar10_mt_aug_sngan_resnet.yaml -subfolder TMP -alpha_c_adv 0.03 -alpha_c_pdl 0.1 -gpu 0,1,2,3

python train_triplegan.py ./configs/triple_gan_cifar10_mt_aug_sngan_resnet_100.yaml -subfolder ResNetDebug -gpu 4,5,6,7
python train_triplegan.py ./configs/triple_gan_cifar10_mt_aug_sugan_double.yaml -subfolder ResNetDebug -gpu 3
python test_resnet.py ./configs/triple_gan_resnet_test.yaml -old_model_c '/home/kunxu/Workspace/Triple-GAN/allresults/AverageBaseline_resnet/(train_classifier.py)_(cifar10)_(2020-07-15-16-36-43)_((ssl_seed_1001)(n_labels_4000))_(NotValid_Signature)/models/model150000.pt' -old_model_gan '/home/kunxu/Workspace/Triple-GAN/allresults/ResNetDebug/(train_triplegan.py)_(cifar10)_(2020-07-28-11-10-37)_()_(NotValid_Signature)/models/model70000.pt'

python train_triplegan_loadc.py ./configs/triple_gan_cifar10_mt_aug_sngan_resnet_load.yaml -old_model_c '/home/kunxu/Workspace/Triple-GAN/allresults/AverageBaseline_resnet/(train_classifier.py)_(cifar10)_(2020-07-15-16-36-49)_((ssl_seed_1001)(n_labels_1000))_(NotValid_Signature)/models/model150000.pt' -alpha_c_pdl 0.1 -gpu 0,1,2,3
python train_triplegan_loadc.py ./configs/triple_gan_cifar10_mt_aug_sngan_resnet_load.yaml -old_model_c '/home/kunxu/Workspace/Triple-GAN/allresults/AverageBaseline_resnet/(train_classifier.py)_(cifar10)_(2020-07-15-16-36-49)_((ssl_seed_1001)(n_labels_1000))_(NotValid_Signature)/models/model150000.pt' -alpha_c_pdl 0.3 -gpu 4,5,6,7

python train_triplegan_loadT.py ./configs/triple_gan_resnet_test.yaml -old_model_c '/home/kunxu/Workspace/Triple-GAN/allresults/ResNet_Tune/(train_triplegan.py)_(cifar10)_(2020-07-28-11-09-12)_((alpha_c_adv_0.03)(alpha_c_pdl_0.3)(n_labels_1000))_(NotValid_Signature)/models/model40000.pt' -gpu 0,1,2,3
# Triple-gan
python train_triplegan.py ./configs/triple_gan_svhn_mt_aug.yaml -subfolder tganmt -translate 2 -c_model_name cifar10_cnn_gaussian -gpu 0
python train_triplegan.py ./configs/triple_gan_svhn_mt_noaug.yaml -subfolder tganmt -translate 0 -c_model_name cifar10_cnn_gaussian -gpu 1
python train_triplegan.py ./configs/triple_gan_cifar10_mt_aug.yaml -subfolder tganmt -key aug -c_model_name cifar10_cnn_gaussian -gpu 2


# STL10 dataset
python train_classifier.py ./configs/classifier_stl10_mt_aug.yaml -subfolder stl10 -gpu 0
python train_triplegan.py ./configs/triple_gan_stl10_mt_aug_sngan.yaml -subfolder stl10 -gpu 1


# SWA
python train_classifier.py ./configs/classifier_cifar10_swa_aug.yaml -subfolder swa -key cnn -gpu 0
python train_classifier.py ./configs/classifier_cifar10_swa_resnet_aug.yaml -subfolder swa -key resnet -gpu 1

# new swa
python train_classifier.py ./configs/classifier_cifar10_swa_resnet_aug.yaml -subfolder swa -key resnet-norm -zca false -norm true -gpu 4
python train_triplegan.py ./configs/triple_gan_cifar10_swa_aug_sngan.yaml -n_iter_pretrain 0 -subfolder swa -gpu 5
python train_triplegan.py ./configs/triple_gan_cifar10_swa_aug_sngan_resnet.yaml -n_iter_pretrain 0 -subfolder swa_debug -gpu 6
python train_classifier.py ./configs/classifier_cifar10_swa_resnet_aug.yaml -subfolder swa -key t-gan-cnn -gpu 7



python test_triplegan.py ./configs/triple_gan_cifar10_mt_aug_sngan.yaml -key test -old_model "/home/kunxu/Workspace/Triple-GAN/allresults/ResNet_Last/(train_triplegan.py)_(cifar10)_(2020-07-31-15-58-04)_((consist_pdl_True)(masked_pdl_False)(n_labels_1000))_(NotValid_Signature)/models/model80000.pt" -gpu 3

python test_triplegan.py ./configs/triple_gan_cifar10_mt_aug_sngan.yaml -key test -old_model "/home/kunxu/Workspace/Triple-GAN/allresults/AverageBaseline_resnet/(train_classifier.py)_(cifar10)_(2020-07-15-16-36-43)_((ssl_seed_1001)(n_labels_4000))_(NotValid_Signature)/models/model150000.pt" -gpu 3

python test_is_fid.py ./configs/triple_gan_cifar10_mt_aug_sngan.yaml -key test -old_model "/home/kunxu/Workspace/Triple-GAN/allresults/GAN_IS/(train_gan.py)_(cifar10)_(2020-08-07-17-46-02)_((n_labels_4000))_(conditional)/models" -gpu 3



# final 
# python train_triplegan_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder elr_tgan_svhn_1000 -n_labels 1000 -ssl_seed 1001 -translate 2 -gpu 0
# python train_triplegan_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder elr_tgan_svhn_1000 -n_labels 1000 -ssl_seed 1002 -translate 2 -gpu 1
# python train_triplegan_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder elr_tgan_svhn_1000 -n_labels 1000 -ssl_seed 1003 -translate 2 -gpu 2 
# python train_triplegan_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder elr_tgan_svhn_500 -n_labels 500 -ssl_seed 1001 -translate 2 -gpu 3 
# python train_triplegan_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder elr_tgan_svhn_500 -n_labels 500 -ssl_seed 1002 -translate 2 -gpu 1 
# python train_triplegan_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder elr_tgan_svhn_500 -n_labels 500 -ssl_seed 1003 -translate 2 -gpu 2

# python train_triplegan_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder elr_tgan_svhn_1000 -n_labels 1000 -ssl_seed 1001 -translate 0 -gpu 1
# python train_triplegan_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder elr_tgan_svhn_1000 -n_labels 1000 -ssl_seed 1002 -translate 0 -gpu 7 
# python train_triplegan_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder elr_tgan_svhn_1000 -n_labels 1000 -ssl_seed 1003 -translate 0 -gpu 5 
# python train_triplegan_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder elr_tgan_svhn_500 -n_labels 500 -ssl_seed 1001 -translate 0 -gpu 0 
# python train_triplegan_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder elr_tgan_svhn_500 -n_labels 500 -ssl_seed 1002 -translate 0 -gpu 1 
# python train_triplegan_elr.py ./configs/triple_gan_svhn_noaug_elr.yaml -subfolder elr_tgan_svhn_500 -n_labels 500 -ssl_seed 1003 -translate 0 -gpu 2

# python train_triplegan_elr.py ./configs/triple_gan_cifar10_noaug_elr.yaml  -subfolder elr_tgan_cifar10 -n_labels 4000 -ssl_seed 1001 -translate 2 -flip_horizontal true -gpu 3
# python train_triplegan_elr.py ./configs/triple_gan_cifar10_noaug_elr.yaml  -subfolder elr_tgan_cifar10 -n_labels 4000 -ssl_seed 1002 -translate 2 -flip_horizontal true -gpu 1 
# python train_triplegan_elr.py ./configs/triple_gan_cifar10_noaug_elr.yaml  -subfolder elr_tgan_cifar10 -n_labels 4000 -ssl_seed 1003 -translate 2 -flip_horizontal true -gpu 2
# python train_triplegan_elr.py ./configs/triple_gan_cifar10_noaug_elr.yaml  -subfolder elr_tgan_cifar10 -n_labels 4000 -ssl_seed 1001 -translate 0 -flip_horizontal false -gpu 0
# python train_triplegan_elr.py ./configs/triple_gan_cifar10_noaug_elr.yaml  -subfolder elr_tgan_cifar10 -n_labels 4000 -ssl_seed 1002 -translate 0 -flip_horizontal false -gpu 3 
# python train_triplegan_elr.py ./configs/triple_gan_cifar10_noaug_elr.yaml  -subfolder elr_tgan_cifar10 -n_labels 4000 -ssl_seed 1003 -translate 0 -flip_horizontal false -gpu 4 

# python train_classifier_elr.py ./configs/classifier_cifar10_mt_aug_elr.yaml -subfolder elr_baseline_cifar10 -n_labels 4000 -ssl_seed 1001 -translate 2 -flip_horizontal true -gpu 1 
# python train_classifier_elr.py ./configs/classifier_cifar10_mt_aug_elr.yaml -subfolder elr_baseline_cifar10 -n_labels 4000 -ssl_seed 1002 -translate 2 -flip_horizontal true -gpu 5 
# python train_classifier_elr.py ./configs/classifier_cifar10_mt_aug_elr.yaml -subfolder elr_baseline_cifar10 -n_labels 4000 -ssl_seed 1003 -translate 2 -flip_horizontal true -gpu 6
# python train_classifier_elr.py ./configs/classifier_cifar10_mt_aug_elr.yaml -subfolder elr_baseline_cifar10 -n_labels 4000 -ssl_seed 1001 -translate 0 -flip_horizontal false -gpu 0 
# python train_classifier_elr.py ./configs/classifier_cifar10_mt_aug_elr.yaml -subfolder elr_baseline_cifar10 -n_labels 4000 -ssl_seed 1002 -translate 0 -flip_horizontal false -gpu 7 
# python train_classifier_elr.py ./configs/classifier_cifar10_mt_aug_elr.yaml -subfolder elr_baseline_cifar10 -n_labels 4000 -ssl_seed 1003 -translate 0 -flip_horizontal false -gpu 1 


# ablation
# aug and no aug
# python train_triplegan.py ./configs/triple_gan_cifar10_mt_aug_sngan.yaml -subfolder ablation -c_loss mtssl -alpha_c_pdl 0.0 -gpu 4
# python train_triplegan.py ./configs/triple_gan_cifar10_mt_aug_sngan.yaml -subfolder ablation -c_loss entropyssl -alpha_entropy 0.3 -gpu 2
# python train_triplegan.py ./configs/triple_gan_cifar10_mt_aug_sngan.yaml -subfolder ablation -c_loss loss_elr_wrap -alpha_c_pdl 3.0 -gpu 6


# TinyImagenet32
python train_classifier.py ./configs/classifier_tinyimagenet_mt_aug.yaml -subfolder TinyImg32 -gpu 0
python train_classifier.py ./configs/classifier_tinyimagenet_mt_noaug.yaml -subfolder TinyImg32 -gpu 1
python train_triplegan.py ./configs/triple_gan_tinyimagenet_mt_aug_sngan.yaml -subfolder TinyImg32 -gpu 2
python train_triplegan.py ./configs/triple_gan_tinyimagenet_mt_noaug_sngan.yaml -subfolder TinyImg32 -gpu 3


# python train_classifier.py ./configs/classifier_tinyimagenet_mt_aug.yaml -subfolder TinyImg32 -n_labels 1000 -ssl_seed 1002 -gpu 0
# python train_classifier.py ./configs/classifier_tinyimagenet_mt_aug.yaml -subfolder TinyImg32 -n_labels 1000 -ssl_seed 1003 -gpu 1
# python train_classifier.py ./configs/classifier_tinyimagenet_mt_noaug.yaml -subfolder TinyImg32 -n_labels 1000 -ssl_seed 1002 -gpu 5
# python train_classifier.py ./configs/classifier_tinyimagenet_mt_noaug.yaml -subfolder TinyImg32 -n_labels 1000 -ssl_seed 1003 -gpu 2

# python train_triplegan.py ./configs/triple_gan_tinyimagenet_mt_aug_sngan.yaml -subfolder TinyImg32 -n_labels 1000 -ssl_seed 1002 -gpu 2
# python train_triplegan.py ./configs/triple_gan_tinyimagenet_mt_aug_sngan.yaml -subfolder TinyImg32 -n_labels 1000 -ssl_seed 1003 -gpu 2
# python train_triplegan.py ./configs/triple_gan_tinyimagenet_mt_noaug_sngan.yaml -subfolder TinyImg32 -n_labels 1000 -ssl_seed 1002 -gpu 7
# python train_triplegan.py ./configs/triple_gan_tinyimagenet_mt_noaug_sngan.yaml -subfolder TinyImg32 -n_labels 1000 -ssl_seed 1003 -gpu 3

# python train_classifier.py ./configs/classifier_tinyimagenet_mt_aug.yaml -subfolder TinyImg32 -n_labels 2000 -ssl_seed 1001 -gpu 4
# python train_classifier.py ./configs/classifier_tinyimagenet_mt_aug.yaml -subfolder TinyImg32 -n_labels 2000 -ssl_seed 1002 -gpu 0
# python train_classifier.py ./configs/classifier_tinyimagenet_mt_aug.yaml -subfolder TinyImg32 -n_labels 2000 -ssl_seed 1003 -gpu 0
# python train_classifier.py ./configs/classifier_tinyimagenet_mt_noaug.yaml -subfolder TinyImg32 -n_labels 2000 -ssl_seed 1001 -gpu 3
# python train_classifier.py ./configs/classifier_tinyimagenet_mt_noaug.yaml -subfolder TinyImg32 -n_labels 2000 -ssl_seed 1002 -gpu 6
# python train_classifier.py ./configs/classifier_tinyimagenet_mt_noaug.yaml -subfolder TinyImg32 -n_labels 2000 -ssl_seed 1003 -gpu 0

# python train_triplegan.py ./configs/triple_gan_tinyimagenet_mt_aug_sngan.yaml -subfolder TinyImg32 -n_labels 2000 -ssl_seed 1001 -gpu 7
# python train_triplegan.py ./configs/triple_gan_tinyimagenet_mt_aug_sngan.yaml -subfolder TinyImg32 -n_labels 2000 -ssl_seed 1002 -gpu 3
# python train_triplegan.py ./configs/triple_gan_tinyimagenet_mt_aug_sngan.yaml -subfolder TinyImg32 -n_labels 2000 -ssl_seed 1003 -gpu 4
# python train_triplegan.py ./configs/triple_gan_tinyimagenet_mt_noaug_sngan.yaml -subfolder TinyImg32 -n_labels 2000 -ssl_seed 1001 -gpu 2
# python train_triplegan.py ./configs/triple_gan_tinyimagenet_mt_noaug_sngan.yaml -subfolder TinyImg32 -n_labels 2000 -ssl_seed 1002 -gpu 5
# python train_triplegan.py ./configs/triple_gan_tinyimagenet_mt_noaug_sngan.yaml -subfolder TinyImg32 -n_labels 2000 -ssl_seed 1003 -gpu 6


# tiny elr

python train_classifier.py ./configs/classifier_tinyimagenet_mt_aug.yaml -subfolder tiny_mt_elr_1000 -ssl_seed 1001 -n_labels 1000 -gpu 0
python train_classifier.py ./configs/classifier_tinyimagenet_mt_aug.yaml -subfolder tiny_mt_elr_1000 -ssl_seed 1002 -n_labels 1000 -gpu 0
python train_classifier.py ./configs/classifier_tinyimagenet_mt_aug.yaml -subfolder tiny_mt_elr_1000 -ssl_seed 1003 -n_labels 1000 -gpu 0
python train_classifier.py ./configs/classifier_tinyimagenet_mt_noaug.yaml -subfolder tiny_mt_elr_1000 -ssl_seed 1001 -n_labels 1000 -gpu 1
python train_classifier.py ./configs/classifier_tinyimagenet_mt_noaug.yaml -subfolder tiny_mt_elr_1000 -ssl_seed 1002 -n_labels 1000 -gpu 0
python train_classifier.py ./configs/classifier_tinyimagenet_mt_noaug.yaml -subfolder tiny_mt_elr_1000 -ssl_seed 1003 -n_labels 1000 -gpu 0

python train_classifier.py ./configs/classifier_tinyimagenet_mt_aug.yaml -subfolder tiny_mt_elr_2000 -ssl_seed 1001 -n_labels 2000 -gpu 7
python train_classifier.py ./configs/classifier_tinyimagenet_mt_aug.yaml -subfolder tiny_mt_elr_2000 -ssl_seed 1002 -n_labels 2000 -gpu 0
python train_classifier.py ./configs/classifier_tinyimagenet_mt_aug.yaml -subfolder tiny_mt_elr_2000 -ssl_seed 1003 -n_labels 2000 -gpu 0
python train_classifier.py ./configs/classifier_tinyimagenet_mt_noaug.yaml -subfolder tiny_mt_elr_2000 -ssl_seed 1001 -n_labels 2000 -gpu 1
python train_classifier.py ./configs/classifier_tinyimagenet_mt_noaug.yaml -subfolder tiny_mt_elr_2000 -ssl_seed 1002 -n_labels 2000 -gpu 0
python train_classifier.py ./configs/classifier_tinyimagenet_mt_noaug.yaml -subfolder tiny_mt_elr_2000 -ssl_seed 1003 -n_labels 2000 -gpu 0

python train_triplegan.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder tiny_tgan_elr_1000 -ssl_seed 1001 -translate 2 -flip_horizontal true -n_labels 1000 -gpu 2
python train_triplegan.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder tiny_tgan_elr_1000 -ssl_seed 1002 -translate 2 -flip_horizontal true -n_labels 1000 -gpu 0
python train_triplegan.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder tiny_tgan_elr_1000 -ssl_seed 1003 -translate 2 -flip_horizontal true -n_labels 1000 -gpu 0
python train_triplegan.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder tiny_tgan_elr_1000 -ssl_seed 1001 -translate 0 -flip_horizontal false -n_labels 1000 -gpu 0
python train_triplegan.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder tiny_tgan_elr_1000 -ssl_seed 1002 -translate 0 -flip_horizontal false -n_labels 1000 -gpu 0
python train_triplegan.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder tiny_tgan_elr_1000 -ssl_seed 1003 -translate 0 -flip_horizontal false -n_labels 1000 -gpu 0

python train_triplegan.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder tiny_tgan_elr_2000 -ssl_seed 1001 -translate 2 -flip_horizontal true -n_labels 2000 -gpu 5
python train_triplegan.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder tiny_tgan_elr_2000 -ssl_seed 1002 -translate 2 -flip_horizontal true -n_labels 2000 -gpu 0
python train_triplegan.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder tiny_tgan_elr_2000 -ssl_seed 1003 -translate 2 -flip_horizontal true -n_labels 2000 -gpu 0
python train_triplegan.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder tiny_tgan_elr_2000 -ssl_seed 1001 -translate 0 -flip_horizontal false -n_labels 2000 -gpu 0
python train_triplegan.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder tiny_tgan_elr_2000 -ssl_seed 1002 -translate 0 -flip_horizontal false -n_labels 2000 -gpu 0
python train_triplegan.py ./configs/triple_gan_tinyimagenet_noaug_elr.yaml -subfolder tiny_tgan_elr_2000 -ssl_seed 1003 -translate 0 -flip_horizontal false -n_labels 2000 -gpu 0















# baseline entropy and cnn for elr

python train_classifier_elr.py ./configs/classifier_svhn_elr.yaml -subfolder elr_base_debug -n_labels 1000 -ssl_seed 1001 -translate 2 -c_step regular -c_loss entropyssl -gpu 0
python train_classifier_elr.py ./configs/classifier_svhn_elr.yaml -subfolder elr_base_debug -n_labels 1000 -ssl_seed 1001 -translate 2 -c_step ramp -c_loss entropyssl -gpu 1
python train_classifier_elr.py ./configs/classifier_svhn_elr.yaml -subfolder elr_base_debug -n_labels 1000 -ssl_seed 1001 -translate 2 -c_step regular -c_loss loss_elr_wrap -gpu 2
python train_classifier_elr.py ./configs/classifier_svhn_elr.yaml -subfolder elr_base_debug -n_labels 1000 -ssl_seed 1001 -translate 2 -c_step ramp -c_loss loss_elr_wrap -gpu 3

python train_classifier_elr.py ./configs/classifier_svhn_elr.yaml -subfolder elr_base_debug -n_labels 500 -ssl_seed 1001 -translate 2 -c_step regular -c_loss entropyssl -gpu 1
python train_classifier_elr.py ./configs/classifier_svhn_elr.yaml -subfolder elr_base_debug -n_labels 500 -ssl_seed 1001 -translate 2 -c_step ramp -c_loss entropyssl -gpu 1
python train_classifier_elr.py ./configs/classifier_svhn_elr.yaml -subfolder elr_base_debug -n_labels 500 -ssl_seed 1001 -translate 2 -c_step regular -c_loss loss_elr_wrap -gpu 1
python train_classifier_elr.py ./configs/classifier_svhn_elr.yaml -subfolder elr_base_debug -n_labels 500 -ssl_seed 1001 -translate 2 -c_step ramp -c_loss loss_elr_wrap -gpu 1