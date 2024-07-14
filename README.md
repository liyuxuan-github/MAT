## MAT

First, train a clean model.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py -msg clean_24000  --train-type none --train_clean
```

Then, generate trigger set through clean model.

```bash
python build_generated_cifar10.py
```

Train a MAT watermark model.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py -msg 100_generated_trigger_set_24000_add_feature_loss_dist_reg_0.01  --train-type none --trigger_type add_feature_loss --dist_reg 0.01
```

Attack the MAT watermark model.

```bash
CUDA_VISIBLE_DEVICES=0 python extraction.py -msg 100_generated_trigger_set_24000_add_feature_loss_dist_reg_0.01  --train-type none
CUDA_VISIBLE_DEVICES=0 python distill.py -msg 100_generated_trigger_set_24000_add_feature_loss_dist_reg_0.01  --train-type none --distill-alpha 0.7
CUDA_VISIBLE_DEVICES=0 python finetune.py -msg 100_generated_trigger_set_24000_add_feature_loss_dist_reg_0.01  --train-type none
CUDA_VISIBLE_DEVICES=0 python fineprune.py -msg 100_generated_trigger_set_24000_add_feature_loss_dist_reg_0.01  --train-type none
```

Verify.

```bash
python test.py -m  ./experiments/cifar10_res18_none_100_100_generated_trigger_set_24000_add_feature_loss_dist_reg_0.01/extraction/checkpoints/checkpoint_nat_best.pt

python test.py -m  ./experiments/cifar10_res18_none_100_100_generated_trigger_set_24000_add_feature_loss_dist_reg_0.01/fineprune/checkpoints/checkpoint_nat_best.pt --pruning
```

## Baseline Watermark 

Train a  watermark model. (base/randomsmooth/minmaxpgd)

```bash
CUDA_VISIBLE_DEVICES=0 python train.py -msg minmaxpgd_100_trigger_set_24000  --train-type minmaxpgd --trigger_type only_clean
```

Attack the watermark model.

```bash
CUDA_VISIBLE_DEVICES=0 python extraction.py -msg minmaxpgd_100_trigger_set_24000  --train-type minmaxpgd
```

Verify.

see the output.log in the folder like "/experiments/cifar10_res18_minmaxpgd_100_minmaxpgd_10
0_trigger_set_24000/extraction/"
