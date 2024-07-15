# Not Just Change the Labels, Learn the Features: Watermarking Deep Neural Networks with Multi-View Data

**M**ultiview d**AT**a  (**MAT**) is a novel watermarking technique based on Multiview data for efficiently embedding watermarks within DNNs. Experiments across various benchmarks demonstrated its efficacy in defending against model extraction attacks.

![image-20240715223746983](C:\Users\Li\AppData\Roaming\Typora\typora-user-images\image-20240715223746983.png)

> [**Not Just Change the Labels, Learn the Features: Watermarking Deep Neural Networks with Multi-View Data**](https://arxiv.org/pdf/2403.10663)            
> [Yuxuan Li , Sarthak Kumar Maharana, [Yunhui Guo](https://yunhuiguo.github.io/)     
> Harbin Institute of Technology, UT Dallas          
> ECCV 2024

[[`arxiv`](https://arxiv.org/pdf/2403.10663)] [[`bibtex`](#citation)] 

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

## Acknowledgement

Our project is implemented base on the following projects. We really appreciate their excellent open-source works!

* [Margin-based Neural Network Watermarking]([GitHub - matbambbang/margin-based-watermarking: The source codes of the paper 'Margin-based Neural Network Watermarking'](https://github.com/matbambbang/margin-based-watermarking))

## Citation

If our work has been helpful to you, we would greatly appreciate a citation.

```
@article{li2024not,
  title={Not Just Change the Labels, Learn the Features: Watermarking Deep Neural Networks with Multi-View Data},
  author={Li, Yuxuan and Maharana, Sarthak Kumar and Guo, Yunhui},
  journal={arXiv preprint arXiv:2403.10663},
  year={2024}
}
```
