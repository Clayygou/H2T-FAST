## H2T-FAST: Head-to-Tail Feature Augmentation by Style Transfer for Long-Tailed Recognition (ECAI 2023)

A Pytorch implementation of our ECAI 2023 paper "H2T-FAST: Head-to-Tail Feature Augmentation by Style Transfer for Long-Tailed Recognition".

How to train
-----------------

For example, CIFAR-10-LT 0.01.

   ```
   python main.py --dataset cifar10 -a resnet32 --num_classes 10 --imbanlance_rate 0.01 --beta 0.5 --lr 0.01 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.0 --label_weighting 1.2 --contrast_weight 1

   ```
See 'main.py' for more parameters.



How to test
-----------------

For example, CIFAR-10-LT 0.01.

   ```
   python test.py --dataset cifar10 --num_classes 10 --imbanlance_rate 0.01 --resume your ckpt.best.pth.tar

   ```
