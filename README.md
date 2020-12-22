# NoisyStudent
A PyTorch implementation of [Self-training with Noisy Student improves ImageNet classification](https://arxiv.org/abs/1911.04252)

# Requirements
Python >= 3.6, pytorch, torchvision, matplotlib, numpy, tensorboardX, tqdm

# Usage
<pre><code>$ python train.py -c config.json</code></pre> 

# Implementation details
* We tested on STL10 dataset, downsized to 32x32
* Model: Resnet18->Resnet26->Resnet34
* For data noise, RandAugment is used.
* For model noise, dropout and stochastic depth are used.

# Parameters
To change parameters, you can modify config.json file, or give -p arguements.
For example, if you want to change batch size to 128 and label type to hard,
<pre><code>$ python train.py -c config.json -p model_config.batch_size=128 model_config.label_type="hard"</code></pre> 

* augment_epoch, unaugment_epoch   
 In the [paper](https://arxiv.org/abs/1911.04252), authors used [a technique to fix train-test resolution discrepancy](https://arxiv.org/abs/1906.06423). They perform normal training with augmented dataset for 350 epochs(augment_epoch), and finetune with unaugmented dataset for 1.5 epochs(unaugment_epoch).
* label_type   
 "soft" or "hard". Used when pseudo-labeling unlabeled data.
* teacher_noise   
 Whether to apply data noise(RandAugment) when training a teacher model. In the original paper, a teacher model is trained without noise.
* dropout_prob   
 If set to 0.0, dropout layers are excluded.
* stochastic_depth_prob   
 If set to 1.0, stochastic depth will not be applied.
* ratio   
 Batch size ratio b/w labeled data and pseudo-labeled data in a mini-batch during training students.

# Acknowledgements
* RandAugment code from [pytorch-randaugment](https://github.com/ildoonet/pytorch-randaugment), and modified so that every noise increases its intensity from M=0 to M=20

