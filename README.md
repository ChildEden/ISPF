# [AAAI 2025] Toward Efficient Data-Free Unlearning

This is a PyTorch implementation of [Toward Efficient Data-Free Unlearning](https://github.com/ChildEden/ISPF) accepted by AAAI 2025.

## Abstract

Machine unlearning without access to real data distribution is challenging. The existing method based on data-free distillation achieved unlearning by filtering out synthetic samples containing forgetting information but struggled to distill the retaining-related knowledge efficiently. In this work, we analyze that such a problem is due to over-filtering, which reduces the synthesized retaining-related information. We propose a novel method, Inhibited Synthetic PostFilter (ISPF), to tackle this challenge from two perspectives: First, the Inhibited Synthetic, by reducing the synthesized forgetting information; Second, the PostFilter, by fully utilizing the retaining-related information in synthesized samples. Experimental results demonstrate that the proposed ISPF effectively tackles the challenge and outperforms existing methods.

## Environment Setting

All necessary packages are listed in `requirements.txt`. You can install with your python environment by running:
      
```bash
pip install -r requirements.txt
```

## Reproduce:  

```bash
# Prepare the original model
bash scripts/scratch/cifar10_allcnn.sh

# Prepare the retrain model
bash scripts/scratch/cifar10_allcnn_retrain.sh

# Run data-free unlearning methods
bash scripts/un_dfq_cifar10_allcnn.sh
```

By launching the tensorboard, you can monitor the training process, e.g., accuracy, loss, generated samples, etc.

```bash
tensorboard --logdir=runs/training_history
```

## Acknowledgement

The code was developed based on the [CMI code](https://github.com/zju-vipa/CMI) that is a well-organized DFKD benchmark. We extend our gratitude to the authors of CMI for their contributions to the DFKD topic and the open-source community.


## Citation

```
@inproceedings{  

}
```