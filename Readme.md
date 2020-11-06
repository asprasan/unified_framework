# A Unified Framework for Compressive Video Recovery from Coded Exposure Techniques

This repository contains the official implementation of the work **A Unified Framework for Compressive Video Recovery from Coded Exposure Techniques** accepted to be published at IEEE/CVF WACV 2021.

## args used in evaluating the model

- ```--ckpt``` : refers to the name of the model to be used
- ```--save_gif``` : saves the ground truth and predicted frames to the disk. Otherwise the code only logs the PSNR values.
- ```--flutter``` : to be used when evaluating the flutter shutter model
- ```--two_bucket``` : to be used when evaluating the two bucked coded-blurred image pair model

## Evaluation
We provide evaluation code for three different and important models in the paper.
We provide the DNN test set (described in the paper) for evaluation, in the ```data``` directory.

- Flutter shutter for 16x reconstruction : [download model](https://drive.google.com/file/d/1pQFYgTXUvbw8tH2Na6OB4RtlRz9qKTEQ/view?usp=sharing)
- Pixel wise coded exposure for 16x reconstruction : [download model](https://drive.google.com/file/d/1pEpvNK1prk9gvLZPvEimKg0em0ofbnKi/view?usp=sharing)
- Coded-Blurred pair for 16x reconstruction: [download model](https://drive.google.com/file/d/1aIhaMVqMjLkCVVjLH4eq_ilaQH5dQncC/view?usp=sharing)

Download the appropriate model file from the links provided above and copy them to the ```models``` directory. The files are compressed in ```.tar.xz``` format which can be extracted by ```tar -xvf <filename>```.

1. Evaluating flutter shutter model:

```python infer_h5.py --ckpt flutter_optimal.pth --gpu 0 --save_gif --flutter```

2. Evaluating the pixel-wise exposure model:
```python infer_h5.py --ckpt pixel_optimal.pth --gpu 0 --save_gif```

3. Evaluating the pixel-wise exposure model:
```python infer_h5.py --ckpt c2b_optimal.pth --gpu 0 --save_gif --two_bucket```
