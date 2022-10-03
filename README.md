# Accurate Image Restoration with Attention Retractable Transformer
This code is the PyTorch implementation of ART model. Our ART achieves **state-of-the-art** performance in
- bicubic image SR
- color image denoising
- jpeg compression artifact reduction
## Requirements
- python 3.8
- pyTorch >= 1.8.0
- [BasicSR V1.3.5](https://github.com/xinntao/BasicSR)
- timm
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

## Install BasicSR
You should install BasicSR from PyPI in advance. More details are [here](https://github.com/XPixelGroup/BasicSR/blob/master/INSTALL.md).
```bash
pip install basicsr
```
## Test on SR
1. We provide some testing data in `/datasets`. Note that we put the pretrained models in `/experiments`, which can be downloaded. Make sure that you have downloaded them before testing. 
2. Follow the instructions below to begin testing our ART model.
```bash
# ART model for (x2) image SR. You can find corresponding results in Table 2 of the main paper.
python basicsr/test.py -opt options/test/test_ART_W8I4_500k_DF2K_x2.yml
# ART-S model for (x2) image SR. You can find corresponding results in Table 2 of the main paper.
python basicsr/test.py -opt options/test/test_ART_W8I8_500k_DF2K_x2.yml
```
You can find the visual results in automatically generated file folder `/results`. 

Note that you can put other benchmark datasets in `/datasets` and change YML files for further test.

## Test on Color Image Denoising
1. We provide some testing data in `/datasets`. Due to the limited space of github, we provide our completed trained model through this link ([net_g_ART_denoising15](https://ufile.io/x9dkndr3)). You should download it and place it in 'experiments' in advance.
2. Follow the instructions below to begin testing our ART model.
```bash
# ART model for Color Image Denoising. You can find corresponding results in Table 4 of the main paper.
python basicsr/test.py -opt options/test/test_art_denoising15.yml
```

Note that you can put other benchmark datasets in `/datasets` and change YML files for further test.

## Test on RealDenoising
1. Download the [SIDD test](https://drive.google.com/file/d/11vfqV-lqousZTuAit1Qkqghiv_taY0KZ/view) and place it in '/datasets/SIDD'.  
2. Due to the limited space of github, we provide our completed trained model through this link ([net_g_ART_realdenoising](https://ufile.io/x9dkndr3)). You should download it and place it in 'experiments' in advance.  
3. Go to folder 'RealDenoising'. Follow the instructions below to test our ART model. The output is in 'RealDenoising/results/Real_Denoising'/SIDD.

   ```bash
   # test our ART (training total iterations = 300K) on SSID
   python test_real_denoising_sidd.py
   ```

4. Run the scripts below to reproduce PSNR/SSIM on SIDD. The value will be 39.96 and 0.960.

   ```shell
   evaluate_sidd.m
   ```

## Test on JPEG Compression Artifact Reduction
1. We provide some testing data in `/datasets`. Due to the limited space of github, we provide our completed trained model through this link ([net_g_ART_jpeg40](https://ufile.io/l3mt29ss)). You should download it and place it in 'experiments' in advance.
2. Follow the instructions below to begin testing our ART model.
```bash
# ART model for JPEG CAR. You can find corresponding results in Table 5 of the main paper.
python basicsr/test.py -opt options/test/test_art_jpeg40.yml
```

Note that you can put other benchmark datasets in `/datasets` and change YML files for further test.

## License and Acknowledgement
This work is released under the Apache 2.0 license.
 The codes are based on [BasicSR](https://github.com/xinntao/BasicSR). Please also follow their licenses. Thanks for their awesome works.
