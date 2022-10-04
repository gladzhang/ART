For testing, we provide the directory structure. You can download the complete datasets and put thme here. 

```shell
|-- datasets
    # image SR
    |-- SR
        |-- Set5
            |-- HR
            |-- LR_bicubic
                |-- X2
                |-- X3
                |-- X4
        |-- Set14
            |-- HR
            |-- LR_bicubic
                |-- X2
                |-- X3
                |-- X4
        |-- B100
            |-- HR
            |-- LR_bicubic
                |-- X2
                |-- X3
                |-- X4
        |-- Urban100
            |-- HR
            |-- LR_bicubic
                |-- X2
                |-- X3
                |-- X4
        |-- Manga109
            |-- HR
            |-- LR_bicubic
                |-- X2
                |-- X3
                |-- X4        
        # real image denoising - test
    |-- ColorDN
        |-- CBSD68HQ
        |-- Kodak24HQ
        |-- McMasterHQ
        |-- Urban100HQ
    |-- RealDN
        |-- SIDD
            |-- ValidationGtBlocksSrgb.mat
            |-- ValidationNoisyBlocksSrgb.mat
        |-- DND
            |-- info.mat
            |-- ValidationNoisyBlocksSrgb
                |-- 0001.mat
                |-- 0002.mat
                ：  
                |-- 0050.mat
    # grayscale JPEG compression artifact reduction - train & test
    |-- CAR
        |-- classic5
            |-- Classic5_HQ
            |-- Classic5_LQ
                |-- 10
                |-- 30
                |-- 40
        |-- LIVE1
            |-- LIVE1_HQ
            |-- LIVE1_LQ
                |-- 10
                |-- 30
                |-- 40
```

