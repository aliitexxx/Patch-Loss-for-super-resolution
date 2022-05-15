# Patch Loss for Super-resolution

对于每个train.yml，需要注意的是：
 - [ ] *yml*文件名和*name*一致；
 - [ ] 采用*lmdb*读取；
 - [ ] *val*的选择(注意倍数)；
 - [ ] 是否加载*pretrain*；
 - [ ] *loss*的选择(注意参数)；
 - [ ] *perceptual_kernels*的设置(多尺度)；
 - [ ] *metrics*的选择(注意倍数)；
 - [ ] 是否命名*wandb project*。

## 训练命令

    CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/EDSR/train_EDSR_Mx2.yml
    CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/RCAN/train_RCAN_x2.yml
    CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/SRResNet_SRGAN/train_MSRGAN_x4.yml
    CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/ESRGAN/train_ESRGAN_x4.yml
	CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/SwinIR/train_SwinIR_SRx4_scratch.yml

### 代码段1（*Validation set*）
    val:
      name: Set5
      type: PairedImageDataset
      dataroot_gt: datasets/Set5/GTmod12
      dataroot_lq: datasets/Set5/LRbicx4
      io_backend:
        type: disk
    
    val_2:
      name: Set14
      type: PairedImageDataset
      dataroot_gt: datasets/Set14/GTmod12
      dataroot_lq: datasets/Set14/LRbicx4
      io_backend:
        type: disk
    
    val_3:
      name: BSDS100
      type: PairedImageDataset
      dataroot_gt: datasets/BSDS100/GTmod12
      dataroot_lq: datasets/BSDS100/LRbicx4
      io_backend:
        type: disk
注意倍数。

### 代码段2（*perceptual loss*）
    perceptual_opt:
      type: PerceptualLoss
      layer_weights:
        'conv5_4': 1  # before relu
      vgg_type: vgg19
      use_input_norm: true
      range_norm: false
      perceptual_weight: 1.0
      style_weight: 0
      criterion: patch
      use_std_to_force: true
      perceptual_patch_weight: 10.0
      perceptual_kernels: [4, 8]
 1. *use_std_to_force*指是否使用*std*修正(默认`use_std_to_force=True`)。
 2. *perceptual_patch_weight*指特征图上额外计算*patch loss*的比重。若`perceptual_patch_weight=0`，则特征图上仅仅使用*L1*距离。
 3. *perceptual_kernels*指特征图上计算*patch loss*的滑动窗口(默认`stride=1`)。若`perceptual_patch_weight=0`则失效。
 4. 若`gt_size=64`，则`perceptual_kernels=[4]`；若`gt_size=128`，则`perceptual_kernels=[4, 8]`；若`gt_size=96`，则`perceptual_kernels=[3, 6]`；若`gt_size=192`，则`perceptual_kernels=[6, 12]`。
### 代码段3（*patch loss*）
    patch_opt_3d_xd:
      type: patchLoss3DXD
      use_std_to_force: true
      kernel_sizes: [3, 5, 7]
      loss_weight: 1.0
 1. *use_std_to_force*指是否使用*std*修正(默认`use_std_to_force=True`)。
 2. *kernel_sizes*指的是原图上计算*patch loss*的滑动窗口(默认`stride=1`)。
 3. *loss_weight*指的是*patch loss*的比重。如果`loss_weight=0`，则不在原图上计算patch loss。
### 代码段4（*metrics*）
    metrics:
      psnr: # metric name, can be arbitrary
        type: calculate_psnr
        crop_border: 4
        test_y_channel: false
      ssim:
        type: calculate_ssim
        crop_border: 4
        test_y_channel: false
        better: higher
      niqe:
        type: calculate_niqe
        crop_border: 4
        better: lower
      lpips:
        type: calculate_lpips
        crop_border: 4
      patch:
        type: calculate_pearson_patch
        crop_border: 4
        better: higher
      patch2:
        type: calculate_cosine_patch
        crop_border: 4
        better: higher

 1. EDSR, RCAN, SRGAN, ESRGAN, SwinIR默认`test_y_channel=False`；
 2. ECBSR默认为`test_y_channel=True`；
 3. *crop_border*的值与倍数相同。

## 测试命令

    CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/EDSR/test_EDSR_Mx2.yml
    CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/RCAN/test_RCAN_x2.yml
    CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/SRResNet_SRGAN/test_MSRGAN_x4.yml
    CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/ESRGAN/test_ESRGAN_x4.yml

对于每个test.yml，需要注意的是：

 - [ ] *datasets*的选择(注意倍数)；
 - [ ] *name*和*pretrain_network_g*一致；
 - [ ] *save_img*填写(默认*true*)；
 - [ ] *draw_curves*填写(默认*false*)；
 - [ ] *iters_stride*填写(当`draw_curves=True`生效，默认`iters_stride=20000`)；
 - [ ] *metrics*的选择(注意倍数)。

### 代码段1（*Test set*）

    datasets:
      test_1:  # the 1st test dataset
        name: Set5
        type: PairedImageDataset
        dataroot_gt: datasets/Set5/GTmod12
        dataroot_lq: datasets/Set5/LRbicx4
        io_backend:
          type: disk
    
      test_2:
        name: Set14
        type: PairedImageDataset
        dataroot_gt: datasets/Set14/GTmod12
        dataroot_lq: datasets/Set14/LRbicx4
        io_backend:
          type: disk
    
      test_3:
        name: BSDS100
        type: PairedImageDataset
        dataroot_gt: datasets/BSDS100/GTmod12
        dataroot_lq: datasets/BSDS100/LRbicx4
        io_backend:
          type: disk
    
      test_4:
        name: urban100
        type: PairedImageDataset
        dataroot_gt: datasets/urban100/GTmod12
        dataroot_lq: datasets/urban100/LRbicx4
        io_backend:
          type: disk
    
      test_5:
        name: manga109
        type: PairedImageDataset
        dataroot_gt: datasets/manga109/GTmod12
        dataroot_lq: datasets/manga109/LRbicx4
        io_backend:
          type: disk
        
      # test_6:
      #   name: DIV2K100
      #   type: PairedImageDataset
      #   dataroot_gt: datasets/DIV2K/DIV2K_valid_HR
      #   dataroot_lq: datasets/DIV2K/DIV2K_valid_LR_bicubic/X4
      #   filename_tmpl: '{}x4'
      #   io_backend:
      #     type: disk

注意倍数。

### 代码段2（*Test settings*）

    val:
      save_img: false
      draw_curves: true 
      iters_stride: 20000  # default
      suffix: ~  # add suffix to saved images, if None, use exp name
or
      
    val:
      save_img: true
      draw_curves: false
      iters_stride: 20000  # default
      suffix: ~  # add suffix to saved images, if None, use exp name

 1. `save_img=False` and `draw_curves=True` (output images in *./visualization*).
 2. `save_img=True` and `draw_curves=False` (output ***metrics.txt***).
 3. `save_img=True` and `draw_curves=True` (not suggest).

### 代码段3（*metrics*）
 
    metrics:
       psnr: # metric name, can be arbitrary
         type: calculate_psnr
         crop_border: 4
         test_y_channel: false
       ssim:
         type: calculate_ssim
         crop_border: 4
         test_y_channel: false
       lpips:
         type: calculate_lpips
         crop_border: 4
       niqe:
         type: calculate_niqe
         crop_border: 4
       patch:
         type: calculate_pearson_patch
         crop_border: 4
       patch2:
         type: calculate_cosine_patch
         crop_border: 4
   注意*crop_border*的值与倍数相同。


## MATLAB CODE
MATLAB代码提供了两个主函数*evaluate_results_dirs_linux.m*和*evaluate_results_dirs_win.m*以方便测试NIQE, Ma, PI这些感知指标。这两个函数在不同的系统上执行，其遍历所有图像目录的目录*test_results*和*GT_datasets*。运行

    evaluate_results_dirs_linux test_results GT_datasets shave_width true
    evaluate_results_dirs_win test_results GT_datasets shave_width true

其中*shave_width*表示边缘裁剪掉的像素数。注意*test_results/visualization*目录和*GT_datasets*目录下面应包含Set5, Set14, BSDS100,  manga109, urban100这五个数据集。
 
## Inference of SwinIR
SwinIRx2的推理代码：

    CUDA_VISIBLE_DEVICES=0 python inference/inference_swinir.py --input datasets/Set5/LRbicx2 --scale 2 --patch_size 48 --model_path experiments/001_0_SwinIR_SRx2_scratch_P48W8_DIV2K_500k_B4G8/models/net_g_500000.pth --output results/SwinIRx2_0/Set5
    CUDA_VISIBLE_DEVICES=0 python inference/inference_swinir.py --input datasets/Set14/LRbicx2 --scale 2 --patch_size 48 --model_path experiments/001_0_SwinIR_SRx2_scratch_P48W8_DIV2K_500k_B4G8/models/net_g_500000.pth --output results/SwinIRx2_0/Set14
    CUDA_VISIBLE_DEVICES=0 python inference/inference_swinir.py --input datasets/BSDS100/LRbicx2 --scale 2 --patch_size 48 --model_path experiments/001_0_SwinIR_SRx2_scratch_P48W8_DIV2K_500k_B4G8/models/net_g_500000.pth --output results/SwinIRx2_0/BSDS100
    CUDA_VISIBLE_DEVICES=0 python inference/inference_swinir.py --input datasets/urban100/LRbicx2 --scale 2 --patch_size 48 --model_path experiments/001_0_SwinIR_SRx2_scratch_P48W8_DIV2K_500k_B4G8/models/net_g_500000.pth --output results/SwinIRx2_0/urban100
    CUDA_VISIBLE_DEVICES=0 python inference/inference_swinir.py --input datasets/manga109/LRbicx2 --scale 2 --patch_size 48 --model_path experiments/001_0_SwinIR_SRx2_scratch_P48W8_DIV2K_500k_B4G8/models/net_g_500000.pth --output results/SwinIRx2_0/manga109
    
    CUDA_VISIBLE_DEVICES=0 python scripts/metrics/calculate_metrics.py --gt datasets/Set5/GTmod12/ --restored results/SwinIRx2_0/Set5 --crop_border 2
    CUDA_VISIBLE_DEVICES=0 python scripts/metrics/calculate_metrics.py --gt datasets/Set14/GTmod12/ --restored results/SwinIRx2_0/Set5 --crop_border 2
    CUDA_VISIBLE_DEVICES=0 python scripts/metrics/calculate_metrics.py --gt datasets/BSDS100/GTmod12/ --restored results/SwinIRx2_0/Set5 --crop_border 2
    CUDA_VISIBLE_DEVICES=0 python scripts/metrics/calculate_metrics.py --gt datasets/urban100/GTmod12/ --restored results/SwinIRx2_0/Set5 --crop_border 2
    CUDA_VISIBLE_DEVICES=0 python scripts/metrics/calculate_metrics.py --gt datasets/manga109/GTmod12/ --restored results/SwinIRx2_0/Set5 --crop_border 2

SwinIRx4的推理代码：

    CUDA_VISIBLE_DEVICES=0 python inference/inference_swinir.py --input datasets/Set5/LRbicx4 --patch_size 48 --model_path experiments/002_0_SwinIR_SRx4_scratch_P48W8_DIV2K_500k_B4G8/models/net_g_500000.pth --output results/SwinIRx4_0/Set5
    CUDA_VISIBLE_DEVICES=0 python inference/inference_swinir.py --input datasets/Set14/LRbicx4 --patch_size 48 --model_path experiments/002_0_SwinIR_SRx4_scratch_P48W8_DIV2K_500k_B4G8/models/net_g_500000.pth --output results/SwinIRx4_0/Set14
    CUDA_VISIBLE_DEVICES=0 python inference/inference_swinir.py --input datasets/BSDS100/LRbicx4 --patch_size 48 --model_path experiments/002_0_SwinIR_SRx4_scratch_P48W8_DIV2K_500k_B4G8/models/net_g_500000.pth --output results/SwinIRx4_0/BSDS100
    CUDA_VISIBLE_DEVICES=0 python inference/inference_swinir.py --input datasets/urban100/LRbicx4 --patch_size 48 --model_path experiments/002_0_SwinIR_SRx4_scratch_P48W8_DIV2K_500k_B4G8/models/net_g_500000.pth --output results/SwinIRx4_0/urban100
    CUDA_VISIBLE_DEVICES=0 python inference/inference_swinir.py --input datasets/manga109/LRbicx4 --patch_size 48 --model_path experiments/002_0_SwinIR_SRx4_scratch_P48W8_DIV2K_500k_B4G8/models/net_g_500000.pth --output results/SwinIRx4_0/manga109
    
    CUDA_VISIBLE_DEVICES=0 python scripts/metrics/calculate_metrics.py --gt datasets/Set5/GTmod12/ --restored results/SwinIRx4_0/Set5 --crop_border 4
    CUDA_VISIBLE_DEVICES=0 python scripts/metrics/calculate_metrics.py --gt datasets/Set14/GTmod12/ --restored results/SwinIRx4_0/Set5 --crop_border 4
    CUDA_VISIBLE_DEVICES=0 python scripts/metrics/calculate_metrics.py --gt datasets/BSDS100/GTmod12/ --restored results/SwinIRx4_0/Set5 --crop_border 4
    CUDA_VISIBLE_DEVICES=0 python scripts/metrics/calculate_metrics.py --gt datasets/urban100/GTmod12/ --restored results/SwinIRx4_0/Set5 --crop_border 4
    CUDA_VISIBLE_DEVICES=0 python scripts/metrics/calculate_metrics.py --gt datasets/manga109/GTmod12/ --restored results/SwinIRx4_0/Set5 --crop_border 4

