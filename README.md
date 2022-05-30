# Scene
## 线上实验
### Gauganv1

#### standard
|  ID    |  描述    | bs | mask_acc  | 美学 | FID | 总分
|  ----    |  ----  | ----  |----  |----  |----  |----  |
|  1    | gauganv1_s512+esrgan_x2  | 12 | 0.8808 | 4.9649 | 47.3168 | 0.4507
|  2    | gauganv1_s256+esrgan_x2  | 8 | 0.8718 | 5.0146 | 52.2234 | 0.4268
|  3    | gauganv1_s512  | 12 | **0.9063** | 4.9318 | 47.7813 | 0.4601
|  4    | gauganv1_s256_vae+esrgan_x2  | 8 | 0.8714 | 5.0663 | 42.69 | 0.4704
|  8    | gauganv1_s256_vae+bicubic_x2  | 8 | 0.8828 | 4.9664 | 51.2145| 0.4346
|  10    | gauganv1_s512_vae  | 12 | 0.9007 | 5.0321 | 37.0918| 0.5099


#### cycleGAN
|  ID    |  描述    | bs | mask_acc  | 美学 | FID | 总分
|  ----    |  ----  | ----  |----  |----  |----  |----  |
|  5    | gauganv1_s512+cycleGAN_ep180  | 6 | 0.8833 | 4.9263 | 41.567 | 0.4757
|  6    | gauganv1_s512+cycleGAN_ep20  | 6 | 0.8877 | 4.8457 | 49.1204 | 0.4409
|  7    | gauganv1_s256_vae+esrgan_x2+cycleGAN_ep180  | 6 | 0.8714 | 4.8706 | 36.3846| 0.4894
|  7    | cocosnetv1+cycleGAN_ep180  | 6 | 0.8789 | 5.1073 | 35.0023| 0.5101
|  7    | cocosnetv2+cycleGAN_ep180  | 6 | 0.8954 | 5.292 | 31.8364| 0.5421

#### cx loss
|  ID    |  描述    | bs | mask_acc  | 美学 | FID | 总分
|  ----    |  ----  | ----  |----  |----  |----  |----  |
|  9    | gauganv1_s256_vae+esrgan_x2+cxloss_w10.0  | 4 | 0.9059 | 4.9234 | 35.0399| 0.5172
|  11   |  gauganv1_s256_vae+esrgan_x2+cxloss_w5  | 4 | 0.9043 | 4.8984 | 35.5186 | 0.513
|  12   |  gauganv1_s256_vae+esrgan_x2+cxloss_w2.5  | 4 | 0.9043 | 5.0655 | **33.2812** | 0.5307
|  13   |  gauganv1_s256_vae+esrgan_x2+cxloss_w2.5_L2  | 4 | **0.9128** | 5.0504 | **32.6841** | **0.5377**
|  17   |  gauganv1_s256_vae+esrgan_x2+cxloss_smoothL1  | 4 | 0.8969 | 4.849 | 35.6237 | 0.5062
|  18   |  gauganv1_s256_vae+esrgan_x2+cxloss_w2.5_L2+cycleGAN  | 4 | 0.8844 | 4.9047 | 36.4918 | 0.4977
|  22   |  gauganv1_s256_vae+esrgan_x2+cxloss_w2.5_L2 (IOU)  | 4 | 0.8998 | 5.1044 | 33.2381 | 0.53
|  27   |  gauganv1_s512_vae+cxloss_w10_L1(e35)w2.5_L2(e45)  | 4 | 0.864 | 5.1484 | 39.2625 | 0.4848
|  28   |  gauganv1_s512_vae+cxloss_w2.5_L2(e50)  | 8 | 0.8758 | 5.1645 | 40.7387 | 0.4857
|  25   |  gauganv1_s512+cxloss_w2.5  | 1 | 0.9005 | 5.1418 | 47.9501 | 0.4659

#### losses
|  ID    |  描述    | bs | mask_acc  | 美学 | FID | 总分
|  ----    |  ----  | ----  |----  |----  |----  |----  |
|  19   |  gauganv1_s256_vae+esrgan_x2+distsloss  | 16 | 0.899 | **5.1087** | **29.8064** | **0.5452**
|  20   |  gauganv1_s256_vae+esrgan_x2+focalfreqloss_w12.5  | 16 | 0.8866 | 5.0784 | 37.064 | 0.5041
|  21   |  gauganv1_s256_vae+esrgan_x2+haarloss  | 16 | 0.8811 | 5.1018 | 42.0288 | 0.4802
|  23   |  gauganv1_s256_vae+esrgan_x2+sploss_w12.5   | 16 | 0.8744 | 5.1517 | 44.5676 | 0.4676
|  25   |  gauganv1_s256_vae+esrgan_x2+focalfreqloss_w7.5  | 16 | 0.8911 | 5.0681 | 35.0263 | 0.5153
|  27   |  gauganv1_s256_vae+esrgan_x2+focalfreqloss_w7.5+dists_w12.5  | 16 | 0.8988 | 5.1371 | 32.7079 | 0.5333


#### tricks
|  ID    |  描述    | bs | mask_acc  | 美学 | FID | 总分
|  ----    |  ----  | ----  |----  |----  |----  |----  |
|  14   |  gauganv1_s256_vae_SA+esrgan_x2  | 4 | 0.8525 | 4.4462 | 55.2136 | 0.3804
|  15   |  gauganv1_s256_vae_gradAccu+esrgan_x2  | 4 | 0.836 | 4.5946 | 58.3473 | 0.3662
|  16   |  gauganv1_s512_vae_DperG2+esrgan_x2  | 4 | 0.8711 | 5.0448 | 35.7907 | 0.4994
|  24   |  gauganv1_s256_vae+esrgan_x2+modeseek+stylecycle+dualattn   | 8 | 0.8766 | 5.0718 | 34.0008 | 0.5116
|  26   |  gauganv1_s256_vae+esrgan_x2+DiffAug  | 8 | 0.8862 | 4.957 | 42.0479 | 0.4764
  
***
  
### SASAME
|  ID    |  描述    | bs | mask_acc  | 美学 | FID | 总分
|  ----    |  ----  | ----  |----  |----  |----  |----  |
|  1    | sasame_s256 + esrgan_x2  | 4 | 0.9193 | **5.1373** | 36.0882 | 0.5299
|  2    | sasame_s256 + esrgan_x2 + cxloss_w2.5 | 4 | 0.9218 | 5.0197 | 37.3302 | 0.5202
|  3    | sasame_s256 + esrgan_x2 + cxloss_w10 | 4 | **0.9255** | 5.034 | **34.6574** | **0.5353**
|  4    | sasame_s256 + esrgan_x2 + cxloss_w5 | 4 | 0.9192 | 5.0386 | 36.5592 | 0.5231
|  5    | sasame_s256 + esrgan_x2 + cxloss_w10 + dists_w12.5 | 8 | **0.9318** | **5.0687** | **32.8758** | **0.5489**
|  6    | sasame_s256 + esrgan_x2 + distsloss_w10 | 4 | 0.8906 | **5.1507** | 38.9355 | 0.5013
|  6    | sasame_s256 + esrgan_x2 + distsloss_w10 + seg | 8 | **0.9317** | 5.0236 | **35.6999** | **0.5336**
|  6    | sasame_s256 + esrgan_x2 + distsloss_w12.5 | 16 | 0.9015 | 5.046 | 43.6526 | 0.4814
  
***
  
### CoCosNetv1
|  ID    |  描述    | bs | mask_acc  | 美学 | FID | 总分
|  ----    |  ----  | ----  |----  |----  |----  |----  |
|  1   |  cocosnetv1_s256_trainR_testR+esrgan_x2 (jpg) | 4 | 0.7691 | 4.7557 | 46.5337 | 0.3885
|  2   |  cocosnetv1_s256_trainVGG_testIOU+esrgan_x2 (jpg) | 4 | 0.8738 | 4.6861 | 42.457 | 0.4561
|  3   |  cocosnetv1_s256_trainIOU_testIOU+esrgan_x2 (jpg) | 16 | 0.8776 | 4.7023 | 42.4705 | 0.4588
|  4   |  cocosnetv1_s256_trainIOU_testIOU+esrgan_x2 (png) | 16 | 0.8852 | **5.247** | **31.7785** | **0.5342**
|  5   |  cocosnetv1_s256_trainIOU_testIOU_ndf128+esrgan_x2 (png) | 16 | 0.8887 | **5.1397** | **31.5109** | **0.5327** 
|  6   |  cocosnetv1_s256_trainIOU_testIOU_ndf64+segw1+esrgan_x2 (png) | 16 | 0.8799 | **5.1514** | **33.1764** | **0.5206** 
 
***
  
### CoCosNetv2
|  ID    |  描述    | bs | mask_acc  | 美学 | FID | 总分
|  ----    |  ----  | ----  |----  |----  |----  |----  |
|  1   |  cocosnetv2_s512_ep60+IOU | 8 | 0.9124 | 5.3079 | 29.9293 | 0.5618
|  2   |  cocosnetv2_s512_ep100+dists50+maskaccIOU | 8 | 0.9159 | 5.3402 | 30.9899 | 0.5606
|  2   |  cocosnetv2_s512_ep100+dists50+IOU | 8 | 0.9121 | 5.3474 | 30.663 | 0.5601
|  3   |  cocosnetv2_s512_ep100+dists30 | 8 | 0.91 | 5.3069 | 29.8367 | 0.5607
|  4   |  cocosnetv2_s512_ep100+maskaccIOU | 8 | 0.9026 | 5.3086 | 30.0753 | 0.5552
|  5   |  cocosnetv2_s512_ep100+IOU | 8 | 0.9146 | 5.3013 | 31.3625 | 0.5563
|  6   |  cocosnetv2_s512_ep66+IOU | 8 | 0.9171 | 5.2917 | 30.8449 | 0.5598
|  7   |  cocosnetv2_s512_ep40+dists_w12.5 | 8 | 0.9032 | 5.2593 | 32.5123 | 0.5423
|  8   |  cocosnetv2_s512_ep50+dists_w12.5 | 8 | 0.9136 | 5.3127 | 30.9537 | 0.5581
|  9   |  cocosnetv2_s512_ep54+dists_w12.5 | 8 | 0.9025 | 5.2437 | 30.9394 | 0.5482
|  9   |  cocosnetv2_s512_ep60+dists_w12.5 | 8 | 0.9034 | 5.3071 | 31.3805 | 0.5497
  
***
  
### UNITE
|  ID    |  描述    | bs | mask_acc  | 美学 | FID | 总分
|  ----    |  ----  | ----  |----  |----  |----  |----  |
|  1   |  unite_s256_cxloss0.1+esrgan_x2 | 4 | 0.8822 | 5.2341 | 32.1044 | 0.5304
|  2   |  unite_s256_cxloss0.1_bicubic+esrgan_x2+cycleGAN | 4 | 0.876 | 5.1704 | 33.6479 | 0.5171
|  3   |  unite_s256_cxloss0.5_epoch200+cycleGAN | 4 | 0.903 | 5.1959 | 28.0596 | 0.5594
|  4   |  unite_s256_cxloss1_dists_epoch250+cycleGAN | 4 | 0.8919 | 5.2321 | 27.9106 | 0.5548
|  5   |  unite_s256_cxloss0.5+dists_w12.5+cycleGAN | 4 | 0.9084 | 5.2136 | 28.2823 | 0.5626
  
***
  

## 线下实验
### Gauganv1_s256_vae
|  ID    |  描述    |   FID | 
|  ----    |  ----  | ----  |
|  0    | baseline  |  44.78
|  1    | + mode seeking loss  |  38.98
|  2    | + dual attention | 41.65
|  3    | + style consistence |  39.16
|  4    | + lpips loss |  32.85
|  5    | + ema |  44.91
|  6    | + app loss |  45.84
|  7    | + haar loss |  43.75
|  8    | + spl loss |  44.95
|  9    | + msgmsdc loss |  44.74
|  10    | + pixelshuffle |  48.00
|  11    | + monce |  58.03
|  12    | + focalfreq loss w12.5 |  36.52
|  13    | + focalfreq loss w1 |  45.71
|  14    | + focalfreq loss w2.5 |  40.64
|  15    | + focalfreq loss w5 |  41.21
|  16    | + focalfreq loss w7.5 |  35.79
|  17    | + focalfreq loss w10 |  49.67
|  18    | + modeseek + stylecycle + dual attn |  34.8

### cocosnetv2
|  ID    |  描述    |   FID | 
|  ----    |  ----  | ----  |
|  1    | 60epoch  |   30.72
|  2    | 86epoch | 33.32
|  3    | 100epoch |  30.84

### UNITE
|  ID    |  描述    |   FID | 
|  ----    |  ----  | ----  |
|  1    | 150epoch_cx0.1  |   31.82
|  2    | 195epoch_cx0.5 | 30.66

## 分析
### **1.超分/分辨率**  
（1）实验1对比实验3：超分得到s1024高清图并不能显著提高美学；超分后mask_acc降低。  
（2）实验2对比实验3：gauganv1得到s256再使用超分得到s512，与gauganv1直接生成s512相比，mask_acc和FID**显著**变差、美学略有提升。直接通过gaugan生成512的mask_acc更高。
（3）实验1对比实验2：同样使用了超分，生成s1024与生成s512相比，s1024的mask_acc和FID更好、美学略差。  
### **2.VAE**  
（1）实验2对比实验4：使用vae将真实图片encode为随机变量（输入时未打乱）可以**显著**提升美学和FID、mask_acc基本不变。  
（2）以实验3为baseline，对比生成s256的小分辨率的实验2、实验4、实验8，vae+普通上采样、不使用vae+超分上采样均降低了性能，但两者结合在一起能超过baseline，比较**奇怪**。
### **3.风格迁移**  
（1）实验3对比实验5：使用cycleGAN可以**显著**提升FID。肉眼上看，主要是饱和度变化了，有些场景的细节有一些提升，但美学和mask_acc都下降了，比较**奇怪**。  
（2）实验5对比实验6：cycleGAN视觉上好看了，但美学并没有提高。迭代过程中，ep20和ep100时很蓝、颜色饱和度高。ep20时各项指标都不好。  
（3）实验21表明以gauganv1_s512训练的cycleGAN并不是万能提分的，用在cxloss上似乎不行，有待进一步探究，考虑以cxloss结果训练、或者换一个风格迁移模型。
### **4.损失函数**  
（1）实验11、12、13体现ContextualLoss能**显著**提升FID，这个Loss是针对未配对问题的，但却很有效果，值得思考；考虑CX的改进CoBi  
|  ID    |  原图    | ep20 | ep60  | ep100 | ep180 
|  ----    |  ----  | ----  |----  |----  |----  
| 1| <img src="https://github.com/22TonyFStark/Scene/raw/main/image/origin.jpg" width="300">| <img src="https://github.com/22TonyFStark/Scene/raw/main/image/ep20.jpg" width="300">| <img src="https://github.com/22TonyFStark/Scene/raw/main/image/ep60.jpg" width="300">| <img src="https://github.com/22TonyFStark/Scene/raw/main/image/ep100.jpg" width="300">| <img src="https://github.com/22TonyFStark/Scene/raw/main/image/ep180.jpg" width="300">  

（3）实验4对比实验7：cycleGAN可以进一步提升gauganv1_vae，把FID在原来降低的基础上进一步**显著**降低。

## 想法
1. backbone 替换成 resnet, vit, swin 等等, 相对应的vgg loss需要修改, 还需要考虑pretrained_model的问题
2. hinge loss修改成wgan-gp, wgan-lp等其他损失，主要想用来提高FID
3. 卷积层增加谱归一化，加入self-attention模块，类似于SNGAN、SAGAN，主要也是用来提高FID
4. 对于Gauganv1_VAE，encode的输入、SPADE的输入可以加入更多先验，比如instance map和纹理图等
5. label-smooth\self-attention\Resize Convolution\Multi-Scale Gradient
6. SIMM LOSS, FocalFrequency Loss
7. 训一个s64的微型分辨率gauganv1_vae，上采样到512作为instance map
## TODO
dzw:
1. 线下美学分数评估，线上线下gap
2. 想法 2 和想法 3 实验  
3. piq的dists loss + pertual loss/cx loss，减小FID

qzf:  
1. 实验：use_vae和cycleGAN一起使用能否更好 √  
2. 实验：gauganv1_s512_vae √  
3. 直接使用风格迁移的方法，cycleGAN、CUT https://github.com/taesungp/contrastive-unpaired-translation
4. 想法1
5. 想法4
## 细节
实验1-6使用的是默认的超参数，模型的细节为：
1. Gauganv1的Backbone
2. Gauganv1的Loss


### FID评估
python -m pytorch_fid train_img512 47.78 --device cuda:3 --dims 2048
|  ID    |  官网    | 自测 
|  ----    |  ----  | ----  |
|1	|29.8	|30.66960283|
|2	|33.2	|34.72185119|
|3  |34.66|35.98603935400291|
|4	|35.6	|37.73446877|
|5  |36.56|37.84640909149283|
|6	|37.3	|38.69920146|
|7	|42.3	|43.7506799|
|**8**|**42.69**	|**44.79961414**|
|9	|44.57	|44.95368953|
|10	|47.78	|45.63134|
|11	|51.21	|51.74274299|

<img src="https://github.com/22TonyFStark/Scene/raw/main/image/线下评估.png" width="512">
