# Scene
## 实验
|  ID    |  描述    | bs | mask_acc  | 美学 | FID | 总分
|  ----    |  ----  | ----  |----  |----  |----  |----  |
|  1    | gauganv1_s512+esrgan_x2  | 12 | 0.8808 | 4.9649 | 47.3168 | 0.4507
|  2    | gauganv1_s256+esrgan_x2  | 8 | 0.8718 | 5.0146 | 52.2234 | 0.4268
|  3    | gauganv1_s512  | 12 | **0.9063** | 4.9318 | 47.7813 | 0.4601
|  4    | gauganv1_s256_vae+esrgan_x2  | 8 | 0.8714 | **5.0663** | 42.69 | 0.4704
|  5    | gauganv1_s512+cycleGAN_ep180  | 6 | 0.8833 | 4.9263 | 41.567 | 0.4757
|  6    | gauganv1_s512+cycleGAN_ep20  | 6 | 0.8877 | 4.8457 | 49.1204 | 0.4409
|  7    | gauganv1_s256_vae+esrgan_x2+cycleGAN_ep180  | 6 | 0.8714 | 4.8706 | 36.3846| 0.4894
|  8    | gauganv1_s256_vae+bicubic_x2  | 8 | 0.8828 | 4.9664 | 51.2145| 0.4346
|  9    | gauganv1_s256_vae+esrgan_x2+cxloss_w10.0  | 4 | **0.9059** | 4.9234 | 35.0399| 0.5172
|  10    | gauganv1_s512_vae  | 12 | 0.9007 | 5.0321 | 37.0918| 0.5099
|  11   |  gauganv1_s256_vae+esrgan_x2+cxloss_w5  | 4 | 0.9043 | 4.8984 | 35.5186 | 0.513
|  12   |  gauganv1_s256_vae+esrgan_x2+cxloss_w2.5  | 4 | 0.9043 | **5.0655** | **33.2812** | **0.5307**
|  13   |  gauganv1_s256_vae+esrgan_x2+cxloss_w2.5_L2  | 4 | 0.9128 | 5.0504 | **32.6841** | **0.5377**
|  14   |  cocosnetv1_s256_trainR_testR+esrgan_x2  | 4 | 0.7691 | 4.7557 | 46.5337 | 0.3885
|  15   |  cocosnetv1_s256_trainVGG_testIOU+esrgan_x2  | 4 | 0.8738 | 4.6861 | 42.457 | 0.4561
|  17   |  gauganv1_s256_vae_SA+esrgan_x2  | 4 | 0.8525 | 4.4462 | 55.2136 | 0.3804
|  18   |  gauganv1_s256_vae_gradAccu+esrgan_x2  | 4 | 0.836 | 4.5946 | 58.3473 | 0.3662
|  19   |  gauganv1_s512_vae_DperG2+esrgan_x2  | 4 | 0.8711 | 5.0448 | 35.7907 | 0.4994
|  20   |  gauganv1_s256_vae+esrgan_x2+cxloss_smoothL1  | 4 | 0.8969 | 4.849 | 35.6237 | 0.5062
|  21   |  gauganv1_s256_vae+esrgan_x2+cxloss_w2.5_L2+cycleGAN  | 4 | 0.8844 | 4.9047 | 36.4918 | 0.4977

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
