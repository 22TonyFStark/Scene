# Scene
## 实验
|  ID    |  描述    | batch_size | mask_acc  | 美学 | FID | 总分
|  ----    |  ----  | ----  |----  |----  |----  |----  |
|  1    | gauganv1_s512+esrgan_x2  | 12 | 0.8808 | 4.9649 | 47.3168 | 0.4507
|  2    | gauganv1_s256+esrgan_x2  | 8 | 0.8718 | **5.0146** | 52.2234 | 0.4268
|  3    | gauganv1_s512  | 12 | **0.9063** | 4.9318 | 47.7813 | 0.4601
|  4    | gauganv1_s256_vae+esrgan_x2  | 8 | 0.8714 | **5.0663** | **42.69** | **0.4704**
|  5    | gauganv1_s512+cycleGAN_ep180  | 6 | 0.8833 | 4.9263 | **41.567** | **0.4757**
|  6    | gauganv1_s512+cycleGAN_ep20  | 6 | **0.8877** | 4.8457 | 49.1204 | 0.4409

## 分析
### **1.超分/分辨率**  
（1）实验1对比实验3：超分得到s1024高清图并不能显著提高美学；超分后mask_acc降低。  
（2）实验2对比实验3：gauganv1得到s256再使用超分得到s512，与gauganv1直接生成s512相比，mask_acc和FID**显著**变差、美学略有提升。直接通过gaugan生成512的mask_acc更高。
（3）实验1对比实验2：同样使用了超分，生成s1024与生成s512相比，s1024的mask_acc和FID更好、美学略差。  
### **2.VAE**  
（1）实验2对比实验4：使用vae将真实图片encode为随机变量（输入时未打乱）可以**显著**提升美学和FID、mask_acc基本不变。  
### **3.风格迁移**  
（1）实验3对比实验5：使用cycleGAN可以**显著**提升FID。肉眼上看，主要是饱和度变化了，有些场景的细节有一些提升，但美学和mask_acc都下降了，比较**奇怪**。  
（2）实验5对比实验6：cycleGAN视觉上好看了，但美学并没有提高。迭代过程中，ep20和ep100时很蓝、颜色饱和度高。ep20时各项指标都不好。  
|  ID    |  原图    | ep20 | ep60  | ep100 | ep180 
|  ----    |  ----  | ----  |----  |----  |----  
| 1| <img src="https://github.com/22TonyFStark/Scene/raw/main/image/origin.jpg" width="300">| <img src="https://github.com/22TonyFStark/Scene/raw/main/image/ep20.jpg" width="300">| <img src="https://github.com/22TonyFStark/Scene/raw/main/image/ep60.jpg" width="300">| <img src="https://github.com/22TonyFStark/Scene/raw/main/image/ep100.jpg" width="300">| <img src="https://github.com/22TonyFStark/Scene/raw/main/image/ep180.jpg" width="300">

## 想法
1. backbone 替换成 resnet, vit, swin 等等, 相对应的vgg loss需要修改, 还需要考虑pretrained_model的问题
2. hinge loss修改成wgan-gp, wgan-lp等其他损失，主要想用来提高FID
3. 卷积层增加谱归一化，加入self-attention模块，类似于SNGAN、SAGAN，主要也是用来提高FID
4. 对于Gauganv1_VAE，encode的输入、SPADE的输入可以加入更多先验，比如instance map和纹理图等
## TODO
dzw:
1. 线下美学分数评估，线上线下gap
2. 想法 2 和想法 3 实验  

qzf:  
1. 实验：use_vae和cycleGAN一起使用能否更好
2. 实验：gauganv1_s512_vae
3. 想法1
4. 想法4
## 细节
实验1-6使用的是默认的超参数，模型的细节为：
1. Gauganv1的Backbone
2. Gauganv1的Loss
