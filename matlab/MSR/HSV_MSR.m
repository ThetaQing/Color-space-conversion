% MSR 算法在HSV空间的应用

close all; clear all; clc

%%
% RGB分解与合成测试
I=im2double(imread('rgb.png'));
subplot(321),imshow(I);
title('原始真彩色图像');

IR=I(:,:,1);
IG=I(:,:,2);
IB=I(:,:,3);

subplot(322),imshow(IR);
title('真彩色图像的红色分量');
subplot(324),imshow(IG);
title('真彩色图像的绿色分量');
subplot(326),imshow(IB);
title('真彩色图像的蓝色分量');
X=cat(3,IR,IG,IB);
subplot(325),imshow(I);
title('原始真彩色图像');
subplot(323),imshow(X);
title('合成真彩色图像');



%% 获取图像RGB分量并转换到HSV空间
I = im2double(imread('example.png'));
% 提取RGB
I_r = I(:,:,1);
I_g = I(:,:,2);
I_b = I(:,:,3);
I1 = im2uint8(cat(3,I_r,I_g,I_b));
I3(:,:,1) = I_r;
I3(:,:,2) = I_g;
I3(:,:,3) = I_b;
% 转换到HSV空间
F = rgb2hsv(I);
F_h = double(F(:,:,1));
F_s = double(F(:,:,2));
F_v = double(F(:,:,3));
F1 = cat(3,F_h,F_s,F_v);

I2 = hsv2rgb(F1);
F2 = rgb2hsv(I1);
% 显示
figure('name','RGB原图与cat合成');
subplot(121);imshow(I);
subplot(122);imshow(I1);
figure('name','HSV原图与cat合成');
subplot(121);imshow(F);
subplot(122);imshow(F1);
figure('name','cat合成后转换的RGB与HSV');
subplot(121);imshow(I2);
subplot(122);imshow(F2);

%% 高斯核函数求解
% 变换到频域
Hfft1 = fft2(F_h);
Sfft1 = fft2(F_s);
Vfft1 = fft2(F_v);
% 确定核函数参数

[m,n] = size(I_r);
sigma(1) = 1/3 * min(max(mean(I,3)) * 255);
sigma(2) = mean(max(mean(I,3))) * 255;
sigma(3) = max(max(mean(I,3))) * 255;
% 确定高斯核函数
f = zeros(m,n,3);
f(:,:,1) = fspecial('gaussian', [m, n], sigma(1));
f(:,:,2) = fspecial('gaussian', [m, n], sigma(2));
f(:,:,3) = fspecial('gaussian', [m, n], sigma(3));

% 对滤波器进行傅里叶变换
efft = zeros(m,n,3);
efft(:,:,1) = fft2(double(f(:,:,1)));
efft(:,:,2) = fft2(double(f(:,:,2)));
efft(:,:,3) = fft2(double(f(:,:,3)));
%%
% 求解亮度分量估计值
L_v = zeros(m,n,3);
L_v(:,:,1) = log(F_v.*f(:,:,1));
L_v(:,:,2) = log(F_v.*f(:,:,2));
L_v(:,:,3) = log(F_v.*f(:,:,3));
%%
% 求解反射分量
R_v = zeros(m,n,3);
R_v(:,:,1) = 0.33 * (log(F_v) - L_v(:,:,1));
R_v(:,:,2) = 0.33 * (log(F_v) - L_v(:,:,2));
R_v(:,:,3) = 0.34 * (log(F_v) - L_v(:,:,3));
%%
% 增强处理
G = zeros(m,n,3);

for i = 1 : 3    
    G(:,:,i) = (max(R_v,[],3)./R_v(:,:,i)).^(1-(sigma(i)./mean(sigma)));    
end

R_value = sum(G .* R_v,3);
%%
% 反对数变换
r_value = abs(exp(R_value));
HSV = cat(3,F_h,F_s,r_value);
HSV8 = im2uint8(HSV);
RGB1 = hsv2rgb(HSV);
RGB1 = im2uint8(RGB1);
figure;
subplot(221);imshow(F);
title('原始HSV图像');
subplot(222);imshow(HSV);
title('合成HSV图像');
subplot(223);imshow(HSV8);
title('uint8格式HSV图像');
subplot(224);imshow(RGB1);
title('HSV转rgb后改uint8格式')
%%
% 颜色恢复函数
C = zeros(m,n,3);
for i = 1 : 3
   C(:,:,i) = log(1 + (75 * I_rgb(:,:,i)./(I_rgb(:,:,i) + 10))); 
end
RGB2 = C .* RGB1;
RGB = 0.7 * (RGB2 - min(RGB2)).^0.45;
figure;
imshow(HSV);
figure;
imshow(RGB,[])


