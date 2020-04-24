% 单尺度Retinex
% 算法思路：
% 1 读取原始图像，分别取出RGB三个通道的数值矩阵，转化成 double 型。
% 2 利用高斯模板对各个通道的进行卷积操作处理。
% 3 将RGB图像以及卷积后的图像转化到对数域，并相减，然后利用反对数转化到 实数域。
% 4 对获得的各个通道的图像进行线性拉伸，合并RGB就可的处理后的图像。
% ————————————————
% 版权声明：本文为CSDN博主「@阿文@」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
% 原文链接：https://blog.csdn.net/qq_33668060/article/details/104265887
close all; clear all; clc
I = imread('example.png');
I_r = double(I(:,:,1));
I_g = double(I(:,:,2));
I_b = double(I(:,:,3));

I_r_log = log(I_r+1);
I_g_log = log(I_g+1);
I_b_log = log(I_b+1);

Rfft1 = fft2(I_r);
Gfft1 = fft2(I_g);
Bfft1 = fft2(I_b);

%  SSR算法
[m,n] = size(I_r);
sigma = 200;
f = fspecial('gaussian', [m, n], sigma);
efft1 = fft2(double(f));

D_r = ifft2(Rfft1.*efft1);
D_g = ifft2(Gfft1.*efft1);
D_b = ifft2(Bfft1.*efft1);

D_r_log = log(D_r + 1);
D_g_log = log(D_g + 1);
D_b_log = log(D_b + 1);

R = I_r_log - D_r_log;
G = I_g_log - D_g_log;
B = I_b_log - D_b_log;

R = exp(R);
MIN = min(min(R)); 
MAX = max(max(R));
R = (R - MIN)/(MAX - MIN);
R = adapthisteq(R);
G = exp(G);
MIN = min(min(G)); 
MAX = max(max(G));
G = (G - MIN)/(MAX - MIN);
G = adapthisteq(G);
B = exp(B);
MIN = min(min(B)); 
MAX = max(max(B));
B = (B - MIN)/(MAX - MIN);
B = adapthisteq(B);

J = cat(3, R, G, B);

figure;
imwrite(J,'SSR.png')
subplot(121);imshow(I);
subplot(122);imshow(J);
figure;imshow(J)
