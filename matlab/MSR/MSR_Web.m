%多尺度Retinex
% 由于SSR需要在颜色保真度和细节保持上追求一个完美的平衡，而这个平衡不宜实现。
% MSR的提出就是解决该问题，它是针对一幅图像在不同尺度上利用高斯滤波处理，
% 然后将多个尺度下的图像进行加权叠加。
% 它其实就是 SSR 的一中叠加操作。
close all; clear all; clc
I = imread('example.png');
% 提取RGB
I_r = double(I(:,:,1));
I_g = double(I(:,:,2));
I_b = double(I(:,:,3));
% RGB对应的对数变换
I_r_log = log(I_r+1);  % 可能是因为图像坐标从0起
I_g_log = log(I_g+1);
I_b_log = log(I_b+1);
% RGB对应的傅里叶变换
Rfft1 = fft2(I_r);
Gfft1 = fft2(I_g);
Bfft1 = fft2(I_b);
% 参数设置
[m,n] = size(I_r);
sigma1 = 15;
sigma2 = 80;
sigma3 = 200;
% 定义高斯滤波器
f1 = fspecial('gaussian', [m, n], sigma1);
f2 = fspecial('gaussian', [m, n], sigma2);
f3 = fspecial('gaussian', [m, n], sigma3);
% 对滤波器进行傅里叶变换
efft1 = fft2(double(f1));
efft2 = fft2(double(f2));
efft3 = fft2(double(f3));
% 第一个高斯滤波器
% 对RGB分量进行高斯滤波并做傅里叶逆变换
D_r1 = ifft2(Rfft1.*efft1);
D_g1 = ifft2(Gfft1.*efft1);
D_b1 = ifft2(Bfft1.*efft1);
% 对滤波后RGB分量的图像取对数
D_r_log1 = log(D_r1 + 1);
D_g_log1 = log(D_g1 + 1);
D_b_log1 = log(D_b1 + 1);
% 原RGB分量的对数图像与高斯滤波后的对数图像做差
R1 = I_r_log - D_r_log1;
G1 = I_g_log - D_g_log1;
B1 = I_b_log - D_b_log1;
% 第二个高斯滤波器
D_r2 = ifft2(Rfft1.*efft2);
D_g2 = ifft2(Gfft1.*efft2);
D_b2 = ifft2(Bfft1.*efft2);
D_r_log2 = log(D_r2 + 1);
D_g_log2 = log(D_g2 + 1);
D_b_log2 = log(D_b2 + 1);
R2 = I_r_log - D_r_log2;
G2 = I_g_log - D_g_log2;
B2 = I_b_log - D_b_log2;
% 第三个高斯滤波器
D_r3 = ifft2(Rfft1.*efft3);
D_g3 = ifft2(Gfft1.*efft3);
D_b3 = ifft2(Bfft1.*efft3);
D_r_log3 = log(D_r3 + 1);
D_g_log3 = log(D_g3 + 1);
D_b_log3 = log(D_b3 + 1);
R3 = I_r_log - D_r_log3;
G3 = I_g_log - D_g_log3;
B3 = I_b_log - D_b_log3;
% 从上述三个参数处理后的RGB分量合成新的RGB分量
R = 0.1*R1 + 0.4*R2 + 0.5*R3;
G = 0.1*G1 + 0.4*G2 + 0.5*G3;
B = 0.1*B1 + 0.4*B2 + 0.5*B3;
% 从对数形式恢复成一般表达
R = exp(R);
G = exp(G);
B = exp(B);
% 最小值和最大值
MIN = min(min(R)); 
MAX = max(max(R));
% 求出新的RGB分量并进行直方图均衡
R = (R - MIN)/(MAX - MIN);
R = adapthisteq(R);

MIN = min(min(G)); 
MAX = max(max(G));
G = (G - MIN)/(MAX - MIN);
G = adapthisteq(G);
MIN = min(min(B)); 
MAX = max(max(B));
B = (B - MIN)/(MAX - MIN);
B = adapthisteq(B);
% 串联RGB图像形成新图像
J = cat(3, R, G, B);

figure;
imwrite(J,'MSR.png')
subplot(121);imshow(I);
subplot(122);imshow(J,[]);

figure;imshow(J)
