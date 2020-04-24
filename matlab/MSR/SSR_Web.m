% ���߶�Retinex
% �㷨˼·��
% 1 ��ȡԭʼͼ�񣬷ֱ�ȡ��RGB����ͨ������ֵ����ת���� double �͡�
% 2 ���ø�˹ģ��Ը���ͨ���Ľ��о����������
% 3 ��RGBͼ���Լ�������ͼ��ת���������򣬲������Ȼ�����÷�����ת���� ʵ����
% 4 �Ի�õĸ���ͨ����ͼ������������죬�ϲ�RGB�ͿɵĴ�����ͼ��
% ��������������������������������
% ��Ȩ����������ΪCSDN������@����@����ԭ�����£���ѭCC 4.0 BY-SA��ȨЭ�飬ת���븽��ԭ�ĳ������Ӽ���������
% ԭ�����ӣ�https://blog.csdn.net/qq_33668060/article/details/104265887
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

%  SSR�㷨
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
