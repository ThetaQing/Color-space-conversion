%��߶�Retinex
% ����SSR��Ҫ����ɫ����Ⱥ�ϸ�ڱ�����׷��һ��������ƽ�⣬�����ƽ�ⲻ��ʵ�֡�
% MSR��������ǽ�������⣬�������һ��ͼ���ڲ�ͬ�߶������ø�˹�˲�����
% Ȼ�󽫶���߶��µ�ͼ����м�Ȩ���ӡ�
% ����ʵ���� SSR ��һ�е��Ӳ�����
close all; clear all; clc
I = imread('example.png');
% ��ȡRGB
I_r = double(I(:,:,1));
I_g = double(I(:,:,2));
I_b = double(I(:,:,3));
% RGB��Ӧ�Ķ����任
I_r_log = log(I_r+1);  % ��������Ϊͼ�������0��
I_g_log = log(I_g+1);
I_b_log = log(I_b+1);
% RGB��Ӧ�ĸ���Ҷ�任
Rfft1 = fft2(I_r);
Gfft1 = fft2(I_g);
Bfft1 = fft2(I_b);
% ��������
[m,n] = size(I_r);
sigma1 = 15;
sigma2 = 80;
sigma3 = 200;
% �����˹�˲���
f1 = fspecial('gaussian', [m, n], sigma1);
f2 = fspecial('gaussian', [m, n], sigma2);
f3 = fspecial('gaussian', [m, n], sigma3);
% ���˲������и���Ҷ�任
efft1 = fft2(double(f1));
efft2 = fft2(double(f2));
efft3 = fft2(double(f3));
% ��һ����˹�˲���
% ��RGB�������и�˹�˲���������Ҷ��任
D_r1 = ifft2(Rfft1.*efft1);
D_g1 = ifft2(Gfft1.*efft1);
D_b1 = ifft2(Bfft1.*efft1);
% ���˲���RGB������ͼ��ȡ����
D_r_log1 = log(D_r1 + 1);
D_g_log1 = log(D_g1 + 1);
D_b_log1 = log(D_b1 + 1);
% ԭRGB�����Ķ���ͼ�����˹�˲���Ķ���ͼ������
R1 = I_r_log - D_r_log1;
G1 = I_g_log - D_g_log1;
B1 = I_b_log - D_b_log1;
% �ڶ�����˹�˲���
D_r2 = ifft2(Rfft1.*efft2);
D_g2 = ifft2(Gfft1.*efft2);
D_b2 = ifft2(Bfft1.*efft2);
D_r_log2 = log(D_r2 + 1);
D_g_log2 = log(D_g2 + 1);
D_b_log2 = log(D_b2 + 1);
R2 = I_r_log - D_r_log2;
G2 = I_g_log - D_g_log2;
B2 = I_b_log - D_b_log2;
% ��������˹�˲���
D_r3 = ifft2(Rfft1.*efft3);
D_g3 = ifft2(Gfft1.*efft3);
D_b3 = ifft2(Bfft1.*efft3);
D_r_log3 = log(D_r3 + 1);
D_g_log3 = log(D_g3 + 1);
D_b_log3 = log(D_b3 + 1);
R3 = I_r_log - D_r_log3;
G3 = I_g_log - D_g_log3;
B3 = I_b_log - D_b_log3;
% ��������������������RGB�����ϳ��µ�RGB����
R = 0.1*R1 + 0.4*R2 + 0.5*R3;
G = 0.1*G1 + 0.4*G2 + 0.5*G3;
B = 0.1*B1 + 0.4*B2 + 0.5*B3;
% �Ӷ�����ʽ�ָ���һ����
R = exp(R);
G = exp(G);
B = exp(B);
% ��Сֵ�����ֵ
MIN = min(min(R)); 
MAX = max(max(R));
% ����µ�RGB����������ֱ��ͼ����
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
% ����RGBͼ���γ���ͼ��
J = cat(3, R, G, B);

figure;
imwrite(J,'MSR.png')
subplot(121);imshow(I);
subplot(122);imshow(J,[]);

figure;imshow(J)
