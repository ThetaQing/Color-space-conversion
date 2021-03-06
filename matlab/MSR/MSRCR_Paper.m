I = imread('paper1_2.png');
%带颜色恢复的多尺度Retinex
% 由于在采用SSR和MSR的过程会产生色差，MSRCR将MSR得到的结果按照一定的比例进行调整
% 以求恢复原来的比例数值，具体是通过引入颜色恢复因子C，
% 来弥补由于图像局部区域对比度增强而导致的图像颜色失真的缺陷。

size(I)
% RGB
R = I(:, :, 1);
G = I(:, :, 2);
B = I(:, :, 3);
R0 = double(R);
G0 = double(G);
B0 = double(B);
% R分量
[N1, M1] = size(R);

Rlog = log(R0+1);
Rfft2 = fft2(R0);

sigma1 = 128;  % 第一个参数，高斯滤波
F1 = fspecial('gaussian', [N1,M1], sigma1);
Efft1 = fft2(double(F1));

DR0 = Rfft2.* Efft1;
DR = ifft2(DR0);

DRlog = log(DR +1);
Rr1 = Rlog - DRlog;

sigma2 = 256;  % 第二个参数
F2 = fspecial('gaussian', [N1,M1], sigma2);
Efft2 = fft2(double(F2));

DR0 = Rfft2.* Efft2;
DR = ifft2(DR0);

DRlog = log(DR +1);
Rr2 = Rlog - DRlog;

sigma3 = 512;  % 第三个参数
F3 = fspecial('gaussian', [N1,M1], sigma3);
Efft3 = fft2(double(F3));

DR0 = Rfft2.* Efft3;
DR = ifft2(DR0);

DRlog = log(DR +1);
Rr3 = Rlog - DRlog;

Rr = (Rr1 + Rr2 +Rr3)/3;

a = 0.25;
II = imadd(R0, G0);
II = imadd(II, B0);
Ir = immultiply(R0, a);  % 图像乘法
C = imdivide(Ir, II);  % 图像除法
C = log(C+1);

Rr = immultiply(C, Rr);
EXPRr = exp(Rr);
MIN = min(min(EXPRr));
MAX = max(max(EXPRr));
EXPRr = (EXPRr - MIN)/(MAX - MIN);
EXPRr = adapthisteq(EXPRr);  % 直方图均衡化

% G分量
Glog = log(G0+1);
Gfft2 = fft2(G0);

DG0 = Gfft2.* Efft1;
DG = ifft2(DG0);

DGlog = log(DG +1);
Gg1 = Glog - DGlog;


DG0 = Gfft2.* Efft2;
DG = ifft2(DG0);

DGlog = log(DG +1);
Gg2 = Glog - DGlog;


DG0 = Gfft2.* Efft3;
DG = ifft2(DG0);

DGlog = log(DG +1);
Gg3 = Glog - DGlog;

Gg = (Gg1 + Gg2 +Gg3)/3;

Ig = immultiply(G0, a);
C = imdivide(Ig, II);
C = log(C+1);

Gg = immultiply(C, Gg);
EXPGg = exp(Gg);
MIN = min(min(EXPGg));
MAX = max(max(EXPGg));
EXPGg = (EXPGg - MIN)/(MAX - MIN);
EXPGg = adapthisteq(EXPGg);

% B通道的处理方法与R和G类似，这里省略
Blog = log(B0+1);
Bfft2 = fft2(B0);

DB0 = Bfft2.* Efft1;
DB = ifft2(DB0);

DBlog = log(DB +1);
Bb1 = Blog - DBlog;


DB0 = Bfft2.* Efft2;
DB = ifft2(DB0);

DBlog = log(DB +1);
Bb2 = Blog - DBlog;


DB0 = Bfft2.* Efft3;
DB = ifft2(DB0);

DBlog = log(DB +1);
Bb3 = Blog - DBlog;

Bb = (Bb1 + Bb2 +Bb3)/3;

Ig = immultiply(B0, a);
C = imdivide(Ig, II);
C = log(C+1);

Bb = immultiply(C, Bb);
EXPBb = exp(Bb);
MIN = min(min(EXPBb));
MAX = max(max(EXPBb));
EXPBb = (EXPBb - MIN)/(MAX - MIN);
EXPBb = adapthisteq(EXPBb);

% 最终
size(EXPRr)
size(EXPGg)
size(EXPBb)
result = cat(3, EXPRr, EXPGg, EXPBb); 


subplot(121), imshow(I);
subplot(122), imshow(result);
imwrite(result,'E:\Design\OpenCV\Color-space-conversion\img\处理结果\PaperMSRCR_RGB.jpg')