% MSR �㷨��HSV�ռ��Ӧ��

close all; clear all; clc


%% ��ȡͼ��RGB������ת����HSV�ռ�
I = im2double(imread('paper1_2.png'));
% ��ȡRGB
I_r = I(:,:,1);
I_g = I(:,:,2);
I_b = I(:,:,3);
I1 = im2uint8(cat(3,I_r,I_g,I_b));
I3(:,:,1) = I_r;
I3(:,:,2) = I_g;
I3(:,:,3) = I_b;
% ת����HSV�ռ�
F = rgb2hsv(I);
F_h = double(F(:,:,1));
F_s = double(F(:,:,2));
F_v = double(F(:,:,3));
F1 = cat(3,F_h,F_s,F_v);

I2 = hsv2rgb(F1);
F2 = rgb2hsv(I1);
% ��ʾ
figure('name','RGBԭͼ��cat�ϳ�');
subplot(121);imshow(I);
subplot(122);imshow(I1);
figure('name','HSVԭͼ��cat�ϳ�');
subplot(121);imshow(F);
subplot(122);imshow(F1);
figure('name','cat�ϳɺ�ת����RGB��HSV');
subplot(121);imshow(I2);
subplot(122);imshow(F2);
figure('name','��������')
subplot(221);imshow(F_h);
title('H����');
subplot(222);imshow(F_s);
title('S����');
subplot(223);imshow(F_v);
title('V����')
%% ��˹�˺������
% �任��Ƶ��
Hfft1 = fft2(F_h);
Sfft1 = fft2(F_s);
Vfft1 = fft2(F_v);
% ȷ���˺�������

[m,n] = size(I_r);
sigma(1) = 1/3 * min(max(mean(I,3)) * 255);
sigma(2) = mean(max(mean(I,3))) * 255;
sigma(3) = max(max(mean(I,3))) * 255;
% ȷ����˹�˺���
f = zeros(m,n,3);
f(:,:,1) = fspecial('gaussian', [m, n], sigma(1));
f(:,:,2) = fspecial('gaussian', [m, n], sigma(2));
f(:,:,3) = fspecial('gaussian', [m, n], sigma(3));

% ���˲������и���Ҷ�任
efft = zeros(m,n,3);
efft(:,:,1) = fft2(double(f(:,:,1)));
efft(:,:,2) = fft2(double(f(:,:,2)));
efft(:,:,3) = fft2(double(f(:,:,3)));
%%
% ������ȷ�������ֵ
L_v = zeros(m,n,3);
L_v(:,:,1) = log(convn(F_v,f(:,:,1),'same'));
L_v(:,:,2) = log(convn(F_v,f(:,:,2),'same'));
L_v(:,:,3) = log(convn(F_v,f(:,:,3),'same'));

%%
% ��ⷴ�����
R_v = zeros(m,n,3);
R_v(:,:,1) = 0.3 * (log(F_v) - L_v(:,:,1));
R_v(:,:,2) = 0.2 * (log(F_v) - L_v(:,:,2));
R_v(:,:,3) = 0.5 * (log(F_v) - L_v(:,:,3));

%%
% ��ǿ����
G = zeros(m,n,3);

for i = 1 : 3    
    G(:,:,i) = (abs(max(R_v,[],3))./abs(R_v(:,:,i))).^(1-(sigma(i)./mean(sigma)));    
end

R_value = sum(G .* R_v,3);
figure('name','��ǿ�����ķ������')
imshow(R_value);
%%
% �������任
%r_value = abs(exp(R_value));
r_value = (R_value - min(min(R_value)))/(max(max(R_value)) - min(min(R_value)));
%r_value = (r_value - min(min(r_value)))/(max(max(r_value)) - min(min(r_value)));

HSV = cat(3,F_h,F_s,r_value);

RGB1 = hsv2rgb(HSV);

figure;
subplot(221);imshow(F);
title('ԭʼHSVͼ��');
subplot(222);imshow(HSV);
title('�ϳ�HSVͼ��');
subplot(223);imshow(hsv2rgb(F));
title('ԭHSVͼ��ת��RGBͼ��');
subplot(224);imshow(RGB1);
title('HSVתrgb���uint8��ʽ')
%%
% ��ɫ�ָ�����
C = zeros(m,n,3);
for i = 1 : 3
   C(:,:,i) = log(1 + (75 * I(:,:,i)./(I(:,:,i) + 10))); 
end
RGB2 = C .* RGB1;
RGB = 0.7 * ((RGB2 - min(RGB2,[],3)).^0.45);
imwrite(RGB2,'paper1_1_20200426RGB.png')
figure('name','��ɫ�ָ����HSVͼ��');
imshow(HSV);
figure('name','��ɫ�ָ����HSVתRGBͼ��');
subplot(121);imshow(RGB,[])
title('��ǿͼ�����')
subplot(122);imshow(RGB2);
title('��ǿͼ����ǰ')


