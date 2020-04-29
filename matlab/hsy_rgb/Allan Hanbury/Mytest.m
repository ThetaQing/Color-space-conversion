
img = imread('E:\Design\OpenCV\Color-space-conversion\img\original.png');
r = img(:,:,1);
g = img(:,:,2);
b = img(:,:,3);

%%
% RGBתHSY
[h,s,y] = rgb2hsy(r,g,b);
HSY = cat(3,h,s,y);
% RGBתHSV
[HSV_h,HSV_s,HSV_v] = rgb2hsv(img);
HSV = cat(3,HSV_h,HSV_s,HSV_v);

%% 
% Retinex��ǿ
Retinex_v = MSR(y);
figure('name','��ǿǰ������ȷ����Ա�');
subplot(121);imshow(y);
title('��ǿǰy����')
subplot(122);imshow(Retinex_v);
title('��ǿ��y����')

%%
% HSYתRGB
[new_r,new_g,new_b] = hsy2rgb(h,s,Retinex_v);
RGB = cat(3,new_r,new_g,new_b);
imwrite(RGB,'E:\Design\OpenCV\Color-space-conversion\img\������\paper1MSR_ƫ�ò���.jpg')
figure('name','RGB�Ա�');
subplot(222);imshow(RGB);
title('IHLS����֮���RGB');
subplot(221);imshow(img);
title('����֮ǰ��RGB')
%%
% ��ʾ
figure('name','ԭͼ');
subplot(221);imshow(r);
title('ԭͼr����')
subplot(222);imshow(g);
title('ԭͼg����')
subplot(223);imshow(img(:,:,3));
title('ԭͼb����');
subplot(224);imshow(img);
title('ԭͼ')

figure('name','RGBתHSY');
subplot(224);imshow(HSY);
title('�ϳ�֮��HSVͼ')
subplot(221);imshow(h);
title('h����')
imwrite(s,'IHLS�ռ�ı��Ͷȷ���.png')
subplot(222);imshow(s);
title('s����');
subplot(223);imshow(y);
title('y����')


figure('name','HSYתRGB');
subplot(221);imshow(new_r);
title('��ת��r����')
subplot(222);imshow(new_g);
title('��ת��g����');
subplot(223);imshow(new_b);
title('��ת��b����');
subplot(224);imshow(RGB);
title('��ת��ϳ�RGBͼ');

% ��ʾ
figure('name','RGBתHSV');
subplot(221);imshow(HSV_h);
title('��ת��h����')
imwrite(HSV_s,'HSV�ռ�ı��Ͷȷ���.png')
subplot(222);imshow(HSV_s);
title('��ת��s����');
subplot(223);imshow(HSV_v);
title('��ת��v����');
subplot(224);imshow(HSV);
title('�ϳ�HSVͼ');
% RGBתHSL
[HLS_h,HLS_s,HLS_l] = colorspace('rgb->hls',img);
HLS = cat(3,HLS_h,HLS_s,HLS_l);
% ��ʾ
figure('name','RGBתHSL');
subplot(221);imshow(HLS_h);
title('��ת��h����')
imwrite(HLS_s,'HLS�ռ��еı��Ͷȷ���.png')
subplot(222);imshow(HLS_s);
title('��ת��s����');
subplot(223);imshow(HLS_l);
title('��ת��l����');
subplot(224);imshow(HLS);
title('�ϳ�HLSͼ');
%%

figure;
subplot(221);imshow(img);
title('ԭͼ');
subplot(222);imshow(HSV_s);
title('HSV�ռ�ı��Ͷȷ���');
subplot(223);imshow(HLS_s);
title('HLS�ռ�ı��Ͷȷ���');
subplot(224);imshow(s);
title('IHLS�ռ�ı��Ͷȷ���');
