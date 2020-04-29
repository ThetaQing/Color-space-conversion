
img = imread('E:\Design\OpenCV\Color-space-conversion\img\original.png');
r = img(:,:,1);
g = img(:,:,2);
b = img(:,:,3);

%%
% RGB转HSY
[h,s,y] = rgb2hsy(r,g,b);
HSY = cat(3,h,s,y);
% RGB转HSV
[HSV_h,HSV_s,HSV_v] = rgb2hsv(img);
HSV = cat(3,HSV_h,HSV_s,HSV_v);

%% 
% Retinex增强
Retinex_v = MSR(y);
figure('name','增强前后的亮度分量对比');
subplot(121);imshow(y);
title('增强前y分量')
subplot(122);imshow(Retinex_v);
title('增强后y分量')

%%
% HSY转RGB
[new_r,new_g,new_b] = hsy2rgb(h,s,Retinex_v);
RGB = cat(3,new_r,new_g,new_b);
imwrite(RGB,'E:\Design\OpenCV\Color-space-conversion\img\处理结果\paper1MSR_偏置不等.jpg')
figure('name','RGB对比');
subplot(222);imshow(RGB);
title('IHLS处理之后的RGB');
subplot(221);imshow(img);
title('处理之前的RGB')
%%
% 显示
figure('name','原图');
subplot(221);imshow(r);
title('原图r分量')
subplot(222);imshow(g);
title('原图g分量')
subplot(223);imshow(img(:,:,3));
title('原图b分量');
subplot(224);imshow(img);
title('原图')

figure('name','RGB转HSY');
subplot(224);imshow(HSY);
title('合成之后HSV图')
subplot(221);imshow(h);
title('h分量')
imwrite(s,'IHLS空间的饱和度分量.png')
subplot(222);imshow(s);
title('s分量');
subplot(223);imshow(y);
title('y分量')


figure('name','HSY转RGB');
subplot(221);imshow(new_r);
title('反转后r分量')
subplot(222);imshow(new_g);
title('反转后g分量');
subplot(223);imshow(new_b);
title('反转后b分量');
subplot(224);imshow(RGB);
title('反转后合成RGB图');

% 显示
figure('name','RGB转HSV');
subplot(221);imshow(HSV_h);
title('反转后h分量')
imwrite(HSV_s,'HSV空间的饱和度分量.png')
subplot(222);imshow(HSV_s);
title('反转后s分量');
subplot(223);imshow(HSV_v);
title('反转后v分量');
subplot(224);imshow(HSV);
title('合成HSV图');
% RGB转HSL
[HLS_h,HLS_s,HLS_l] = colorspace('rgb->hls',img);
HLS = cat(3,HLS_h,HLS_s,HLS_l);
% 显示
figure('name','RGB转HSL');
subplot(221);imshow(HLS_h);
title('反转后h分量')
imwrite(HLS_s,'HLS空间中的饱和度分量.png')
subplot(222);imshow(HLS_s);
title('反转后s分量');
subplot(223);imshow(HLS_l);
title('反转后l分量');
subplot(224);imshow(HLS);
title('合成HLS图');
%%

figure;
subplot(221);imshow(img);
title('原图');
subplot(222);imshow(HSV_s);
title('HSV空间的饱和度分量');
subplot(223);imshow(HLS_s);
title('HLS空间的饱和度分量');
subplot(224);imshow(s);
title('IHLS空间的饱和度分量');
