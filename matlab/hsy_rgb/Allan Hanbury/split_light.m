% RGB 转 IHLS，并保存L分量
img = imread('E:\Design\OpenCV\Color-space-conversion\img\paper1.png');
r = img(:,:,1);
g = img(:,:,2);
b = img(:,:,3);

%%
% RGB转HSY
[h,s,y] = rgb2hsy(r,g,b);
imwrite(y,'E:\Design\OpenCV\Color-space-conversion\img\Light_paper1.png');
figure();
imshow(y);

%%
re_y = double(imread('E:\Design\OpenCV\Color-space-conversion\img\retinex_channel3_paper1.png'));
[r,g,b] = hsy2rgb(h,s,re_y);
RGB = cat(3,r,g,b);
figure();
imshow(RGB);
imwrite(RGB,'E:\Design\OpenCV\Color-space-conversion\img\retinex_IHLS_paper1.png')
