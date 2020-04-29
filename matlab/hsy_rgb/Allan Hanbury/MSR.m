% 多尺度Retinex图像增强
function MSR_v = MSR(v)
%%
% 单独测试
% img = imread('E:\Design\OpenCV\Color-space-conversion\img\paper1.png');
% r = img(:,:,1);
% g = img(:,:,2);
% b = img(:,:,3);

%%
% RGB转HSY
% [h,s,v] = rgb2hsy(r,g,b);
[m,n] = size(v);
% 确定核函数参数
sigma = [15,80,200];
% 确定高斯核函数

f1 = fspecial('gaussian', [7, 7], sigma(1));
f2 = fspecial('gaussian', [11, 11], sigma(2));
f3 = fspecial('gaussian', [63, 63], sigma(3));
f4 = fspecial('gaussian',[m,n],sigma(3));

%%
% 求解亮度分量估计值
% +1 图像坐标与matlab坐标的区别
L_v = zeros(m,n,4);
L_v(:,:,1) = log(convn(v+1,f1,'same'));
L_v(:,:,2) = log(convn(v+1,f2,'same'));
L_v(:,:,3) = log(convn(v+1,f3,'same'));
L_v(:,:,4) = log(convn(v+1,f4,'same'));

%%
% 求解反射分量
% 偏置直接影响效果
R_v = zeros(m,n,4);
R_v(:,:,1) = 0.3 * (log(v) - L_v(:,:,1)+0.6);
R_v(:,:,2) = 0.2 * (log(v) - L_v(:,:,2)+0.7);
R_v(:,:,3) = 0.3 * (log(v) - L_v(:,:,3)+0.8);
R_v(:,:,4) = 0.2 * (log(v) - L_v(:,:,4)+1.3);
%% MSR
MSR_v = exp(sum(R_v,3));
% SSR
% SSR_v = exp(R_v(:,:,2));
% newHSY = cat(3,h,s,MSR_v);
% RGB = hsy2rgb(newHSY);
% imwrite(RGB,'paper1_MSR_Filter4.jpg')
% figure('name','转换后的');
% imshow(RGB)
