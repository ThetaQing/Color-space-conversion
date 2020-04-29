% ��߶�Retinexͼ����ǿ
function MSR_v = MSR(v)
%%
% ��������
% img = imread('E:\Design\OpenCV\Color-space-conversion\img\paper1.png');
% r = img(:,:,1);
% g = img(:,:,2);
% b = img(:,:,3);

%%
% RGBתHSY
% [h,s,v] = rgb2hsy(r,g,b);
[m,n] = size(v);
% ȷ���˺�������
sigma = [15,80,200];
% ȷ����˹�˺���

f1 = fspecial('gaussian', [7, 7], sigma(1));
f2 = fspecial('gaussian', [11, 11], sigma(2));
f3 = fspecial('gaussian', [63, 63], sigma(3));
f4 = fspecial('gaussian',[m,n],sigma(3));

%%
% ������ȷ�������ֵ
% +1 ͼ��������matlab���������
L_v = zeros(m,n,4);
L_v(:,:,1) = log(convn(v+1,f1,'same'));
L_v(:,:,2) = log(convn(v+1,f2,'same'));
L_v(:,:,3) = log(convn(v+1,f3,'same'));
L_v(:,:,4) = log(convn(v+1,f4,'same'));

%%
% ��ⷴ�����
% ƫ��ֱ��Ӱ��Ч��
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
% figure('name','ת�����');
% imshow(RGB)
