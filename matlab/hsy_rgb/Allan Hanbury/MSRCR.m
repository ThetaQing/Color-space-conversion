% 在IHLS空间内做图像增强，多尺度Retinex带颜色恢复
function Retinex_v = MSRCR(v)

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


%%
% 求解反射分量
% 偏置直接影响效果
R_v = zeros(m,n,3);
R_v(:,:,1) = 0.3 * (log(v) - L_v(:,:,1)+0.6);
R_v(:,:,2) = 0.4 * (log(v) - L_v(:,:,2)+0.9);
R_v(:,:,3) = 0.3 * (log(v) - L_v(:,:,3)+0.8);

%% MSR
% 增强处理
G = zeros(m,n,3);

for i = 1 : 3    
    G(:,:,i) = (abs(max(R_v,[],3))./abs(R_v(:,:,i))).^(1-(sigma(i)./mean(sigma)));    
end

R_value = sum(G .* R_v,3);

%%
% 反对数变换
% R_value = sum(R_v);
Retinex_v = abs(exp(R_value));
% r_value = (R_value - min(min(R_value)))/(max(max(R_value)) - min(min(R_value)));
% r_value = (r_value - min(min(r_value)))/(max(max(r_value)) - min(min(r_value)));

