% ��IHLS�ռ�����ͼ����ǿ����߶�Retinex����ɫ�ָ�
function Retinex_v = MSRCR(v)

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


%%
% ��ⷴ�����
% ƫ��ֱ��Ӱ��Ч��
R_v = zeros(m,n,3);
R_v(:,:,1) = 0.3 * (log(v) - L_v(:,:,1)+0.6);
R_v(:,:,2) = 0.4 * (log(v) - L_v(:,:,2)+0.9);
R_v(:,:,3) = 0.3 * (log(v) - L_v(:,:,3)+0.8);

%% MSR
% ��ǿ����
G = zeros(m,n,3);

for i = 1 : 3    
    G(:,:,i) = (abs(max(R_v,[],3))./abs(R_v(:,:,i))).^(1-(sigma(i)./mean(sigma)));    
end

R_value = sum(G .* R_v,3);

%%
% �������任
% R_value = sum(R_v);
Retinex_v = abs(exp(R_value));
% r_value = (R_value - min(min(R_value)))/(max(max(R_value)) - min(min(R_value)));
% r_value = (r_value - min(min(r_value)))/(max(max(r_value)) - min(min(r_value)));

