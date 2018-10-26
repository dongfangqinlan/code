function [ pcaA,V] = fastPCA( A,k,mA)  
%主成份分析  A-样本矩阵，每行是一个样本，列是样本的维数  
% k-降至k维  
% mA-图像矩阵f_matrix每一列的均值排成一个行向量，即mean(f_matrix)
% pacA-降维后，训练样本在低维空间中的系数坐标表示 
% V-主成分分量，即低维空间当中的基  
m=size(A,1);  %m为读取图片的张数
Z=(A-repmat(mA,m,1));  %中心化样本矩阵
T=Z*Z';  
[V1,D]=eigs(T,k);  %计算T的最大的k个特征值和特征向量  
V=Z'*V1;         %协方差矩阵的特征向量  
for i=1:k       %特征向量单位化  
    l=norm(V(:,i));  
    V(:,i)=V(:,i)/l;  
end  
pcaA=Z*V;   %线性变换，降至k维  ，将中心化的矩阵投影到低维空间的基中，V就是低维空间的基
end 
