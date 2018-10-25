function [ scaledface] = scaling( faceMat,lowvec,upvec )  
%特征数据规范化  
%即是将同一个样本中的不同维度归一化
%因为因为对于不同的属性，如果不归一化是不具有比较性的，两者不在一个量级上
%输入——faceMat需要进行规范化的图像数据，  
%                lowvec原来图像数据中的最小值  
%                upvec原来图像数据中的最大值  
upnew=1;  
lownew=-1;  
[m,n]=size(faceMat);  
scaledface=zeros(m,n);  
for i=1:m  
    scaledface(i,:)=lownew+(faceMat(i,:)-lowvec)./(upvec-lowvec)*(upnew-lownew);  
    %将图像数据中一个样本的不同维度的值，最小值和最大值规范到-1和1，其他值按比例规范到（-1,1）
end  
end 
