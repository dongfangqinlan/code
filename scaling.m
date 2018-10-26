function [ scaledface] = scaling( faceMat,lowvec,upvec )  
% lowvec原来图像数据中的最小值  
% upvec原来图像数据中的最大值  
upnew=1;  
lownew=-1;  
[m,n]=size(faceMat);  
scaledface=zeros(m,n);  
for i=1:m  
    scaledface(i,:)=lownew+(faceMat(i,:)-lowvec)./(upvec-lowvec)*(upnew-lownew);  
  %将图像数据中一个样本的不同维度的值，最小值和最大值规范到-1和1，其他值按比例规范到（-1,1）
end  
end 
