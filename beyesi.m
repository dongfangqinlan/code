load('data.txt');%数据是10560*52，就是说有10560个数据，每个数据有52维，一共22类，每类480个。10560=480*22
data_predict=zeros(5280,52);%预测数据
for i=1:22
    tempdata{i}=data((i-1)*480+1:i*480,:);%这些数据都有52维
    data_train{i}=tempdata{i}(1:240,:);%训练数据
    data_predict((i-1)*240+1:i*240,:)=tempdata{i}(241:480,:);%预测数据
    for j=1:52
        [mu(i,j),sigma(i,j)]=normfit(data_train{i}(:,j));%求训练数据各变量（维）均值和方差
    end
end
%%
post=zeros(5280,22);%后验概率设初值为0
for i=1:5280
    for j=1:22
        prodt=ones(22,1);%prodt是似然函数（即是P（wi|x))
        for k=1:52
            prodt(j)=prodt(j)*normpdf(data_predict(i,k),mu(j,k),sigma(j,k));%根据朴素贝叶斯法来计算似然函数p（x|wi)，p（x|wi)=p(x1|wi)*...*p(xn|wi)，其中n为x的维数
        end
        post(i,j)=prodt(j);%post为后验概率，由于P（wi|x)*p(x)=p(x|wi)*P(wi)，但是P（wi)相同，p(x)假设相等，所以P（wi|x)=p(x|wi)
    end
    [a,Ind]=max(post(i,:));
    label(i)=Ind;
end
%%
%判断测试集中有多少数据判断错误进而得到错误率
c=0;
for i=1:5280
    b=ceil(i/240); 
    if label(i)~=b
        c=c+1;
    end
end
error=c/5280%错误率
%%
% figure;
% scatter(1:5280,data_predict(:,1),label,'filled');%第一个变量的散点图
 