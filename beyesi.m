load('data.txt');%������10560*52������˵��10560�����ݣ�ÿ��������52ά��һ��22�࣬ÿ��480����10560=480*22
data_predict=zeros(5280,52);%Ԥ������
for i=1:22
    tempdata{i}=data((i-1)*480+1:i*480,:);%��Щ���ݶ���52ά
    data_train{i}=tempdata{i}(1:240,:);%ѵ������
    data_predict((i-1)*240+1:i*240,:)=tempdata{i}(241:480,:);%Ԥ������
    for j=1:52
        [mu(i,j),sigma(i,j)]=normfit(data_train{i}(:,j));%��ѵ�����ݸ�������ά����ֵ�ͷ���
    end
end
%%
post=zeros(5280,22);%����������ֵΪ0
for i=1:5280
    for j=1:22
        prodt=ones(22,1);%prodt����Ȼ����������P��wi|x))
        for k=1:52
            prodt(j)=prodt(j)*normpdf(data_predict(i,k),mu(j,k),sigma(j,k));%�������ر�Ҷ˹����������Ȼ����p��x|wi)��p��x|wi)=p(x1|wi)*...*p(xn|wi)������nΪx��ά��
        end
        post(i,j)=prodt(j);%postΪ������ʣ�����P��wi|x)*p(x)=p(x|wi)*P(wi)������P��wi)��ͬ��p(x)������ȣ�����P��wi|x)=p(x|wi)
    end
    [a,Ind]=max(post(i,:));
    label(i)=Ind;
end
%%
%�жϲ��Լ����ж��������жϴ�������õ�������
c=0;
for i=1:5280
    b=ceil(i/240); 
    if label(i)~=b
        c=c+1;
    end
end
error=c/5280%������
%%
% figure;
% scatter(1:5280,data_predict(:,1),label,'filled');%��һ��������ɢ��ͼ
 