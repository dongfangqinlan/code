%%
%主程序，程序从此开始
clc,clear  
npersons=40;%选取40个人的脸  
global imgrow;  
global imgcol;  
imgrow=112;  
imgcol=92;  
%%
%读取训练数据 
disp('读取训练数据...')  
[f_matrix,train_label]=ReadFace(npersons,0);%读取训练数据  
nfaces=size(f_matrix,1);%样本人脸的数量  
disp('.................................................')  
%低维空间的图像是(nperson*5)*k的矩阵，每行代表一个主成分脸，每个脸20维特征  
%%
%对训练集进行降维处理
disp('训练数据PCA特征提取...')  
mA=mean(f_matrix);  
k=20;%降维至20维  
%按理来讲，协方差矩阵的维数是10304维，而训练集的样本数量只有200
%那么起作用的特征值对应的特征向量小于等于199个，其余的特征向量对应的特征值都为0
[train_pcaface,V]=fastPCA(f_matrix,k,mA);%主成分分析法特征提取  
disp('.................................................')  
 
%%
%显示主成分脸，即特征脸，低维的基
disp('显示主成分脸...')  
visualize(V)%显示主分量脸  ,即特征脸
disp('.................................................')  
  
%%
%低维训练集归一化
disp('训练特征数据归一化...')  
disp('.................................................')  
lowvec=min(train_pcaface);  
upvec=max(train_pcaface);  
train_scaledface = scaling( train_pcaface,lowvec,upvec);  
 
%%
%SVM样本训练
%本人的程序叫源程序有改动，使用的是libsvm工具箱进行SVM分类
disp('SVM样本训练...')  
disp('.................................................')  
 model = svmtrain(train_label,train_scaledface,'-t 0');
 
 %%
 %读取测试数据
disp('读取测试数据...')  
disp('.................................................')  
[test_facedata,test_facelabel]=ReadFace(npersons,1);  
  
%%
%测试数据降维
%test_pcatestface-测试数据低维空间的表示
disp('测试数据特征降维...')  
disp('.................................................')  
m=size(test_facedata,1);  
for i=1:m  
    test_facedata(i,:)=test_facedata(i,:)-mA;  
end  
test_pcatestface=test_facedata*V;  
  
%%
%测试数据归一化
disp('测试特征数据归一化...')  
disp('.................................................')  
scaled_testface = scaling( test_pcatestface,lowvec,upvec);  
  
%%
%利用训练集建立的模型，对测试集进行分类
disp('SVM样本分类...')  
disp('.................................................')  
[predict_label,accuracy,decision_values]=svmpredict(test_facelabel,scaled_testface,model);
 
%%
%人脸识别模块
disp('人脸识别模块')
disp('.................................................')  
recognition(mA,V,model) ;
