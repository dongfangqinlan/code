function recognition(mA,V,model) 
%函数作用：人脸识别模块，利用已经建好的模型，重新找一个样本进行识别
%mA-均值
%V-协方差矩阵特征向量
%model-通过SVM对训练集训练得出的已经建立好的模型

global imgrow;  
global imgcol;  

%弹出输入框，选择要识别的图片
select_person_num=str2double(cell2mat(inputdlg('请输入想要识别的人的编号(总共40个人)：')));%总共40个人
select_img_num=str2double(cell2mat(inputdlg('请输入此人图片的编号(总共10张)：')));%总共10张图

%对图片信息进行处理，化为1*10304的行向量
disp('读取选择的图片...')  
select_facepath=strcat('F:\MATLAB人脸识别\Face\facedata\s',num2str(select_person_num),'\',num2str(select_img_num),'.pgm');
select_img=imread(select_facepath);
select_matrix=zeros(1,imgrow*imgcol);
select_matrix(1,:)=select_img(:)';
select_matrix=(select_matrix-mA)*V;%PCA降维后的低维表示

%图形归一化
disp('规范化选择的图片...')  
select_matrix = scaling( select_matrix,min(select_matrix),max(select_matrix));  

%测试选择的图片，accuracy只有两个值，100%表示匹配正确，0%表示匹配错误
disp('测试选择的图片...')  
[select_predict_label,accuracy,decision_values]=svmpredict(select_person_num,select_matrix,model);

%显示原有图片和匹配图片进行比较
disp('显示选择的图片...')  
figure(2);
subplot(1,2,1);imshow(select_img);title('你选择的图片');
subplot(1,2,2);
imshow(imread(strcat('F:\MATLAB人脸识别\Face\facedata\s',num2str(select_predict_label),'\',num2str(1),'.pgm')));
title('匹配的图片');
