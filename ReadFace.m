function [f_matrix,realclass]=ReadFace(n_persons,flag)
%40人，每人10幅图，图像大小为112*92像素。每个人有10幅照片，前5幅当作训练集，后5幅当作测试集

% flag是一个标识变量，当flag为0时，表示输入为训练集，flag为1时，表示输入为测试集

%   n_persons标志着你想要识别的人脸个数

%   realclass是一个n_person*5行，1列的列向量。
%   realclass即是数据的标签，无论训练集测试集都进行了标签处理

%   f_matrix是一个n_person*5行，112*92列的矩阵  每一行便是每一张图片的灰度数据
%   将每一张图片列向量排成一个列向量后转置得到放入f_matrix各行当中
 
imgrow=112;imgcol=92;
global imgrow;   %载入图片行数
global imgcol;    %载入图片列数
realclass=zeros(n_persons*5,1);
f_matrix=zeros(n_persons*5,imgrow*imgcol);
for i=1:n_persons
    %路径设置
    %函数num2str(i)说明：将数字转化为字符
    facepath=strcat('D:\matlab文件\renlianshibie\faces\s',num2str(i),'\');  %路径因不同情况而定
    cachepath=facepath;
    for j=1:5
        facepath=cachepath;
        if flag==0     %函数strcat(a,b,...)说明:将输入字符a,b...连接成单个字符
            facepath=strcat(facepath,num2str(j));
        else
            facepath=strcat(facepath,num2str(j+5));
        end
        realclass((i-1)*5+j)=i;
        facepath=strcat(facepath,'.pgm'); %函数imread说明：读取输入路径的图片，将每个像素灰度值保存在输出的矩阵中
        img=imread(facepath);
        f_matrix((i-1)*5+j,:)=img(:)';
    end
end
end
