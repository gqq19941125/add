function [f1,auc]=ADD(D)
 %% 人工数据集 %%%%%%%%%%%%%%%%%%%%%%%%%
 X=D(:,1:2);
 Y=D(:,3);
 
%% 真实数据集 %%%%%%%%%%%%%%%%%%%%%%%%
%  %D=processdcle;
%  X1=D(:,1:13);
%  Y=D(:,14);

% %  %D=heartdisease_xiugai2;
%  X1=D(:,1:13);
%  Y=D(:,14);

% % %D=ionosphere;
%  X1=D(:,1:34);
%  Y=D(:,35);

% % %D=iris;  %0.8000 0.8400
%  X1=D(:,1:4);
%  Y=D(:,5);

 % % %D=audit_risk;
%  X1=D(:,1:26);
%  Y=D(:,27);
 
% % %D=heartdisease;
%  X1=D(:,1:13);
%  Y=D(:,14);

% D=BreastCancerCoimbra;
%  X1=D(:,1:9);
%  Y=D(:,10);

%D=transfusion;
% X1=D(:,1:4);
% Y=D(:,5);

% %D=SPECTFNew;
%  X1=D(:,1:44);
%  Y=D(:,45);

%D=shuttle1  ADD效果是最好的
% X1=D(:,1:9);
% Y=D(:,10);

%D=zoo %ADD仅次于COF是最好的
% X1=D(:,1:17);
% Y=D(:,18);

% D=HTRU_2 %不是最好的
% X1=D(:,1:8);
% Y=D(:,9);

% %D=Wilt
% X1=D(:,2:6);
% Y=D(:,1);

% %D=ann  %ADD在0.01时效果最好  Thyroid Disease  instance:3772  outliers:284
% X1=D(:,1:21);
% Y=D(:,22);

% %D=winequality_white2 %ADD0.2时仅次于COF，但是当0.1时最好
% X1=D(:,1:11);
% Y=D(:,12);

% D=risk_factors_cervical_cancer; %ADD 0.2时效果最好
%  X1=D(:,1:35);
%  Y=D(:,36);

%D=Breast cancer Wisconsin
%  X1=D(:,1:10);
%  Y=D(:,11);


% %D=Indian Liver Patient Dataset (ILPD)
%  X1=D(:,1:10);
%  Y=D(:,11);

%% 处理数据 %%%%%%%%%%%%%%%%%%%%%%
tic;
  [n,m]=size(X); %数据的个数n
%  [X]=Preprocessing(X1,n,m);
  dist=pdist2(X,X); 
 [Neicount,NN,RNN,NNN,nb,NaN]=NaNSearching(dist,n)
 
 %计算每个点的偏度值
 Ske_value=zeros(n,1);
 Ske_temp=zeros(n,m);
 for i = 1:m   %每个维度
     for j = 1:n  %每个点
         for z = 1:Neicount %遍历这个点的每个邻居的每个维度的值来计算偏度值
             Ske_temp(j,i)=Ske_temp(j,1)+power(X(NaN{j,1}(1,z),i)-X(j,i),2);
         end
     end
 end
 for i =1:n
     if Neicount==0
         Ske_value(i,1)=0;
     else
         Ske_value(i,1)=(sum(Ske_temp(i,:),2))/Neicount;
     end
 end
 
%计算每个维度中每个点和它的自然邻之间的平均值
average_value=zeros(n,m);
sum_value=zeros(n,m);
for i = 1:m%每个维度
    for j = 1:n%每个点
        for z = 1:Neicount%遍历每个点的邻居，计算它的邻居的每个维度的x值
             sum_value(j,i)=sum_value(j,i)+X(NaN{j,1}(1,z),i);
        end
        sum_value(j,i)=sum_value(j,i)+X(j,i);%加上它自己的x值
    end
end
for i =1:n%针对每一个点
    for j =1:m%每个维度计算一个平均值
        average_value(i,j)=sum_value(i,j)/Neicount;%average_value矩阵中存储的就是每个维度每个点和它的自然邻的x的平均值
    end
end

Den_value=zeros(n,1);
Den_temp=zeros(n,m);
for i = 1:m   %每个维度
    for j = 1:n  %每个点
        for z = 1:Neicount %遍历这个点的每个邻居的每个维度的值来计算局部密度值
            Den_temp(j,i)=Den_temp(j,1)+power(X(NaN{j,1}(1,z),i)-average_value(j,i),2);
        end
        if Neicount~=0
            Den_temp(j,i)=sqrt(Den_temp(j,i)/Neicount);
        else
           Den_temp(j,i)=0; 
        end
    end
end
for i =1:n
    if Den_temp(i,1)==0 && Den_temp(i,2)==0
        Den_value(i,1)=0;
    else
        Den_value(i,1)=1/(sum(Den_temp(i,:),2));%得到的就是每一个点的局部密度
    end
end

%求每个点的边界度值
Boundary_Degree=zeros(n,1);
for i = 1:n
    if Neicount==0
        Boundary_Degree(i,1)=0;
    else
        %Boundary_Degree(i,1)=Ske_value(i,1)*Den_value(i,1)*power(1/Neicount,2);
        Boundary_Degree(i,1)=Ske_value(i,1)/Den_value(i,1)*power(1/Neicount,2);
    end
end

%计算每个点的边界度改变率
Var_BD=zeros(n,1);
%将每个点和它的每个自然邻的边界度之差进行累加，明名为边界度改变率，边界点的边界度改变率比较大，核心点的边界度改变率很小，噪声点为0
Tempsum=zeros(n,1);
for i = 1:n
    if Neicount
        for j =1:Neicount
            Tempsum(i,1) = Tempsum(i,1) + abs((Boundary_Degree(i,1)-Boundary_Degree(NaN{i,1}(1,j),1)));
        end
    end
end
for i = 1:n
    if Neicount==0
        Var_BD(i,1)=0;
    else
        Var_BD(i,1)=abs(Tempsum(i,1)/Neicount);%abs()是取了一个绝对值
    end
end

x=1:1:n;%x轴上的数据，第一个值代表数据开始，第二个值代表间隔，第三个值代表终止
a=Var_BD';
plot(x,a,'-');hold on;%先x轴后y轴，第三个是点的形状
ylabel('LAMD value',...
    'FontName','Times New Roman','FontSize',14)
xlabel('n',...
    'FontName','Times New Roman','FontSize',14)
hold off;
%axis equal;%等比坐标轴
%axis off;%去掉坐标轴 

%设置阈值
alp=0.2*Neicount*sum(Var_BD)/n;%最终就用这个

%划分异常点和正常点
%如果边界度为0，那么将它识别为正常点，标签为1
label=zeros(n,1);
for i = 1:n
    if Var_BD(i,1)==0
        label(i,1)=0;
    else
        if Var_BD(i,1)>alp
            label(i,1)=1; 
        end
    end
end

%画出最后的数据集
scatter(X(label==0,1),X(label==0,2),40,'g','.');hold on 
scatter(X(label==1,1),X(label==1,2),40,'k','.');hold on
scatter(X(label==1,1),X(label==1,2),40,[0.5,0.5,0.5],'*');
toc;
outliers=find(label==1);%异常点的索引

%%compute AUC
% %SVM模型预测
% model = fitcsvm(Y,X,'-b 1');
% [predict_label, accuracy, scores] = svmpredict(Y,X,model);

%  XX=[X,Y];
% [N,M]=size(XX);
%逻辑回归
% Factor = glmfit(X,Y,'binomial', 'link', 'logit');
% Scores = glmval(Factor,X,'logit');
% Factor =glmfit(XX(:,1:M-1),XX(:,M),'binomial', 'link', 'logit'); %用逻辑回归来计算系数矩阵 
% Scores = glmval(Factor,XX(:,1:M-1), 'logit'); %用逻辑回归的结果预测测试集的结果

Scores = zeros(n,1);
Scores(outliers) = 1; % outlier scores
proclass = 1; 
[~,~,~,auc] = perfcurve(Y,Scores,proclass);

%计算F1值
f1= F1(Y,label);