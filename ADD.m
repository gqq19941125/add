function [f1,auc]=ADD(D)
 %% �˹����ݼ� %%%%%%%%%%%%%%%%%%%%%%%%%
 X=D(:,1:2);
 Y=D(:,3);
 
%% ��ʵ���ݼ� %%%%%%%%%%%%%%%%%%%%%%%%
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

%D=shuttle1  ADDЧ������õ�
% X1=D(:,1:9);
% Y=D(:,10);

%D=zoo %ADD������COF����õ�
% X1=D(:,1:17);
% Y=D(:,18);

% D=HTRU_2 %������õ�
% X1=D(:,1:8);
% Y=D(:,9);

% %D=Wilt
% X1=D(:,2:6);
% Y=D(:,1);

% %D=ann  %ADD��0.01ʱЧ�����  Thyroid Disease  instance:3772  outliers:284
% X1=D(:,1:21);
% Y=D(:,22);

% %D=winequality_white2 %ADD0.2ʱ������COF�����ǵ�0.1ʱ���
% X1=D(:,1:11);
% Y=D(:,12);

% D=risk_factors_cervical_cancer; %ADD 0.2ʱЧ�����
%  X1=D(:,1:35);
%  Y=D(:,36);

%D=Breast cancer Wisconsin
%  X1=D(:,1:10);
%  Y=D(:,11);


% %D=Indian Liver Patient Dataset (ILPD)
%  X1=D(:,1:10);
%  Y=D(:,11);

%% �������� %%%%%%%%%%%%%%%%%%%%%%
tic;
  [n,m]=size(X); %���ݵĸ���n
%  [X]=Preprocessing(X1,n,m);
  dist=pdist2(X,X); 
 [Neicount,NN,RNN,NNN,nb,NaN]=NaNSearching(dist,n)
 
 %����ÿ�����ƫ��ֵ
 Ske_value=zeros(n,1);
 Ske_temp=zeros(n,m);
 for i = 1:m   %ÿ��ά��
     for j = 1:n  %ÿ����
         for z = 1:Neicount %����������ÿ���ھӵ�ÿ��ά�ȵ�ֵ������ƫ��ֵ
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
 
%����ÿ��ά����ÿ�����������Ȼ��֮���ƽ��ֵ
average_value=zeros(n,m);
sum_value=zeros(n,m);
for i = 1:m%ÿ��ά��
    for j = 1:n%ÿ����
        for z = 1:Neicount%����ÿ������ھӣ����������ھӵ�ÿ��ά�ȵ�xֵ
             sum_value(j,i)=sum_value(j,i)+X(NaN{j,1}(1,z),i);
        end
        sum_value(j,i)=sum_value(j,i)+X(j,i);%�������Լ���xֵ
    end
end
for i =1:n%���ÿһ����
    for j =1:m%ÿ��ά�ȼ���һ��ƽ��ֵ
        average_value(i,j)=sum_value(i,j)/Neicount;%average_value�����д洢�ľ���ÿ��ά��ÿ�����������Ȼ�ڵ�x��ƽ��ֵ
    end
end

Den_value=zeros(n,1);
Den_temp=zeros(n,m);
for i = 1:m   %ÿ��ά��
    for j = 1:n  %ÿ����
        for z = 1:Neicount %����������ÿ���ھӵ�ÿ��ά�ȵ�ֵ������ֲ��ܶ�ֵ
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
        Den_value(i,1)=1/(sum(Den_temp(i,:),2));%�õ��ľ���ÿһ����ľֲ��ܶ�
    end
end

%��ÿ����ı߽��ֵ
Boundary_Degree=zeros(n,1);
for i = 1:n
    if Neicount==0
        Boundary_Degree(i,1)=0;
    else
        %Boundary_Degree(i,1)=Ske_value(i,1)*Den_value(i,1)*power(1/Neicount,2);
        Boundary_Degree(i,1)=Ske_value(i,1)/Den_value(i,1)*power(1/Neicount,2);
    end
end

%����ÿ����ı߽�ȸı���
Var_BD=zeros(n,1);
%��ÿ���������ÿ����Ȼ�ڵı߽��֮������ۼӣ�����Ϊ�߽�ȸı��ʣ��߽��ı߽�ȸı��ʱȽϴ󣬺��ĵ�ı߽�ȸı��ʺ�С��������Ϊ0
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
        Var_BD(i,1)=abs(Tempsum(i,1)/Neicount);%abs()��ȡ��һ������ֵ
    end
end

x=1:1:n;%x���ϵ����ݣ���һ��ֵ�������ݿ�ʼ���ڶ���ֵ��������������ֵ������ֹ
a=Var_BD';
plot(x,a,'-');hold on;%��x���y�ᣬ�������ǵ����״
ylabel('LAMD value',...
    'FontName','Times New Roman','FontSize',14)
xlabel('n',...
    'FontName','Times New Roman','FontSize',14)
hold off;
%axis equal;%�ȱ�������
%axis off;%ȥ�������� 

%������ֵ
alp=0.2*Neicount*sum(Var_BD)/n;%���վ������

%�����쳣���������
%����߽��Ϊ0����ô����ʶ��Ϊ�����㣬��ǩΪ1
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

%�����������ݼ�
scatter(X(label==0,1),X(label==0,2),40,'g','.');hold on 
scatter(X(label==1,1),X(label==1,2),40,'k','.');hold on
scatter(X(label==1,1),X(label==1,2),40,[0.5,0.5,0.5],'*');
toc;
outliers=find(label==1);%�쳣�������

%%compute AUC
% %SVMģ��Ԥ��
% model = fitcsvm(Y,X,'-b 1');
% [predict_label, accuracy, scores] = svmpredict(Y,X,model);

%  XX=[X,Y];
% [N,M]=size(XX);
%�߼��ع�
% Factor = glmfit(X,Y,'binomial', 'link', 'logit');
% Scores = glmval(Factor,X,'logit');
% Factor =glmfit(XX(:,1:M-1),XX(:,M),'binomial', 'link', 'logit'); %���߼��ع�������ϵ������ 
% Scores = glmval(Factor,XX(:,1:M-1), 'logit'); %���߼��ع�Ľ��Ԥ����Լ��Ľ��

Scores = zeros(n,1);
Scores(outliers) = 1; % outlier scores
proclass = 1; 
[~,~,~,auc] = perfcurve(Y,Scores,proclass);

%����F1ֵ
f1= F1(Y,label);