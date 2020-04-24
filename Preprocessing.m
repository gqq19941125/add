function [nor_X]=Preprocessing(data,n,m)
%min-max 归一化处理方法：对于每一个点，找它每一列（多个属性值）的最大值Ximax和最小值Ximin,
%标准化之后的数据Xnorm=(Xi-Ximin)/(Ximax-Ximin)
%标准化之前若有缺省值，则用平均值代替

%缺省值处理
for i =1:n%遍历每一个点
    ind=cell(1,1);%记录每一个数据的哪几个属性列是有缺省列
    %mean_value=0;%记录除去缺省值之外的剩余属性值的值的平均值
    for j=1:m%遍历每一个属性
        if isnan(data(i,j))
            ind{1}=[ind{1},j]; 
        end
    end
    le=length(ind{1});%记录缺省值的个数
    if le%如果有缺省值
        iid=ind{1}(1,:);
        data(i,iid)=0;
        mean_value=sum(data(i,:))/(m-le);
        data(i,iid)=mean_value;
    end
end






% for i =1:n
%     mean_value=zeros(n,1);%记录除去缺省值之外的剩余属性值的值的平均值
%     for j=1:m
%         count=cell(n,1);%记录每一列的缺省值的索引
%         %ind=find(data(i,:)==NaN,2);
%         ind=find(isnan(data(i,j)));
%         if ind
%             count{j}=[count{j},ind];
%             le=length(count{j}(1,:));%共有几个缺省值
%             data(ind,j)=0;
%             mean_value(i,1)=sum(data(i,:))/(n-le);
%             data(i,j)=mean_value(i,1);  
%         else
%             continue;
%         end
%     end
% end

%最小最大归一化预处理
nor_X=zeros(n,m);
Ximax=zeros(1,m);
Ximin=zeros(1,m);
for i=1:m
    Ximax(1,i)=max(data(:,i));
    Ximin(1,i)=min(data(:,i)); 
end
for i =1:n 
    for j=1:m
        if Ximax(1,j)~=Ximin(1,j)%保证分母不为0
             nor_X(i,j)=(data(i,j)-Ximin(1,j))/(Ximax(1,j)-Ximin(1,j));
        else
            continue;
        end
    end
end


