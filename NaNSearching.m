function[Neicount,NN,RNN,NNN,nb,NaN]=NaNSearching(dist,n)
%D=varargin{1};
r=1;
nb=zeros(n,1);
C=cell(n,1);
NN=cell(n,1);%初始化每个点的KNN邻居
RNN=cell(n,1);%初始化每个点的RKNN邻居
%NNN=cell(n,1);%是NN和RNN的交集，也就每个点的自然邻个数
NaN=cell(n,1);
%A=pdist2(D,D);
Numb1=0;
Numb2=0;
for ii=1:n
   [sa,index]=sort(dist(:,ii));
   C{ii}=[sa,index];
end
while(r<n)
    for kk=1:n
        x=kk;
        y=C{x}(r+1,2);%找到对应的索引
        nb(y)= nb(y)+1;
        NN{x}=[NN{x},y];
        RNN{y}=[RNN{y},x];
    end
    Numb1=sum(nb==0);
    if Numb2~=Numb1
        Numb2=Numb1;
    else
       break; 
    end
    r=r+1;
end
for jj=1:n
    NNN{jj}=intersect(NN{jj},RNN{jj});
end
Neicount=r;
for jj =1:n
    temp=C{jj}(2:r+1,2);
    NaN{jj}=temp';
end
end