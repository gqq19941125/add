function [f1]= F1(test_targets,output)
len=length(output);
% sens = zeros(n,1);
% spec = zeros(n,1);
% F1 = zeros(n,1);
% pre = zeros(n,1);
% rec = zeros(n,1);
tp = 0;%真正例
fn = 0;%假反例
fp = 0;%假正例
tn = 0;%真反例
for i = 1:len
        if output(i,1)==1 && test_targets(i,1)==1
            tp=tp+1;
        elseif output(i,1)==1 && test_targets(i,1)==0
            fp=fp+1;
        elseif output(i,1)==0 && test_targets(i,1)==1
            fn=fn+1;
        elseif output(i,1)==0 && test_targets(i,1)==0
            tn=tn+1;
        end
end
% sens = tp/(tp+fn);
% spec = tn/(tn+fp);
pre = tp/(tp+fp);
rec = tp/(tp+fn);
f1 = (2*pre*rec)/(pre+rec);
