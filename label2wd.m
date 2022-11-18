function [wd,theta] = label2wd(data,label,p)
m=size(data,2);
theta=zeros(p,m);
for l = 1:p
    theta(l, :) = mean(data(label == l, :));
end

n=size(data,1);
wd=0;
for i=1:n
    wd=wd+sum((data(i,:)-theta(label(i),:)).^2);
end
