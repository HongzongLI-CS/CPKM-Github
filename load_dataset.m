function [X,label,p,diversity_threshold] = load_dataset(id)
%% load data
if id==1
    data=load('satimage.dat');
    p=7;
    X=data(:,1:end-1);
    label=data(:,end)';
    diversity_threshold=0.000885;
elseif id==2
    data=load('coil2000.dat');
    p=2;
    X=data(:,1:end-1);
    label=data(:,end)';
    diversity_threshold=0.000017;
end

X=(X-min(X))./(max(X)-min(X));

end

