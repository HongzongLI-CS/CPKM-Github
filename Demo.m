%% load datasets
id=1; %1 for dataset SATIMAGE, 2 for dataset COIL2000
[X,label,p,diversity_threshold] = load_dataset(id);

%% setting hyper-parameters
M=30;
N=5; 

%% launch the algorithm
[resulting_label,obj_value]=CAPKM(X,p,M,N,diversity_threshold);
%% It is also optional to use default hyper-parameters
%[resulting_label,obj_value]=CAPKM(X,p);


%% show the value of objective function
disp(obj_value)

%% plot results
colors=hsv(p);
for i=1 %using feature 1
    for j=2 %using feature 2
        figure()
        for l=1:n
            hold on
            plot(X(l,i),X(l,j),'.','color', colors(resulting_label(l),:),'MarkerSize',20);
        end
    end
end