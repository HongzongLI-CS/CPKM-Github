function [glabel,gbest]=CAPKM(X,p,M,N,diversity_threshold)
if nargin < 4
    % default parameters
    M=30;
    N=10;
    diversity_threshold=0.0005;
end

n=size(X,1);
m=size(X,2);
w=1;beta1=1;beta2=1;
tmax=1000;tol=0.001;

% sam=randperm(n,2*p);
% theta2=X(sam,:);
% theta2=theta2+normrnd(0,0.1,2*p,m);

v=2*rand(N*p,m)-2;
theta2=zeros(N*p,m);
for i=1:N
    theta2((i-1)*p+1:i*p,:)=initialization_kpp(X, p);
end
gbest=1e10;
pbestx=zeros(N*p,m);
pbest=1e10*ones(1,N);
plabel=zeros(n,N);
count_gli=0;
%for mutation
gbestx=zeros(N*p,m);
it=0;
Div_list=[];

while true
    for nn_count=1:N
        theta=theta2((nn_count-1)*p+1:nn_count*p,:);
        
        s=(-0.2+30).*rand -30;
        eta=(1.4-1.1).*rand +1.1;
        [label,theta,obj]=power_k_means_module(X,m,n,p,s,theta,eta,tmax,tol);
        
        %for mutation
        theta_new=zeros(p,m);
        D=pdist2(gbestx(1:p,:),theta);
        selected_list=[];
        for i=1:p
            [~,index]=sort(D(:,i));
            for k=1:p
                if ~ismember(index(k),selected_list)
                    selected_list=[selected_list,index(k)];
                    break
                end
            end
            theta_new(index(k),:)=theta(i,:);
        end
        theta_r((nn_count-1)*p+1:nn_count*p,:)=theta_new;
        
        if obj<pbest(1,nn_count)
            pbest(1,nn_count)=obj;
            pbestx((nn_count-1)*p+1:nn_count*p,:)=theta_new;
            plabel(:,nn_count)=label;
        end
    end
    [~,pc]=min(pbest);
    if pbest(1,pc)<gbest
        gbest=pbest(1,pc);
        gbestx=repmat(pbestx((pc-1)*p+1:pc*p,:),N,1);
        glabel=plabel(:,pc);
        count_gli=0;
    else
        count_gli=count_gli+1;
    end
    
    if count_gli>M
        break
    end
    
    % Compute the diversity and perform mutation if necessary
    Div=norm(theta_r-gbestx,'fro');
    Div=Div/(N*m*p);
    Div_list=[Div_list;Div];
    it=it+1;
    if Div < diversity_threshold
        %% Dispersed particles
        a=exp(it/1000);%1-e
        %a=exp(10*(it/10000));
        %a=exp(10*(it/params.itMax));
        psi=-2.5*a+5*a*rand(N*p,m);%-2.5a-2.5a
        et=(1/sqrt(a))*exp((-(psi/a).^2)/2).*cos((5*(psi/a)));
        larger_zero=et>0;
        lower_zero=et<0;
        %eta=(1/sqrt(a))*exp(((-psi/a)^2)/2)*cos((5*(psi/a)));
        theta2=theta2+larger_zero.*et.*(1-theta2);
        theta2=theta2+lower_zero.*et.*theta2;
        v=rand(N*p,m)-1;
        pbestx=zeros(N*p,m);
        pbest=1e10*ones(1,N);
    else
        v=w*v+beta1*rand(N*p,m).*(pbestx-theta2)+beta2*rand(N*p,m).*(gbestx-theta2);
        theta2=theta2+v;
    end
end
end

function theta=initialization_lsi(X,p,n,m)
theta=zeros(p,m);
r=zeros(1,p);
d=squareform(pdist(X));
[r1,r2]=find(d==max(max(d)));
theta(1,:)=X(r1(1),:);
theta(2,:)=X(r2(1),:);
r(1,1)=r1(1);
r(1,2)=r2(1);
if p>2
    for k=3:p
        max_value=0;
        for i=1:n
            value=0;
            for l=1:k-1
                value=value+d(i,r(1,l));
            end
            if value>max_value
                max_value=value;
                r(1,k)=i;
            end
        end
        theta(k,:)=X(r(1,k),:);
    end
end
end

function centroid=initialization_kpp(data, k)
% Choose the first inital centroid randomly
centroid = data(randperm(size(data,1),1)',:);

% Select remaining initial centroids (a total number of k-1)
for i = 2:k
    distance_matrix = zeros(size(data,1),i-1);
    for j = 1:size(distance_matrix,1)
        for p = 1:size(distance_matrix,2)
            distance_matrix(j,p) = sum((data(j,:)-centroid(p,:)) .^ 2);
        end
    end
    % Choose next centroid according to distances between points and
    % previous cluster centroids.
    index = Roulettemethod(distance_matrix);
    centroid(i,:) = data(index,:);
end
end

function [index] = Roulettemethod(distance_matrix)

% Find shortest distance between one sample and its closest cluster centroid
[min_distance,~] = min(distance_matrix,[],2);

% Normalize for further operations
min_distance = min_distance ./ sum(min_distance);

% Construct roulette according to min_distance
temp_roulette = zeros(size(distance_matrix,1),1);
for i = 1:size(distance_matrix,1)
    temp_roulette(i,1) = sum(min_distance(1:i,:));
end

% Generate a random number for selection
temp_rand = rand();

% Find the corresponding index
for i = 1:size(temp_roulette,1)
    if((i == 1) && temp_roulette(i,1) > temp_rand)
        index = 1;
    elseif((temp_roulette(i,1) > temp_rand) && (temp_roulette(i-1,1) < temp_rand))
        index = i;
    end
end
end