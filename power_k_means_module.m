function [label,theta,wd]=power_k_means_module(X,p,n,k,s,theta,eta,tmax,tol)
W=zeros(n,k);
m=zeros(n,k);
label=zeros(1,n);
dd=zeros(1,k);
val1=theta;
coef=zeros(1,n);
norm_flg=1;
for t=1:tmax
    for i=1:n
        for l=1:k
            m(i,l)=sum((X(i,:)-theta(l,:)).^2);
        end
        coef(1,i)=(sum(m(i,:).^s))^((1/s)-1); %coef(1,i)=(sum(m(i,:).^s)+0.000001)^((1/s)-1);
    end
    
    for i=1:n
        for l=1:k
            W(i,l)=((m(i,l)+0.000001)^(s-1))*coef(1,i);
        end
    end
    
    for l=1:k
        for d=1:p
            theta(l,d)=(sum(W(:,l).*X(:,d)))/sum(W(:,l));
        end
    end
    
    if (mod(t,2))
        s=eta*s;
    end
    
    %If NA values exist, s is too small, then the power kmean degenerates to kmean (due to the limitations of computer accuracy).
    if (sum(isnan(theta))~=0)
        %disp(t);
        norm_flg=0;
        [label,wd,theta] = kmeans_for_power(X,k,val1);
        break
    end
    
    val2=theta;
    if (norm(val1-val2)<tol)&&(s<-20)
        for i=1:n
            for j=1:k
                dd(j)=sum((X(i,:)-theta(j,:)).^2);
            end
            [~,label(i)]=min(dd);
        end
        p_num=length(unique(label));
        if p_num==k
            %disp("Break by PKM")
            break
            %else
            %disp(p_num)
        end
    else
        val1=val2;
    end
end

%disp(t)

if norm_flg
    for i=1:n
        for j=1:k
            dd(j)=sum((X(i,:)-theta(j,:)).^2);
        end
        [~,label(i)]=min(dd);
    end
    [wd,theta] = label2wd(X,label,k);
end

% if isnan(obj)
%     pause
% end

end

function [clusters,error,mean_matrix] = kmeans_for_power(data,k,mean_matrix)
%=======================================
%           Initialising Data
%=======================================
rows = size(data, 1);
clusters = randi([1 k], rows, 1);
%clustered_data = [data clusters];
%=======================================
%           Computing Error
%=======================================
error = get_error(data, clusters, mean_matrix);
%fprintf('After initialization: error = %.4f \n', error);
pre_error=error;
p=1;
%=======================================
%           Starting Iterations
%=======================================
while 1
    for q = 1:rows
        %=======================================
        % Deciding which Cluster data belongs
        %=======================================
        dist = sum((data(q, :)-mean_matrix).^2,2);
        [~, clusters(q)] = min(dist);
    end
    %=======================================
    %          Calculating Mean
    %=======================================
    for i = 1:k
        mean_matrix(i, :) = mean(data(clusters == i, :),1);
    end
    %=======================================
    %           Computing Error
    %=======================================
    [error] = get_error(data, clusters, mean_matrix);
    if (pre_error-error)<1e-6
        break
    end
    pre_error=error;
    %fprintf('After iteration %d: error = %.4f \n', p, error);
    p=p+1;
end
end

function [error] = get_error(data, clusters, mean_matrix)
error = 0;
for j = 1: size(data, 1)
    error = error + sum((data(j, :)- mean_matrix(clusters(j), :)).^2);
end
end