function [IDX,SS]=ksubspaces_imp(X,k,d,init_type)
switch init_type
    case 'random'
        [IDX, SS, inpnorm] = ksubspaces(X',k,d,[],200,1e-4);
    case 'k-means'
        SS=km(X,k,d);  
        disp('Running k-PC ...')
        [IDX, SS, inpnorm] = ksubspaces(X',k,d,SS,200,1e-4);
end
end
%%
function D=km(X,k,d)
disp('Initializing D by k-means algorithm ...')
[id,C,~,dist]=kmeans(X',k,'Distance','cosine','Replicates',10);
[~,idx]=sort(dist,'ascend');
m=size(X,1);
for i=1:k
    Z=X(:,idx(1:d*5,i));
    [U,~,~]=svd(Z,'econ');
    D(:,:,k)=U(:,1:d)';
end
end