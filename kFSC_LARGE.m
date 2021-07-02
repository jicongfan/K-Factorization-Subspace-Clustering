function [Label,output]=kFSC_LARGE(X,d,k,lambda,opt,n_sel,sel_type)
% Written by Jicong Fan. fanjicong@cuhk.edu.cn
% Large-Scale Subspace Clustering via k-Factorization. KDD 2021.
n=size(X,2);
switch sel_type
    case 'random'
        disp(['Select landmark data points randomly...'])
        ids=sort(randperm(n,n_sel*k),'ascend');
        Xs=X;
        X_train=Xs(:,ids);
    case 'k-means'
        disp(['Select landmark data points by k-means...'])
        ids=sort(randperm(n,min(n,n_sel*k*5)),'ascend');
        Xs=X(:,ids);
        [id,C,~,dist]=kmeans(Xs',n_sel*k,'Distance','cosine','Replicates',1);
        [~,idx]=sort(dist,'ascend');
        ids=idx(1,:);
        X_train=C';
end
clear Xs C dist
disp('Perform kFSC on the selected landmark data points...')
[L_kFSC,output]=kFSC(X_train,d,k,lambda,opt);
disp('Predict the labels of all data points...')
% ||X-D_iC||+lambda||C|| (-D')X+D'DC+lambdaC
D=output.D;
for i=1:k
    Di=D(:,(i-1)*d+1:i*d);
    C=inv(Di'*Di+1e-5*eye(d))*Di'*X;
    E(i,:)=sum((X-Di*C).^2);
end
[~,Label]=min(E);
end