function [L,output]=kFSC_minibatch(X,d,k,lambda,options)
% Written by Jicong Fan. fanjicong@cuhk.edu.cn
% Large-Scale Subspace Clustering via k-Factorization. KDD 2021.
if isfield(options,'maxiter') maxiter=options.maxiter;else maxiter=500;end
if isfield(options,'iter_D') iter_D=options.iter_D;else iter_D=5;end
if isfield(options,'tol') tol=options.tol;else tol=1e-4;end
if isfield(options,'solver') solver=options.solver;else solver=2;end
if isfield(options,'init_type') init_type=options.init_type;else init_type='k-means';end
if isfield(options,'nrep_kmeans') nrep_kmeans=options.nrep_kmeans;else nrep_kmeans=10;end
if isfield(options,'obj_all') obj_all=options.obj_all;else obj_all=0;end
if isfield(options,'classifier') classifier=options.classifier;else classifier='abs';end
if isfield(options,'bs') bs=options.bs;else bs=1000;end
if isfield(options,'np') np=options.np;else np=1;end

switch solver
    case 0
        disp('Solve k-FSC by Jacobi optimization ...')
    case 1
        disp('Solve k-FSC by Gauss-Seidel optimization ...')
    case 2
        disp('Solve k-FSC by accelerated Gauss-Seidel optimization ...')
end
[m,n]=size(X);
X=X./repmat(sum(X.^2).^0.5,m,1);
[D,C]=initial_DC(X(:,randperm(n,min(n,50000))),m,d,k,init_type,nrep_kmeans);
LM=zeros(k,n);
for j=1:k
    temp=D(:,(j-1)*d+1:j*d)'*X;
    LM(j,:)=sum(temp.^2).^0.5;
end
LM=sort(LM,'descend');
% lam_max=min(LM(1,:));
% lam_min=max(LM(2,:));
% disp(['The estimated lambda is ' num2str((lam_min+lam_max)/2)])

C=inv(D'*D+eye(d*k)*1e-5)*D'*X;
dfC=zeros(size(C));
Q=[];
i=0;
for p=1:np
    i=1;
    idx=randperm(n,n);
    Xr=X(:,idx);
    nb=0;
    while i<=n
        ib=i:min(i+bs-1,n);
        Xb=Xr(:,ib);
        [D,Cr(:,ib),dD,dC]=compute_DC(D,C(:,idx(ib)),Xb,d,k,lambda,1,5);
        i=i+bs;
        nb=nb+1;
        if mod(nb,10)==0
           disp(['pass ' num2str(p) ' batch ' num2str(nb)...
               ' sample ' num2str(i)...
               ' reldiff_D=' num2str(dD) ' reldiff_C=' num2str(dC)])
        end
    end
    C(:,idx)=Cr;
end
for i=1:k
    Di=D(:,(i-1)*d+1:i*d);
    C_t=inv(Di'*Di+1e-5*eye(d))*Di'*X;
    E(i,:)=sum((X-Di*C_t).^2);
end
[~,L]=min(E);
output.D=D;
output.C=C;
output.loss=[];
end
%%
function [D_new,C_new,dD,dC]=compute_DC(D,C,X,d,k,lambda,iter_D,iter_C)
Ct=C;
dfC=zeros(size(C));
Q=[];
for i=1:iter_C   
%     C_new=update_C_GaussSeidel(X,D,Ct,d,k,lambda); 
    [C_new,Q]=update_C_GaussSeidel_extrop(X,D,Ct,d,k,lambda,dfC,Q,i);
    dfC=Ct-C_new;
    Ct=C_new;
end
D_new=update_D(X,D,C_new,iter_D);
dC=norm(dfC,'fro')/norm(C,'fro');
dD=norm(D_new-D,'fro')/norm(D,'fro');
end
%%
function [D,C]=initial_DC(X,m,d,k,init_type,nrep_km)
switch init_type
    case 'random'
        D=randn(m,d*k);
    case 'k-means'
        disp(['Initializing D by k-means algorithm (' num2str(nrep_km) ' replicates)...'])
        [id,C,~,dist]=kmeans(X',k,'Distance','cosine','Replicates',nrep_km);
        [~,idx]=sort(dist,'ascend');
        for i=1:k
            temp=X(:,idx(1:d,i));
            if m<d
            D(:,(i-1)*d+1:i*d)=[temp];
            else
            [U,~,~]=svd(temp,'econ');
            D(:,(i-1)*d+1:i*d)=U(:,1:d);
            end
        end
end
D=D./repmat(sum(D.^2).^0.5,m,1);
C=inv(D'*D+eye(d*k)*1e-5)*D'*X;
end
%%
function C_new=update_C_Jacobi(X,D,C,d,k,lambda)
gC=(-D')*(X-D*C);
C_new=C;
tau=1.0*normest(D)^2;
for j=1:k
%     tau=1.01*normest(D(:,(j-1)*d+1:j*d))^2;
    temp=C((j-1)*d+1:j*d,:)-gC((j-1)*d+1:j*d,:)/tau;
    C_new((j-1)*d+1:j*d,:)=solve_L21(temp,lambda/tau);
end
end
%%
function C_new=update_C_GaussSeidel(X,D,C,d,k,lambda)
%%%%% gC=(-D')*(X-D*C);
C_new=C;
Xh=D*C_new;
for j=1:k
    gC=-D(:,(j-1)*d+1:j*d)'*(X-Xh);
    tau=1.0*normest(D(:,(j-1)*d+1:j*d))^2;
    temp=C((j-1)*d+1:j*d,:)-gC/tau;
    C_new((j-1)*d+1:j*d,:)=solve_L21(temp,lambda/tau);
    Xh=Xh+D(:,(j-1)*d+1:j*d)*(C_new((j-1)*d+1:j*d,:)-C((j-1)*d+1:j*d,:));
end
end
%%
function [C_new,Q]=update_C_GaussSeidel_extrop(X,D,C,d,k,lambda,dfC,Q,i)
%%%%% gC=(-D')*(X-D*C);
for j=1:k
    if i>2
        eta=sqrt(Q(i-2,j)/Q(i-1,j))*0.95;
    else
        eta=0;
    end
    C_h((j-1)*d+1:j*d,:)=C((j-1)*d+1:j*d,:)-dfC((j-1)*d+1:j*d,:)*eta;
end
C_new=C_h;
Xh=D*C_new;
for j=1:k
    gC=-D(:,(j-1)*d+1:j*d)'*(X-Xh);
    tau=1.0*normest(D(:,(j-1)*d+1:j*d))^2;
    Q(i,j)=tau;
    temp=C_h((j-1)*d+1:j*d,:)-gC/tau;
    C_new((j-1)*d+1:j*d,:)=solve_L21(temp,lambda/tau);
%     l2=sum(C_new((j-1)*d+1:j*d,:).^2).^0.5;
%     if sum(l2~=0)<1000
%         l2t=sum(C_new((j-1)*d+1:j*d,:).^2).^0.5;
%         [~,idt]=sort(l2t,'ascend');
%         idt=idt(1:1000);
%         temp1=C_new((j-1)*d+1:j*d,:);
%         temp1(:,idt)=temp(:,idt);
%         C_new((j-1)*d+1:j*d,:)=temp1;
%     end   
    Xh=Xh+D(:,(j-1)*d+1:j*d)*(C_new((j-1)*d+1:j*d,:)-C_h((j-1)*d+1:j*d,:));
end
end
%%
function D_new=update_D(X,D,C_new,iter_D)
m=size(D,1);
D_t=D;
XC=X*C_new';
CC=C_new*C_new';
tau=1.0*normest(CC);
for j=1:iter_D
    gD=-XC+D_t*CC;
    D_t=D_t-gD/tau;
    ld=sum(D_t.^2).^0.5;
    idx=find(ld>1);
    D_t(:,idx)=D_t(:,idx)./repmat(ld(idx),m,1);
%     D_t=D_t./repmat(sum(D_t.^2).^0.5,m,1);
    if norm(gD/tau,'fro')/norm(D_t,'fro')<1e-3
        break
    end  
end
D_new=D_t;
end
%%
function X=solve_L21(X,thr)
L=(sum(X.^2)).^0.5;
Lc=max(0,L-thr)./L;
X=X.*repmat(Lc,size(X,1),1);
X(:,find(L==0))=0;
end