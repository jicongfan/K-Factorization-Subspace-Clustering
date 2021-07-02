function [L,output]=kFSC_M(X,M,d,k,lambda,options)
% Written by Jicong Fan. fanjicong@cuhk.edu.cn
% Large-Scale Subspace Clustering via k-Factorization. KDD 2021.
if isfield(options,'maxiter') maxiter=options.maxiter;else maxiter=500;end
if isfield(options,'iter_D') iter_D=options.iter_D;else iter_D=5;end
if isfield(options,'tol') tol=options.tol;else tol=1e-4;end
if isfield(options,'solver') solver=options.solver;else solver=2;end
if isfield(options,'init_type') init_type=options.init_type;else init_type='k-means';end
if isfield(options,'nrep_kmeans') nrep_kmeans=options.nrep_kmeans;else nrep_kmeans=100;end
if isfield(options,'obj_all') obj_all=options.obj_all;else obj_all=0;end
if isfield(options,'classifier') classifier=options.classifier;else classifier='abs';end
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
[D,C]=initial_DC(X,m,d,k,init_type,nrep_kmeans);
LM=zeros(k,n);
for j=1:k
    temp=D(:,(j-1)*d+1:j*d)'*X;
    LM(j,:)=sum(temp.^2).^0.5;
end
LM=sort(LM,'descend');
% lam_max=min(LM(1,:));
% lam_min=max(LM(2,:));
% disp(['The estimated lambda is ' num2str((lam_min+lam_max)/2)])
dfC=zeros(size(C));
Q=[];
i=0;
while i<=maxiter
    i=i+1;
    if length(find(sum(abs(C))==0))>=n*0.1
        lambda=lambda/2;
        disp(['Too large lambda! Restart with lambda=' num2str(lambda) ' ......'])
        [D,C]=initial_DC(X,m,d,k,init_type,nrep_kmeans);
        i=1;
        dfC=zeros(size(C));
        Q=[];
    end
    [C_new,Q]=update_C_GaussSeidel_extrop(X,M,D,C,d,k,lambda,dfC,Q,i);
    dfC=C-C_new;
    D_new=update_D(X,M,D,C_new,iter_D);
    %%%%
    dC=norm(C_new-C,'fro')/norm(C,'fro');
    dD=norm(D_new-D,'fro')/norm(D,'fro');
    isstop=(max(dC,dD)<tol);
    if i<=3||mod(i,50)==0||isstop||obj_all
        loss(i)=0.5*norm(M.*(X-D_new*C_new),'fro')^2+lambda*sum(sum(reshape(C_new,d,k*n).^2).^0.5);
        disp(['iteration ' num2str(i) ' reldiff_D=' num2str(dD) ' reldiff_C=' num2str(dC)...
           ' objective=' num2str(loss(i))])
    end
    if isstop
        disp('Converged!')
        break
    end
    if i==maxiter
        disp('Max iteration reached!')
    end
    C=C_new;
    D=D_new;
end
% X=M.*X+(1-M).*(D*C);
X=D*C;
switch classifier
    case 'abs'
        disp('Asign clusters by C')
        for i=1:k
%             Y(i,:)=sum(abs(C((i-1)*d+1:i*d,:)));
            Y(i,:)=sum((C((i-1)*d+1:i*d,:)).^2);
        end
        [~,L]=max(Y);
    case 're'
        disp('Asign clusters by reconstruction error')
        for i=1:k
            Di=D(:,(i-1)*d+1:i*d);
            C_t=inv(Di'*Di+1e-5*eye(d))*Di'*X;
            E(i,:)=sum(((X-Di*C_t)).^2);
        end
        [~,L]=min(E);
end
output.D=D;
output.C=C;
output.loss=loss;
end
%%
function [D,C]=initial_DC(X,m,d,k,init_type,nrep_km)
X0=X;
X=X(:,randperm(size(X,2),min(size(X,2),50000)));

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
C=inv(D'*D+eye(d*k)*1e-5)*D'*X0;
end
%%
function [C_new,Q]=update_C_GaussSeidel_extrop(X,M,D,C,d,k,lambda,dfC,Q,i)
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
    gC=-D(:,(j-1)*d+1:j*d)'*(M.*(X-Xh));
    tau=1.0*normest(D(:,(j-1)*d+1:j*d))^2;
    Q(i,j)=tau;
    temp=C_h((j-1)*d+1:j*d,:)-gC/tau;
    C_new((j-1)*d+1:j*d,:)=solve_L21(temp,lambda/tau);
    Xh=Xh+D(:,(j-1)*d+1:j*d)*(C_new((j-1)*d+1:j*d,:)-C_h((j-1)*d+1:j*d,:));
end
end
%% |M.*(X-DC)|
function D_new=update_D(X,M,D,C_new,iter_D)
m=size(D,1);
D_t=D;
tau=1.0*normest(C_new)^2;
for j=1:iter_D
    gD=M.*(X-D_t*C_new)*(-C_new');
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