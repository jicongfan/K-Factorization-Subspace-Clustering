clc
warning off
clear all
for u=1:10
k=5;
m=25;
r=5;
n=50;
X=[];
Label=[];
A=randn(m,r)*1;
for i=1:k
X=[X (A+randn(m,r))*randn(r,n)];
Label((i-1)*n+1:i*n)=i;
end
E1=0.1*std(X(:))*randn(size(X));
E2=zeros(size(X));
ss=prod(size(X));
nne=round(ss*0.4);
E2(randperm(ss,nne))=randn(1,nne)*std(X(:));
X=X+E1+E2;
%% kss
disp('KSS...')
[L_kss,~]=ksubspaces_imp(X,k,r+1,'random');
L_kss = bestMap(Label(:),L_kss(:));
acc_kss_r=cluster_accuracy(Label(:),L_kss(:));
%% SSC
alpha=5;
[missrate,CMat] = SSC(X,0,0,alpha,1,1,Label);
acc_SSC=1-missrate;
%% kFSC
opt.solver=2;
opt.maxiter=500;
opt.tol=1e-3;
opt.init_type='random';
k=length(unique(Label));
lambda=0.2;
lambda_E=0.07;
d=2*r;
[L_kFSC,OUT]=kFSC_E(X,d,k,lambda,lambda_E,opt);
L_kFSC = bestMap(Label(:),L_kFSC(:));
acc_kFSC_r=cluster_accuracy(Label,L_kFSC);
imagesc(abs(OUT.C))
%%
acc(u,:)=[acc_kss_r acc_SSC acc_kFSC_r];
end
acc_mean=mean(acc)