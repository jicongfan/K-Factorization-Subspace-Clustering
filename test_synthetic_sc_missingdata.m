clc
warning off
clear all
for u=1:1
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
E=0.1*std(X(:))*randn(size(X));
M=ones(size(X));
ss=prod(size(X));
nne=round(ss*0.4);
M(randperm(ss,nne))=0;
X=M.*(X+E);
%% kss
disp('KSS...')
[L_kss,~]=ksubspaces_imp(X,k,r+1,'random');
L_kss = bestMap(Label(:),L_kss(:));
acc_kss_r=cluster_accuracy(Label(:),L_kss(:));
%% SSC
alpha=5;
[missrate,CMat] = SSC(X,0,0,alpha,1,1,Label);
acc_SSC=1-missrate;
%% LRMC
Xr=MC_IALM(X,M);
%% mc+kss
disp('KSS...')
[L_kss,~]=ksubspaces_imp(Xr,k,r+1,'random');
L_kss = bestMap(Label(:),L_kss(:));
acc_kss_r_mc=cluster_accuracy(Label(:),L_kss(:));
%% mc+SSC
alpha=5;
[missrate,CMat] = SSC(Xr,0,0,alpha,1,1,Label);
acc_SSC_mc=1-missrate;
%% kFSC
opt.solver=2;
opt.maxiter=500;
opt.tol=1e-3;
opt.init_type='random';
k=length(unique(Label));
lambda=0.1;
d=2*r;
[L_kFSC,OUT]=kFSC_M(X,M,d,k,lambda,opt);
L_kFSC = bestMap(Label(:),L_kFSC(:));
acc_kFSC_r=cluster_accuracy(Label,L_kFSC);
imagesc(abs(OUT.C))
%%
acc(u,:)=[acc_kss_r acc_kss_r_mc acc_SSC acc_SSC_mc acc_kFSC_r];
end
acc_mean=mean(acc)