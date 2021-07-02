clc
clear all
load('fmnist_fea_150.mat');
X=normalizeL2(X);
k=length(unique(Label));
%% kFSC
for i=1:1
tic
opt.solver=2;
opt.maxiter=200;
opt.tol=1e-4;
opt.init_type='k-means';
opt.classifier='re';
opt.nrep_kmeans=10;
k=length(unique(Label));
lambda=0.5;
d=30;
[L_kFSC,OUT]=kFSC(X,d,k,lambda,opt);
L_kFSC = bestMap(Label(:),L_kFSC(:));
acc_kFSC(i)=cluster_accuracy(Label,L_kFSC);
nmi_kFSC(i)=MutualInfo(Label,L_kFSC);
imagesc(abs(OUT.C))
toc
end
%% kFSC-LARGE
for i=1:1
tic
opt.solver=2;
opt.maxiter=300;
opt.tol=1e-4;
opt.init_type='k-means';
opt.nrep_kmeans=100;
opt.classifier='re';
k=length(unique(Label));
lambda=0.5;
d=30;
[L_kFSC,OUT]=kFSC_LARGE(X,d,k,lambda,opt,500,'k-means');
L_kFSC = bestMap(Label(:),L_kFSC(:));
acc_kFSC_L(i)=cluster_accuracy(Label,L_kFSC);
nmi_kFSC_L(i)=MutualInfo(Label,L_kFSC);
imagesc(abs(OUT.C))
toc
end
%% kFSC-MB
for i=1:1
tic
opt.solver=2;
opt.maxiter=200;opt.nrep_kmeans=10;opt.np=5;opt.bs=1000;
opt.tol=1e-4;
opt.init_type='k-means';
opt.classifier='re';
k=length(unique(Label));
lambda=0.5;
d=30;
[L_kFSC,out]=kFSC_minibatch(X,d,k,lambda,opt);
L_kFSC = bestMap(Label(:),L_kFSC(:));
acc_kFSC_MB(i)=cluster_accuracy(Label,L_kFSC);
nmi_kFSC_MB(i)=MutualInfo(Label,L_kFSC);
imagesc(abs(out.C))
TC(i)=toc;
end