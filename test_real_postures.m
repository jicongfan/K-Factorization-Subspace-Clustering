clc
clear all
warning off
%
load('Postures_m.mat');
X(isnan(X))=0;
[Label,id]=sort(Label);
X=X(:,id);
X=normalizeL2(X);
k=length(unique(Label));
%% kFSC
for i=1:1
tic
opt.solver=2;
opt.maxiter=200;opt.nrep_kmeans=10;
opt.tol=1e-4;
opt.init_type='k-means';
opt.classifier='re';
k=length(unique(Label));
lambda=0.2;
d=30;
[L_kFSC,out]=kFSC(X,d,k,lambda,opt);
L_kFSC = bestMap(Label(:),L_kFSC(:));
acc_kFSC(i)=cluster_accuracy(Label,L_kFSC);
nmi_kFSC(i)=MutualInfo(Label,L_kFSC);
imagesc(abs(out.C))
toc
end
%% k-FSC-L
for i=1:1
tic
opt.solver=2;
opt.maxiter=300;
opt.tol=1e-4;
opt.init_type='k-means';
opt.nrep_kmeans=100;
opt.classifier='re';
k=length(unique(Label));
lambda=0.2;
d=30;
[L_kFSC,OUT]=kFSC_LARGE(X,d,k,lambda,opt,500,'k-means');
L_kFSC = bestMap(Label(:),L_kFSC(:));
acc_kFSC_L(i)=cluster_accuracy(Label,L_kFSC);
nmi_kFSC_L(i)=MutualInfo(Label,L_kFSC);
imagesc(abs(OUT.C))
toc
end
%% minibatch
for i=1:1
tic
opt.solver=2;
opt.maxiter=200;opt.nrep_kmeans=10;
opt.np=10;
opt.bs=1000;
opt.tol=1e-4;
opt.init_type='k-means';
k=length(unique(Label));
lambda=0.2;
d=30;
[L_kFSC,out]=kFSC_minibatch(X,d,k,lambda,opt);
L_kFSC = bestMap(Label(:),L_kFSC(:));
acc_kFSC_MB(i)=cluster_accuracy(Label,L_kFSC);
nmi_kFSC_MB(i)=MutualInfo(Label,L_kFSC);
imagesc(abs(out.C))
toc
end