X = randn(1024,128);

tic;
[IDX SS] = ksubspaces(X,16,2);
toc
Y = projnnsubspaces(SS,X,IDX);

tic;
[aIDX aSS] = seqksubspaces(X,2,0.25);
toc
aY = projnnsubspaces(aSS,X,aIDX);

