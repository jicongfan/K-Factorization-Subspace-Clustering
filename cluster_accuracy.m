function acc=cluster_accuracy(L,Lr)
acc=sum(L(:)-Lr(:)==0)/length(L);
end