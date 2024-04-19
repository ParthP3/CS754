function out = SVT(M, sigma, tau,n,m)
Y = double(zeros(n,m));
th1 = zeros(n,m);
count = 0;
while 1
    
    [U,S,V] = svd(Y);
    rankY = nnz(S);
    for i = 1:rankY
        S(i,i) = max(0,S(i,i)-tau);
    end
    th1 = U*S*V';
    Y1 = Y + M - sigma.*th1;
    if (norm(Y1-Y,"fro")<0.05)
        break;
    end
    if count>500
        break;
    end
    norm(Y1-Y,"fro");
    Y = Y1;
    count = count+1;
end

out = th1;
end