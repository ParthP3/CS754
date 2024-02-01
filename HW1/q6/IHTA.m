function [out] = IHTA(A, A1, y, k, m)
out = zeros(1024, 1);
x=zeros(1024, 1);
it = 0;
while norm((y-A1*x))>=1e-9 && it<=1000

    t = (x + A1'*(y-A1*x));
    [val, ind] = sort(abs(t), "descend");
    ind1 = ind(1:k, 1);
    x1 = zeros(1024, 1);
    x1(ind1, 1) = t(ind1, 1);
    x = x1;
    it = it+1;
end
out = x;
end