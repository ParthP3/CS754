function [out] = OMP(A, A1, y, k, m)
out = zeros(1024, 1);

thet_i = 0;
supp = zeros(1, 1024);
r = y;
for it = 1:m
    if norm(r, "fro")<= 1e-10
        break;
    end
    [J,j] = max(abs(r'*A1));
    supp(j) = 1;
    sub_rad = A(:, (supp~=0));
    thet_i = pinv(sub_rad)*y;
    r = y - sub_rad*thet_i;
end
count1 = 1;
for count = 1:1024
    if supp(count) == 1
        out(count) = thet_i(count1);
        count1 = count1+1;
    end
end
end