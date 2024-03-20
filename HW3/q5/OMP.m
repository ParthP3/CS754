function [out] = OMP(A, A1, y, g)
out = zeros(g, 1);

thet_i = 0;
supp = zeros(1, g);
r = y;
for it = 1:64
    if norm(r, "fro")<= 1e-5
        break;
    end
    [J,j] = max(abs(r'*A1));
    supp(j) = 1;
    sub_rad = A(:, (supp~=0));
    thet_i = pinv(sub_rad)*y;
    r = y - sub_rad*thet_i;
end
count1 = 1;
for count = 1:g
    if supp(count) == 1
        out(count) = thet_i(count1);
        count1 = count1+1;
    end
end
end