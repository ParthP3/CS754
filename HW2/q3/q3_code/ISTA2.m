function [theta_rec] = ISTA2(y,A, W)
alpha = 200;
theta = randn(16, 16)+100;
theta_new = theta;
l = 1/(2*alpha);
i = 0;
while i<60
    [a,b,c,d] = W(theta);
    a = reshape(a', [], 1);
    b = reshape(b', [], 1);
    c = reshape(c', [], 1);
    d = reshape(d', [], 1);
    w = A'*(y-A*a);
    x = A'*(y-A*b);
    y1 = A'*(y-A*c);
    z = A'*(y-A*d);
    w = reshape(w', [8,8])';
    x = reshape(x', [8,8])';
    y1 = reshape(y1', [8,8])';
    z = reshape(z', [8,8])';
    H1 = idwt2(w,x,y1,z,'db1');
    H = theta + (1/alpha)*H1;
    theta_new = wthresh(H, "s", l);
    if(norm(theta_new -theta) <= 0.5)
        break
    end
    norm(theta_new -theta);
    theta = theta_new;
    i=i+1;
end
theta_rec = theta;

end