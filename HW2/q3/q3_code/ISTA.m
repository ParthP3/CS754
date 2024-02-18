function [theta_rec] = ISTA(y, A, alpha)
theta = randn(64, 1)+100;
theta_new = theta;
l = 1/(2*alpha);
i = 0;
while i<500
    H = theta + (1/alpha)*(A'*(y-A*theta));
    theta_new = wthresh(H, "s", l);
    if(vecnorm(theta_new -theta) <= 0.1)
        break
    end
    theta = theta_new;
    i=i+1;
end
theta_rec = theta;
end