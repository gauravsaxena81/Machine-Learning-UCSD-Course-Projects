function x = gi(yiminus1, yi, xbar, i, ff, w)
%ff : array of feature functions
%w  : array of weights
    sum = 0;
    for i = 0:size(ff)
        sum = sum + w(i)* ff{1}(yiminus1, yi, xbar, i);
    end