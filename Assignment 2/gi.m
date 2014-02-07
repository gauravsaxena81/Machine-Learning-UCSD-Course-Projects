function x = gi(yiminus1, yi, xbar, i, ff, w)
%ff : array of feature functions
%w  : array of weights
    sum = 0;
    size = size(ff);
    for i = 0:size
        sum = sum + w(i)* f{1}(yiminus1, yi, xbar, i);