%
% Parzen - compute 1D Parzen pdf
%
% [p] = Parzen1D( X, data, std_dev )    
%
%  data - data samples
%  sigma - standard deviation for all windows
%  X - X axis length
%  p    - estimated 2D PDF
%

function [p] = Parzen1D( X, data, std_dev)
    % Number of data points
    N = length(data);
    p = zeros(size(X));
    
%     Cycles through each data point, adding windows
    for i = 1:N
        win = normpdf(X, data(1,i), std_dev);
        p = p + win;
    end
%     Normalize
    p = (1/N)*p;
end
