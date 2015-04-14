%
% MLClassifier - Classifies a point based on ML Estimate
%
% [p] = Parzen1D( A, B, C )    
%
%  data - data samples
%  sigma - standard deviation for all windows
%  X - X axis length
%  p    - estimated 2D PDF
%

function [determined_class] = MLClassifier( X, Y, sigma_A, mu_A, sigma_B, mu_B, sigma_C, mu_C)
    P = [X; Y];
%   Calculates probability of a certain class given the coordinates X,Y
%   p(A|x) Class that is the most likely given x is maximum likelihood
    p_A = 1/sqrt(det(sigma_A)) * exp(-0.5*((P - mu_A')'*inv(sigma_A)*(P - mu_A')));
    p_B = 1/sqrt(det(sigma_B)) * exp(-0.5*((P - mu_B')'*inv(sigma_B)*(P - mu_B')));
    p_C = 1/sqrt(det(sigma_C)) * exp(-0.5*((P - mu_C')'*inv(sigma_C)*(P - mu_C')));

    if( p_A > p_B && p_A > p_C)
        determined_class = 1;
    elseif( p_B > p_C)
        determined_class = 2;
    else
        determined_class = 3;
    end
end
