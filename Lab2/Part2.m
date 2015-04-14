%---------------------------------------------------------------
% Part 2: Model Estimation: 2D Case
%---------------------------------------------------------------
% Load data
data = matfile('lab2_2');
A = data.al;
B = data.bl;
C = data.cl;

% Section 1: Parametric Estimation: Gaussian
%---------------------------------------------------------------
% Calculate means and variances of clusters
N = length(A);
mu_A = mean(A);

% Used to check that covariance is unbiased estimate
% Covariance is multiplied by N/(N-1) to unbias estimate.
% the MATLAb function cov does this automatically
% sigma_A = N/(N-1) * 1/N*sum((A(:,1)-mu_A(1)).^2)

% Calculates mean and covariance matrices
sigma_A = cov(A);
mu_B = mean(B);
sigma_B = cov(B);
mu_C = mean(C);
sigma_C = cov(C);

[x_values, y_values, ML_Classifier] = makeGrid(1, A, B, C);
for i = 1:size(x_values, 2)
    for j = 1:size(y_values, 2)
        ML_Classifier(j,i) = MLClassifier(x_values(i), y_values(j), sigma_A, mu_A, sigma_B, mu_B, sigma_C, mu_C);
    end
end
[c1, h1] = contour(x_values, y_values, ML_Classifier, 2, 'c');

%Draw the clusters and contours
hold on;

% Section 2: Non-Parametric Estimation: Gaussian Parzen Window
%---------------------------------------------------------------
% Predefined variance
sigma = 20;
% Changed resolution to 1 -->.25 from last lab too small for this data set
res = 1;

% Gaussian Window needed for Parzen2D
% Window of alpha 20 std deviations
% 20 std deviations*20 sigma/1 resolution = 400
% Made it so large so that boundary line is smooth throughout whole region.
% At 4 or 5 standard deviations there were square borders because of the
% matrix shape.
win = gausswin(400,20)*gausswin(400,20)';

% Create pdfs using Parzen2D
pdf_A_parzen = Parzen2D(A, [res, min(x_values), min(y_values), max(x_values), max(y_values)], win);
pdf_B_parzen = Parzen2D(B, [res, min(x_values), min(y_values), max(x_values), max(y_values)], win);
pdf_C_parzen = Parzen2D(C, [res, min(x_values), min(y_values), max(x_values), max(y_values)], win);

[x_values, y_values, ML_Classifier_Parzen] = makeGrid(1, A, B, C);
for i = 1:size(x_values, 2)
    for j = 1:size(y_values, 2)
        if( pdf_A_parzen(j,i) > pdf_B_parzen(j,i) && pdf_A_parzen(j,i) > pdf_C_parzen(j,i))
            ML_Classifier_Parzen(j,i) = 1;
        elseif( pdf_B_parzen(j,i) > pdf_C_parzen(j,i))
            ML_Classifier_Parzen(j,i) = 2;
        else
            ML_Classifier_Parzen(j,i) = 3;
        end
        
    end
end
[c1, h1] = contour(x_values, y_values, ML_Classifier_Parzen, 2, 'm');

% Plots both graphs
hold on;
plot(A(:,1), A(:,2), 'r.')
plot(B(:,1), B(:,2), 'b.')
plot(C(:,1), C(:,2), 'g.')
xlabel('x');
ylabel('y');
title('2D Clusters - Parametric Estimation - ML Boundary');
legend('Parametric', 'Non-Parametric', 'Cluster A','Cluster B', 'Cluster C');
axis equal;
