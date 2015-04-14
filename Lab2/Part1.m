%---------------------------------------------------------------
% Part 1: Model Estimation: 1D Case
%---------------------------------------------------------------
% Load data
data = matfile('lab2_1');
A = data.a;
B = data.b;

% True data distribution
mu_A = 5;
sigma_A = 1;
lambda_B = 1;

% Create true PDFs
% Leave 5 standard deviations on either side. Checked & reaches 0.000
% probability for exponential distribution as well.
X = linspace(0,10,100);
pdf_A = normpdf(X, mu_A, sigma_A);
pdf_B = exppdf(X, lambda_B);

% Section 1: Parametric Estimation: Gaussian
%---------------------------------------------------------------
% Data Set A
% Estimate Data
% Not sure if normfit is the right way to go. It says to use censoring for
% ML Estimate with normfit. Will ask TA about this.
[est_mu_A, est_sigma_A] = normfit(A);
est_gauss_pdf_A = normpdf(X, est_mu_A, est_sigma_A);
% Plot both PDFs
figure; 
plot(X, pdf_A, X, est_gauss_pdf_A, 'r');
xlabel('x');
ylabel('p(x)');
title('Parametric Estimation - Gaussian');
legend('True PDF','Estimated PDF');

% Data Set B
% Estimate Data
[est_mu_B, est_sigma_B] = normfit(B);
est_gauss_pdf_B = normpdf(X, est_mu_B, est_sigma_B);
% Plot both PDFs
figure; 
plot(X, pdf_B, X, est_gauss_pdf_B, 'r');
xlabel('x');
ylabel('p(x)');
title('Parametric Estimation - Gaussian');
legend('True PDF','Estimated PDF');

% Section 2: Parametric Estimation: Exponential
%---------------------------------------------------------------
% Data Set A
% Estimate Data
est_lambda_A = expfit(A);
est_exp_pdf_A = exppdf(X, est_mu_A);
% Plot both PDFs
figure; 
plot(X, pdf_A, X, est_exp_pdf_A, 'r');
xlabel('x');
ylabel('p(x)');
title('Parametric Estimation - Exponential');
legend('True PDF','Estimated PDF', 'Estimated with expfit');

% Data Set B
% Estimate Data
est_lambda_B = expfit(B);
est_exp_pdf_B = exppdf(X, est_mu_B);
% Plot both PDFs
figure; 
plot(X, pdf_B, X, est_exp_pdf_B, 'r');
xlabel('x');
ylabel('p(x)');
title('Parametric Estimation - Exponential');
legend('True PDF','Estimated PDF', 'Estimated with expfit');

% Section 3: Parametric Estimation: Uniform
%---------------------------------------------------------------
% Data Set A
% Estimate Data
% Tried doing it both ways for this. They are different. I think it should
% be unifit(A) and minu and plus this from the mean, it fits the data
% better. Will ask the TA.
est_unif_pdf_A2 = unifpdf(X, est_mu_A - unifit(A), est_mu_A + unifit(A));
est_unif_pdf_A = unifpdf(X, min(A), max(A));
% Plot both PDFs
figure; 
plot(X, pdf_A, X, est_unif_pdf_A, 'r', X, est_unif_pdf_A2, 'g');
xlabel('x');
ylabel('p(x)');
title('Parametric Estimation - Uniform');
legend('True PDF','Estimated min/max', 'Estimated mean/variance');

% Data Set B
% Estimate Data
est_unif_pdf_B2 = unifpdf(X, est_mu_B - unifit(B), est_mu_B + unifit(B));
est_unif_pdf_B = unifpdf(X, min(B), max(B));
% Plot both PDFs
figure; 
plot(X, pdf_B, X, est_unif_pdf_B, 'r', X, est_unif_pdf_A2, 'g');
xlabel('x');
ylabel('p(x)');
title('Parametric Estimation - Uniform');
legend('True PDF','Estimated min/max', 'Estimated mean/variance');

% Section 4: Non-Parametric Estimation: Parzen Method
%---------------------------------------------------------------
% Standard deviations for 2 parzen windows
sigma_window1 = 0.1;
sigma_window2 = 0.4;

% Data Set A
% Window Size .1
pdf_parzen_A_win1 = Parzen1D(X, A, sigma_window1);
% Plot both PDFs
figure;
plot(X, pdf_A, X, pdf_parzen_A_win1, 'r');
xlabel('x');
ylabel('p(x)');
title('Non-Parametric Estimation - Parzen Window Size 0.1');
legend('True PDF','Estimated PDF');

% Window Size .4
pdf_parzen_A_win2 = Parzen1D(X, A, sigma_window2);
% Plot both PDFs
figure;
plot(X, pdf_A, X, pdf_parzen_A_win2, 'r');
xlabel('x');
ylabel('p(x)');
title('Non-Parametric Estimation - Parzen Window Size 0.4');
legend('True PDF','Estimated PDF');

% Data Set B
% Window Size .1
pdf_parzen_B_win1 = Parzen1D(X, B, sigma_window1);
% Plot both PDFs
figure;
plot(X, pdf_B, X, pdf_parzen_B_win1, 'r');
xlabel('x');
ylabel('p(x)');
title('Non-Parametric Estimation - Parzen Window Size 0.1');
legend('True PDF','Estimated PDF');

% Window Size .4
pdf_parzen_B_win2 = Parzen1D(X, B, sigma_window2);
% Plot both PDFs
figure;
plot(X, pdf_B, X, pdf_parzen_B_win2, 'r');
xlabel('x');
ylabel('p(x)');
title('Non-Parametric Estimation - Parzen Window Size 0.4');
legend('True PDF','Estimated PDF');
figure;
