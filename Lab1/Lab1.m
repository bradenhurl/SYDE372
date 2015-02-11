% SYDE 372 - Lab 1
% Braden Hurl - 20406936
% Nicholas Chen - 20375332

clc;
close all;
clear all; 

% % % Starts timer
tic

% Creating Class Data Parameters
A = defineClass([5 10],[8 0; 0 4],200);
B = defineClass([10 15],[8 0; 0 4],200);
C = defineClass([5 10],[8 4; 4 40],100);
D = defineClass([15 10],[8 0; 0 8],200);
E = defineClass([10 5],[10 -5; -5 20],150);

%---------------------------------------------------------------
% Part 2: Generating Clusters
%---------------------------------------------------------------

R1=chol(A.sigma);
A_Cluster = repmat(A.mu,A.size,1) + randn(A.size,2)*R1;

R2=chol(B.sigma);
B_Cluster = repmat(B.mu,B.size,1) + randn(B.size,2)*R2;

R3=chol(C.sigma);
C_Cluster = repmat(C.mu,C.size,1) + randn(C.size,2)*R3;

R4=chol(D.sigma);
D_Cluster = repmat(D.mu,D.size,1) + randn(D.size,2)*R4;

R5=chol(E.sigma);
E_Cluster = repmat(E.mu,E.size,1) + randn(E.size,2)*R5;

A_contour = Functions.DrawContour(A.mu, A.sigma);
B_contour = Functions.DrawContour(B.mu, B.sigma);
C_contour = Functions.DrawContour(C.mu, C.sigma);
D_contour = Functions.DrawContour(D.mu, D.sigma);
E_contour = Functions.DrawContour(E.mu, E.sigma);

%Draw the A-B clusters and contours
figure(1);
hold on;
plot(A_Cluster(:,1), A_Cluster(:,2), 'r.')
plot(B_Cluster(:,1), B_Cluster(:,2), 'b.')
plot(A_contour(:,1), A_contour(:,2), 'r-');
plot(B_contour(:,1), B_contour(:,2), 'b-');
axis equal;

%Draw the C-D-E clusters and contours
figure(2);
hold on;
plot(C_Cluster(:,1), C_Cluster(:,2), 'r.')
plot(D_Cluster(:,1), D_Cluster(:,2), 'b.')
plot(E_Cluster(:,1), E_Cluster(:,2), 'g.')
plot(C_contour(:,1), C_contour(:,2), 'r-');
plot(D_contour(:,1), D_contour(:,2), 'b-');
plot(E_contour(:,1), E_contour(:,2), 'g-');
axis equal;

%---------------------------------------------------------------
% Part 3: Classifiers
%---------------------------------------------------------------

%AB_Classifiers-------------------------------------------

%First graph Classifier
figure(3);
hold on;
plot(A_Cluster(:,1), A_Cluster(:,2), 'r.')
plot(B_Cluster(:,1), B_Cluster(:,2), 'b.')
plot(A_contour(:,1), A_contour(:,2), 'r-');
plot(B_contour(:,1), B_contour(:,2), 'b-');
axis equal;

[x_valuesAB, y_valuesAB, AB_MED] = makeGrid(0.25, A_Cluster, B_Cluster);
for i = 1:size(x_valuesAB, 2)
    for j = 1:size(y_valuesAB, 2)
        AB_MED(j,i) = Functions.MED(x_valuesAB(i), y_valuesAB(j), A, B);
    end
end
[c1, h1] = contour(x_valuesAB, y_valuesAB, AB_MED, 1, ':c');

%GED Classifier
[x_valuesAB, y_valuesAB, AB_GED] = makeGrid(0.25, A_Cluster, B_Cluster);
for i = 1:size(x_valuesAB, 2)
    for j = 1:size(y_valuesAB, 2)
        AB_GED(j,i) = Functions.GED(x_valuesAB(i), y_valuesAB(j), A, B);
    end
end
[c2, h2] = contour(x_valuesAB, y_valuesAB, AB_GED, 1, '-m');

%MAP Classifier
[x_valuesAB, y_valuesAB, AB_MAP] = makeGrid(0.25, A_Cluster, B_Cluster);
for i = 1:size(x_valuesAB, 2)
    for j = 1:size(y_valuesAB, 2)
        AB_MAP(j,i) = Functions.MAP(x_valuesAB(i), y_valuesAB(j), A, B);
    end
end
[c3, h3] = contour(x_valuesAB, y_valuesAB, AB_MAP, 1, '-.y');

% % Second graph Classifiers
% figure(4);
% hold on;
% plot(A_Cluster(:,1), A_Cluster(:,2), 'r.')
% plot(B_Cluster(:,1), B_Cluster(:,2), 'b.')
% plot(A_contour(:,1), A_contour(:,2), 'r-');
% plot(B_contour(:,1), B_contour(:,2), 'b-');
% axis equal;
% 
% % Nearest Neighbor
% [x_valuesAB, y_valuesAB, AB_NN] = makeGrid(0.25, A_Cluster, B_Cluster);
% for i = 1:size(x_valuesAB, 2)
%     for j = 1:size(y_valuesAB, 2)
%         AB_NN(j,i) = Functions.NearestNeighbor(x_valuesAB(i), y_valuesAB(j), A_Cluster, B_Cluster);
%     end
% end
% [c4, h4] = contour(x_valuesAB, y_valuesAB, AB_NN, 1, '-c');
% 
% %K Nearest Neighbor
% [x_valuesAB, y_valuesAB, AB_5NN] = makeGrid(0.25, A_Cluster, B_Cluster);
% for i = 1:size(x_valuesAB, 2)
%     for j = 1:size(y_valuesAB, 2)
%         AB_5NN(j,i) = Functions.kNearestNeighbor(x_valuesAB(i), y_valuesAB(j), A_Cluster, B_Cluster);
%     end
% end
% [c5, h5] = contour(x_valuesAB, y_valuesAB, AB_5NN, 1, '-m');
% 
% 
% CDE_Classifiers-------------------------------------------

% First graph Classifiers
figure(5);
hold on;
plot(C_Cluster(:,1), C_Cluster(:,2), 'r.')
plot(D_Cluster(:,1), D_Cluster(:,2), 'b.')
plot(E_Cluster(:,1), E_Cluster(:,2), 'g.')
plot(C_contour(:,1), C_contour(:,2), 'r-');
plot(D_contour(:,1), D_contour(:,2), 'b-');
plot(E_contour(:,1), E_contour(:,2), 'g-');
axis equal;

[x_valuesCDE, y_valuesCDE, CDE_MED] = makeGrid(0.25, C_Cluster, D_Cluster, E_Cluster);
for i = 1:size(x_valuesCDE, 2)
    for j = 1:size(y_valuesCDE, 2)
        CDE_MED(j,i) = Functions.MED(x_valuesCDE(i), y_valuesCDE(j), C, D, E);
    end
end
[c1, h1] = contour(x_valuesCDE, y_valuesCDE, CDE_MED, 2, ':c');

%GED Classifier
[x_valuesCDE, y_valuesCDE, CDE_GED] = makeGrid(0.25, C_Cluster, D_Cluster, E_Cluster);
for i = 1:size(x_valuesCDE, 2)
    for j = 1:size(y_valuesCDE, 2)
        CDE_GED(j,i) = Functions.GED(x_valuesCDE(i), y_valuesCDE(j), C, D, E);
    end
end
[c2, h2] = contour(x_valuesCDE, y_valuesCDE, CDE_GED, 2, '-m');

%MAP Classifier
[x_valuesCDE, y_valuesCDE, CDE_MAP] = makeGrid(0.25, C_Cluster, D_Cluster, E_Cluster);
for i = 1:size(x_valuesCDE, 2)
    for j = 1:size(y_valuesCDE, 2)
        CDE_MAP(j,i) = Functions.MAP(x_valuesCDE(i), y_valuesCDE(j), C, D, E);
    end
end
[c3, h3] = contour(x_valuesCDE, y_valuesCDE, CDE_MAP, 2, '-.y');

% Second graph Classifiers
figure(6);
hold on;
plot(C_Cluster(:,1), C_Cluster(:,2), 'r.')
plot(D_Cluster(:,1), D_Cluster(:,2), 'b.')
plot(E_Cluster(:,1), E_Cluster(:,2), 'g.')
plot(C_contour(:,1), C_contour(:,2), 'r-');
plot(D_contour(:,1), D_contour(:,2), 'b-');
plot(E_contour(:,1), E_contour(:,2), 'g-');
axis equal;

% % Nearest Neighbor
% [x_valuesCDE, y_valuesCDE, CDE_NN] = makeGrid(0.25, C_Cluster, D_Cluster, E_Cluster);
% for i = 1:size(x_valuesCDE, 2)
%     for j = 1:size(y_valuesCDE, 2)
%         CDE_NN(j,i) = Functions.NearestNeighbor(x_valuesCDE(i), y_valuesCDE(j), C_Cluster, D_Cluster, E_Cluster);
%     end
% end
% [c4, h4] = contour(x_valuesCDE, y_valuesCDE, CDE_NN, 2, '-c');
% 
% %K Nearest Neighbor
% [x_valuesCDE, y_valuesCDE, CDE_5NN] = makeGrid(0.25, C_Cluster, D_Cluster, E_Cluster);
% for i = 1:size(x_valuesCDE, 2)
%     for j = 1:size(y_valuesCDE, 2)
%         CDE_5NN(j,i) = Functions.kNearestNeighbor(x_valuesCDE(i), y_valuesCDE(j), C_Cluster, D_Cluster, E_Cluster);
%     end
% end
% [c5, h5] = contour(x_valuesCDE, y_valuesCDE, CDE_5NN, 2, '-m');


%---------------------------------------------------------------
% Part 4: Error Analysis
%---------------------------------------------------------------
% 4.1 Error Probabilities Equation from p. 68 of course notes
%---------------------------------------------------------------
% % Calculate probabilities of classes
Prob_A = A.size/(A.size + B.size);
Prob_B = B.size/(A.size + B.size);
Prob_C = C.size/(C.size + D.size + E.size);
Prob_D = D.size/(C.size + D.size + E.size);
Prob_E = E.size/(C.size + D.size + E.size);

% Generate Binary Regions of Classes
Region_A = zeros(size(AB_MAP));
Region_B = zeros(size(AB_MAP));
Region_C = zeros(size(CDE_MAP));
Region_D = zeros(size(CDE_MAP));
Region_E = zeros(size(CDE_MAP));

% Find regions where certain class is most likely
Region_A(find(AB_MAP == 1)) = 1;
Region_B(find(AB_MAP == 2)) = 1;
Region_C(find(CDE_MAP == 1)) = 1;
Region_D(find(CDE_MAP == 2)) = 1;
Region_E(find(CDE_MAP == 3)) = 1;

% Create conditional probabilities
P_x_condA = Functions.getCond(A, x_valuesAB, y_valuesAB);
P_x_condB = Functions.getCond(B, x_valuesAB, y_valuesAB);
P_x_condC = Functions.getCond(C, x_valuesCDE, y_valuesCDE);
P_x_condD = Functions.getCond(D, x_valuesCDE, y_valuesCDE);
P_x_condE = Functions.getCond(E, x_valuesCDE, y_valuesCDE);

% Find probability of error for each point in grid
P_Error_Points_AB = P_x_condB*Prob_B.*Region_A + P_x_condA*Prob_A.*Region_B;
P_Error_AB = sum(P_Error_Points_AB(:))*0.25^2;
P_Error_Points_CDE = P_x_condC*Prob_C.*(Region_D + Region_E) + ...
    P_x_condD*Prob_D.*(Region_C + Region_E) + P_x_condE*Prob_E.*(Region_D + Region_C);
P_Error_CDE = sum(P_Error_Points_CDE(:))*0.25^2;

% For checking to make sure probability of error is correct
P_Error_Points_check_AB = P_x_condB*Prob_B.*Region_B + P_x_condA*Prob_A.*Region_A;
P_Error_check_AB = 1 - sum(P_Error_Points_check_AB(:))*0.25^2;
P_Error_Points_check_CDE = P_x_condC*Prob_C.*(Region_C) + ...
    P_x_condD*Prob_D.*(Region_D) + P_x_condE*Prob_E.*(Region_E);
P_Error_check_CDE = 1 - sum(P_Error_Points_check_CDE(:))*0.25^2;


% 4.2 Confusion Matrices
%---------------------------------------------------------------
AB_Cluster = [[A_Cluster, ones(A.size, 1)]; [B_Cluster, ones(B.size, 1)*2]];
CDE_Cluster = [[C_Cluster, ones(C.size, 1)]; [D_Cluster, ones(D.size, 1)*2]; [E_Cluster, ones(E.size, 1)*3]];

% Creates Clusters with Class associated for each data point
Cluster_Class_AB_MED = griddata(x_valuesAB(1,:), y_valuesAB(1,:), AB_MED, AB_Cluster(:, 1), AB_Cluster(:,2), 'nearest');
Cluster_Class_AB_GED = griddata(x_valuesAB(1,:), y_valuesAB(1,:), AB_GED, AB_Cluster(:, 1), AB_Cluster(:,2), 'nearest');
Cluster_Class_AB_MAP = griddata(x_valuesAB(1,:), y_valuesAB(1,:), AB_MAP, AB_Cluster(:, 1), AB_Cluster(:,2), 'nearest');
Cluster_Class_CDE_MED = griddata(x_valuesCDE(1,:), y_valuesCDE(1,:), CDE_MED, CDE_Cluster(:, 1), CDE_Cluster(:,2), 'nearest');
Cluster_Class_CDE_GED = griddata(x_valuesCDE(1,:), y_valuesCDE(1,:), CDE_GED, CDE_Cluster(:, 1), CDE_Cluster(:,2), 'nearest');
Cluster_Class_CDE_MAP = griddata(x_valuesCDE(1,:), y_valuesCDE(1,:), CDE_MAP, CDE_Cluster(:, 1), CDE_Cluster(:,2), 'nearest');

% Creates confusion matrices
conf_AB_MED = confusionmat(AB_Cluster(:,3), Cluster_Class_AB_MED);
conf_AB_GED = confusionmat(AB_Cluster(:,3), Cluster_Class_AB_GED);
conf_AB_MAP = confusionmat(AB_Cluster(:,3), Cluster_Class_AB_MAP);
conf_CDE_MED = confusionmat(CDE_Cluster(:,3), Cluster_Class_CDE_MED);
conf_CDE_GED = confusionmat(CDE_Cluster(:,3), Cluster_Class_CDE_GED);
conf_CDE_MAP = confusionmat(CDE_Cluster(:,3), Cluster_Class_CDE_MAP);

% Generate Test data for NN
A_Test_Cluster = repmat(A.mu,A.size,1) + randn(A.size,2)*R1;
B_Test_Cluster = repmat(B.mu,B.size,1) + randn(B.size,2)*R2;
C_Test_Cluster = repmat(C.mu,C.size,1) + randn(C.size,2)*R3;
D_Test_Cluster = repmat(D.mu,D.size,1) + randn(D.size,2)*R4;
E_Test_Cluster = repmat(E.mu,E.size,1) + randn(E.size,2)*R5;

% Create Case Test Clusters
AB_Test_Cluster = [[A_Test_Cluster, ones(A.size, 1)]; [B_Test_Cluster, ones(B.size, 1)*2]];
CDE_Test_Cluster = [[C_Test_Cluster, ones(C.size, 1)]; [D_Test_Cluster, ones(D.size, 1)*2]; [E_Test_Cluster, ones(E.size, 1)*3]];

% Creates Clusters with Class associate for each point
Test_Cluster_Class_AB_NN = griddata(x_valuesAB(1,:), y_valuesAB(1,:), AB_NN, AB_Test_Cluster(:, 1), AB_Test_Cluster(:,2), 'nearest');
Test_Cluster_Class_AB_5NN = griddata(x_valuesAB(1,:), y_valuesAB(1,:), AB_5NN, AB_Test_Cluster(:, 1), AB_Test_Cluster(:,2), 'nearest');
Test_Cluster_Class_CDE_NN = griddata(x_valuesCDE(1,:), y_valuesCDE(1,:), CDE_NN, CDE_Test_Cluster(:, 1), CDE_Test_Cluster(:,2), 'nearest');
Test_Cluster_Class_CDE_5NN = griddata(x_valuesCDE(1,:), y_valuesCDE(1,:), CDE_5NN, CDE_Test_Cluster(:, 1), CDE_Test_Cluster(:,2), 'nearest');

% Creates confusion matrices
conf_AB_NN = confusionmat(AB_Test_Cluster(:,3), Test_Cluster_Class_AB_NN);
conf_AB_5NN = confusionmat(AB_Test_Cluster(:,3), Test_Cluster_Class_AB_5NN);
conf_CDE_NN = confusionmat(CDE_Test_Cluster(:,3), Test_Cluster_Class_CDE_NN);
conf_CDE_5NN = confusionmat(CDE_Test_Cluster(:,3), Test_Cluster_Class_CDE_5NN);

filename = 'ConfusionMatrices.xlsx';
xlswrite(filename,conf_AB_MED,1,'A1')
xlswrite(filename,conf_AB_GED,1,'A3')
xlswrite(filename,conf_AB_MAP,1,'A5')
xlswrite(filename,conf_AB_NN,1,'A7')
xlswrite(filename,conf_AB_5NN,1,'A9')
xlswrite(filename,conf_CDE_MED,1,'A12')
xlswrite(filename,conf_CDE_GED,1,'A15')
xlswrite(filename,conf_CDE_MAP,1,'A18')
xlswrite(filename,conf_CDE_NN,1,'A21')
xlswrite(filename,conf_CDE_5NN,1,'A24')

% Stops timer
toc
