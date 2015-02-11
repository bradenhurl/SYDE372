classdef Functions
    
    methods (Static=true)
%---------------------------------------------------------------
% Part 3: Classifier Functions
%---------------------------------------------------------------
        function [STDevContour] = DrawContour (Mu, Sigma)
            ContourSize = 10000;
            unitContour = [cos((1:ContourSize)/ContourSize*2*pi()); ...
                            sin((1:ContourSize)/ContourSize*2*pi())]';
            R = chol(Sigma);
            STDevContour = repmat(Mu,length(unitContour),1) + unitContour*R;
        end
        
        function [Distance] = CalcDistance(point1, point2, class)
            Distance = sqrt((point1 - class.mu(1))^2 + (point2 - class.mu(2))^2);
        end
        
        function [determined_class] = MED(point_x, point_y, varargin)
            min_class = -1;
            min_dist = -1;
            for i = 1:length(varargin)
                distance = sqrt((point_x - varargin{i}.mu(1))^2 + (point_y - varargin{i}.mu(2))^2);
                %distance = Functions.CalcDistance(point_x, point_y, varargin(i));
                if distance < min_dist | min_dist == -1
                    min_dist = distance;
                    min_class = i;
                end                    
            end
            determined_class = min_class;
        end
        
        function [determined_class] = GED(point_x, point_y, varargin)
            min_class = -1;
            min_dist = -1;
            matrix = [point_x, point_y];
            for i = 1:length(varargin)
                distance = (matrix - varargin{i}.mu)*inv(varargin{i}.sigma)...
                    *(matrix - varargin{i}.mu)';
                if distance < min_dist | min_dist == -1
                    min_dist = distance;
                    min_class = i;
                end                    
            end
            determined_class = min_class;
        end
        
        function [determined_class] = MAP(point_x, point_y, varargin)
            min_class = -1;
            min_dist = -1;
            matrix = [point_x, point_y];
            sum = 0;
            
            for i = 1:length(varargin)
                sum = sum + varargin{i}.size;
            end
            
            for i = 1:length(varargin)
                distance = (matrix - varargin{i}.mu)*inv(varargin{i}.sigma)...
                    *(matrix - varargin{i}.mu)' - 2*log(varargin{i}.size/sum)...
                    +log(det(varargin{i}.sigma));
                if distance < min_dist | min_dist == -1
                    min_dist = distance;
                    min_class = i;
                end
            end
            determined_class = min_class;
        end
        
        function [distance] = NNDistance(dataPoint, clusterPoint)
            distance = sqrt((dataPoint(1) - clusterPoint(1))^2 + (dataPoint(2) - clusterPoint(2))^2);
        end
        
        function [determined_class] = NearestNeighbor(point_x, point_y, varargin)
            min_class = -1;
            min_dist = -1;
            matrix = [point_x, point_y];
            
            for i = 1:length(varargin)
                for j = 1:size(varargin{i},1)
                    distance = Functions.NNDistance(matrix, varargin{i}(j,:));
                    if distance < min_dist | min_dist == -1
                        min_dist = distance;
                        min_class = i;
                    end
                end
            end
            determined_class = min_class;
        end
        
        function [determined_class] = kNearestNeighbor(point_x, point_y, varargin)
            
            initArray1 = false;
            initArray2 = false;
            initArray3 = false;
            determined_class = 0;
            
            matrix = [point_x, point_y];
            
            for i = 1:length(varargin)
                if i == 1
                    distArray1 = zeros(1,size(varargin{1},1));
                    initArray1 = true;
                end
                if i == 2
                    distArray2 = zeros(1,size(varargin{2},1));
                    initArray2 = true;
                end
                if i == 3
                    distArray3 = zeros(1,size(varargin{3},1));
                    initArray3 = true;
                end
            end
            
            for i = 1:length(varargin)
                for j = 1:size(varargin{i},1)
                    if i == 1 && initArray1 == true
                        distArray1(1,j) = Functions.NNDistance(matrix, varargin{i}(j,:));
                    end
                    if i == 2 && initArray2 == true
                        distArray2(1,j) = Functions.NNDistance(matrix, varargin{i}(j,:));
                    end
                    if i == 3 && initArray3 == true
                        distArray3(1,j) = Functions.NNDistance(matrix, varargin{i}(j,:));
                    end
                end
            end
            
            if initArray1 == true
                sortArray1 = sort(distArray1);
                avg1 = sum(sortArray1(1:5))/5;
            end
            if initArray2 == true
                sortArray2 = sort(distArray2);
                avg2 = sum(sortArray2(1:5))/5;
            end
            if initArray3 == true
                sortArray3 = sort(distArray3);
                avg3 = sum(sortArray3(1:5))/5;
            end
            
            if length(varargin) == 2
                if min(avg1, avg2) == avg1
                    determined_class = 1;
                else
                    determined_class = 2;
                end
            end
            if length(varargin) == 3
                minimumDistance = min(avg1, min(avg2, avg3));
                if minimumDistance == avg1
                    determined_class = 1;
                elseif minimumDistance == avg2
                    determined_class = 2;
                else
                    determined_class = 3;
                end
            end
            
        end
        
%---------------------------------------------------------------
% Part 4: Error Analysis Functions
%---------------------------------------------------------------
        function [P_x_cond] = getCond(class, x_values, y_values)
            P_x_cond = zeros(size(y_values,2), size(x_values,2));
            for i=1:size(x_values, 2)
                for j=1:size(y_values, 2)
                    P_x_cond(j,i) = (1/((2*pi)*sqrt(det(class.sigma))))*...
                        exp(-(([x_values(i), y_values(j)]-class.mu)*inv(class.sigma)*([x_values(i), y_values(j)]-class.mu)')/2);
                end
            end
        end

    end
end
