function [X, class]  = function_prepare_DL_data(ext_idx, TABLE_ML_TOTAL, TABLE_ML_TOTAL_upper, TABLE_ML_TOTAL_lower, Clinical_Scores_TOTAL_median)

% Convert the tables into double
TABLE_ML_TOTAL_short = removevars(TABLE_ML_TOTAL, 'Class');             
X_total_DL           = table2array(table2array(TABLE_ML_TOTAL_short(:,1:44)));   
X_upper_DL           = table2array(table2array(TABLE_ML_TOTAL_upper(:,2:12)));
X_lower_DL           = table2array(table2array(TABLE_ML_TOTAL_lower(:,2:12)));

% Pick the input (R) and output (class) belonging to the current extremity
if ext_idx == 1 % dystonia lower extremities
    class = [categorical(table2array(Clinical_Scores_TOTAL_median(:,2))) ; categorical(table2array(Clinical_Scores_TOTAL_median(:,1)))];
    X = X_lower_DL;    
end
if ext_idx == 2 % dystonia upper extremities 
    class = [categorical(table2array(Clinical_Scores_TOTAL_median(:,4))) ; categorical(table2array(Clinical_Scores_TOTAL_median(:,3)))];
    X = X_upper_DL;    
end
if ext_idx == 3 % choreoathetosis lower extremities
    class = [categorical(table2array(Clinical_Scores_TOTAL_median(:,8))) ; categorical(table2array(Clinical_Scores_TOTAL_median(:,7)))];
    X = X_lower_DL;    
end
if ext_idx == 4 % choreoathetosis upper extremities
    class = [categorical(table2array(Clinical_Scores_TOTAL_median(:,10))) ; categorical(table2array(Clinical_Scores_TOTAL_median(:,9)))];
    X = X_upper_DL;    
end
if ext_idx == 5 % dystonia total
    class = categorical((table2array(Clinical_Scores_TOTAL_median(:,1)) + table2array(Clinical_Scores_TOTAL_median(:,2)) + table2array(Clinical_Scores_TOTAL_median(:,3)) + table2array(Clinical_Scores_TOTAL_median(:,4))) /4);
    X = X_total_DL;
end
if ext_idx == 6 % choreoathetosis total
    class = categorical((table2array(Clinical_Scores_TOTAL_median(:,8)) + table2array(Clinical_Scores_TOTAL_median(:,9)) + table2array(Clinical_Scores_TOTAL_median(:,10)) + table2array(Clinical_Scores_TOTAL_median(:,11))) /4);
    X = X_total_DL;    
end

% Make sure all the classes are in the categorical output vector. Necessary
% for creating confusion matrices
class(end+1,1)='0';
class(end+1,1)='1';
class(end+1,1)='2';
class(end+1,1)='3';
class(end+1,1)='4';
class(end-4:end,:)=[];

% Delete unscored samples from the dataset
Undefined = isundefined(class);
X(Undefined,:) = [];
class(Undefined) = [];

end
