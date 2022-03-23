function [dataframes, dataframeTrain_dystonia_upper , dataframeTrain_dystonia_lower , dataframeTrain_choreoathetosis_upper , dataframeTrain_choreoathetosis_lower , dataframeTrain_dystonia_total , dataframeTrain_choreoathetosis_total] = function_create_dataframes(table_ML_total, table_ML_total_upper,table_ML_total_lower, class, Clinical_Scores_TOTAL_median)

% Calculate features
clear T_*
for i=1:44
    X= table2array(vertcat(table_ML_total{:,i+1}));    
    
    T_max(:,i)           = max(X,[],2);
    T_min(:,i)           = min(X,[],2);
    T_median(:,i)        = median(X,2);
    T_rms(:,i)           = rms(X,2);
    T_rssq(:,i)          = rssq(X,2);
    T_absmax(:,i)        = max(abs(X),[],2);
    T_geomean(:,i)       = geomean(abs(X),2);
    T_absharmmean(:,i)   = harmmean(abs(X),2);
    T_harmmean(:,i)      = harmmean(X,2);
    for j = 1:size(X,1)
        %T_entropy(j,i)   = bandpower(X(j,:));
        T_entropy(j,i)   = wentropy(X(j,:),'shannon');
        T_bandpower(j,i) = bandpower(X(j,:));
    end
end

        
% Calculate features (left and right combined)
for i=1:11
    
    % Calculate features upper extremity
    X_upper = table2array(vertcat(table_ML_total_upper{:,i+1}));
    X_fft_upper = fft(table2array(vertcat(table_ML_total_upper{:,i+1})));

    T_max_upper(:,i)              = max(X_upper,[],2);
    T_min_upper(:,i)              = min(X_upper,[],2);
    T_median_upper(:,i)           = median(X_upper,2);
    T_rms_upper(:,i)              = rms(X_upper,2);
    T_rssq_upper(:,i)             = rssq(X_upper,2);
    T_absmax_upper(:,i)           = max(abs(X_upper),[],2);
    T_geomean_upper(:,i)          = geomean(abs(X_upper),2);
    T_absharmmean_upper(:,i)      = harmmean(abs(X_upper),2);
    T_harmmean_upper(:,i)         = harmmean(X_upper,2);
    for j = 1:size(X_upper,1)
        %T_entropy_upper(j,i)      = bandpower(X_upper(j,:));
        T_entropy_upper(j,i)      = wentropy(X_upper(j,:),'shannon');
        T_bandpower_upper(j,i)    = bandpower(X_upper(j,:));
    end
        
    % Calculate features lower extremity
    X_lower = table2array(vertcat(table_ML_total_lower{:,i+1}));
    X_fft_lower = fft(table2array(vertcat(table_ML_total_lower{:,i+1})));

    T_max_lower(:,i)              = max(X_lower,[],2);
    T_min_lower(:,i)              = min(X_lower,[],2);
    T_median_lower(:,i)           = median(X_lower,2);
    T_rms_lower(:,i)              = rms(X_lower,2);
    T_rssq_lower(:,i)             = rssq(X_lower,2);
    T_absmax_lower(:,i)           = max(abs(X_lower),[],2);
    T_geomean_lower(:,i)          = geomean(abs(X_lower),2);
    T_absharmmean_lower(:,i)      = harmmean(abs(X_lower),2);
    T_harmmean_lower(:,i)         = harmmean(X_lower,2);
    for j = 1:size(X_lower,1)
        %T_entropy_lower(j,i)      = bandpower(X_lower(j,:));
        T_entropy_lower(j,i)      = wentropy(X_lower(j,:),'shannon');
        T_bandpower_lower(j,i)    = bandpower(X_lower(j,:));
    end
end
    
    
% Rescale (0-1 scale)
R       = normalize([T_max T_min T_absmax T_median T_rms T_rssq T_geomean T_absharmmean T_entropy T_bandpower],'range'); 
R_upper = normalize([T_max_upper T_min_upper T_absmax_upper T_median_upper T_rms_upper T_rssq_upper T_geomean_upper T_absharmmean_upper T_entropy_upper T_bandpower_upper],'range'); 
R_lower = normalize([T_max_lower T_min_lower T_absmax_lower T_median_lower T_rms_lower T_rssq_lower T_geomean_lower T_absharmmean_lower T_entropy_lower T_bandpower_lower],'range'); 

dataframeTrain_all   = array2table(R);          % dataframe with features calculated for all four extremities
dataframeTrain_upper = array2table(R_upper);    % dataframe with features calculcated for the upper extremity data
dataframeTrain_lower = array2table(R_lower);    % dataframe with features calculcated for the lower extremity data

% Add column names to the dataframes
Names = table_ML_total.Properties.VariableNames(2:45);
Names_upper = table_ML_total_upper.Properties.VariableNames(2:12);
Names_lower = table_ML_total_lower.Properties.VariableNames(2:12);
dataframeTrain_all.Properties.VariableNames   = [string('max_')+Names string('min_')+Names string('absmax_')+Names string('median_')+Names string('rms_')+Names string('rssq_')+Names string('geomean_')+Names string('absharmmean_')+Names string('entropy_')+Names string('bandpower_')+Names];
dataframeTrain_upper.Properties.VariableNames = [string('max_')+Names_upper string('min_')+Names_upper string('absmax_')+Names_upper string('median_')+Names_upper string('rms_')+Names_upper string('rssq_')+Names_upper string('geomean_')+Names_upper string('absharmmean_')+Names_upper string('entropy_')+Names_upper string('bandpower_')+Names_upper];
dataframeTrain_lower.Properties.VariableNames = [string('max_')+Names_lower string('min_')+Names_lower string('absmax_')+Names_lower string('median_')+Names_lower string('rms_')+Names_lower string('rssq_')+Names_lower string('geomean_')+Names_lower string('absharmmean_')+Names_lower string('entropy_')+Names_lower string('bandpower_')+Names_lower];        
dataframeTrain_all = [table(class) dataframeTrain_all];

% Column names per sensor
Names_S1 = table_ML_total.Properties.VariableNames(2:12);   % Left ankle column names
Names_S2 = table_ML_total.Properties.VariableNames(13:23);  % Right ankle column names
Names_S3 = table_ML_total.Properties.VariableNames(24:34);  % Left wrist column names
Names_S4 = table_ML_total.Properties.VariableNames(35:45);  % Right wrist column names
        
% Create dataframe dystonia upper extremities combined
dys_upper_left = Clinical_Scores_TOTAL_median(:,4);
dys_upper_left.Properties.VariableNames{1} = 'dystonia_upper_extemity';
dys_upper_right = Clinical_Scores_TOTAL_median(:,3);
dys_upper_right.Properties.VariableNames{1} = 'dystonia_upper_extemity';        
dataframeTrain_dystonia_upper = [dataframeTrain_upper [dys_upper_left;dys_upper_right]];
        
% Create dataframe dystonia lower extremities combined
dys_lower_left = Clinical_Scores_TOTAL_median(:,2);
dys_lower_left.Properties.VariableNames{1} = 'dystonia_lower_extemity';
dys_lower_right = Clinical_Scores_TOTAL_median(:,1);
dys_lower_right.Properties.VariableNames{1} = 'dystonia_lower_extemity';        
dataframeTrain_dystonia_lower = [dataframeTrain_lower [dys_lower_left;dys_lower_right]];
        
% Create dataframe choreoathetosis upper extremities combined
cho_upper_left = Clinical_Scores_TOTAL_median(:,10);
cho_upper_left.Properties.VariableNames{1} = 'choreoathetosis_upper_extemity';
cho_upper_right = Clinical_Scores_TOTAL_median(:,9);
cho_upper_right.Properties.VariableNames{1} = 'choreoathetosis_upper_extemity';        
dataframeTrain_choreoathetosis_upper = [dataframeTrain_upper [cho_upper_left;cho_upper_right]];
        
% Create dataframe choreoathetosis lower extremities combined
cho_lower_left = Clinical_Scores_TOTAL_median(:,8);
cho_lower_left.Properties.VariableNames{1} = 'choreoathetosis_lower_extemity';
cho_lower_right = Clinical_Scores_TOTAL_median(:,7);
cho_lower_right.Properties.VariableNames{1} = 'choreoathetosis_lower_extemity';        
dataframeTrain_choreoathetosis_lower = [dataframeTrain_lower [cho_lower_left;cho_lower_right]];        
        
% Create dataframe dystonia total
dystonia_total = round(sum(table2array(Clinical_Scores_TOTAL_median  (:,1:6)),2)/6);    % average score of the 6 dystinia scores
dataframeTrain_dystonia_total = [dataframeTrain_all(:,2:end) table(dystonia_total)];

% Create dataframe choreoathetosis total
choreoathetosis_total = round(sum(table2array(Clinical_Scores_TOTAL_median  (:,7:12)),2)/6);
dataframeTrain_choreoathetosis_total = [dataframeTrain_all(:,2:end) table(choreoathetosis_total)];

dataframes = {dataframeTrain_dystonia_lower ; dataframeTrain_dystonia_upper ; dataframeTrain_choreoathetosis_lower ; dataframeTrain_choreoathetosis_upper ; dataframeTrain_dystonia_total ; dataframeTrain_choreoathetosis_total};

end