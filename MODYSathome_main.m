clear all
clc

% Choose for which extremities a ML model should be trained
dystonia_lower          =   "yes";
dystonia_upper          =   "yes";
choreoathetosis_lower   =   "yes";
choreoathetosis_upper   =   "yes";
dystonia_total          =   "no";
choreoathetosis_total   =   "no";

% Choose which type of model to train
machine_learning        =   "yes";
deep_learning           =   "yes";

% Oversampling
oversampling            =   "yes";

% Load the preprocessed data
disp('Loading data...')
DIR = pwd;
Patient = char(DIR(end-9:end));                            
load_name = "load_preprocessed_data_" + string(Patient) + ".mat";
load(load_name)
disp('Complete') ; disp(' ')
   
% Create dataframes for machine learning & deep learning. (function_create_dataframes.m)
disp('Creating dataframes for machine learning...')
[dataframes, dataframeTrain_dystonia_upper , dataframeTrain_dystonia_lower , dataframeTrain_choreoathetosis_upper , dataframeTrain_choreoathetosis_lower , dataframeTrain_dystonia_total , dataframeTrain_choreoathetosis_total] = function_create_dataframes(TABLE_ML_TOTAL, TABLE_ML_TOTAL_upper, TABLE_ML_TOTAL_lower, Class, Clinical_Scores_TOTAL_median);
disp('Complete') ; disp(' ')

% Determine which extremities to model. (function_index_extremities.m)
disp('Determining which extremities are being modelled...')
extremity_idx = function_index_extremities(dystonia_lower, dystonia_upper, choreoathetosis_lower, choreoathetosis_upper, dystonia_total, choreoathetosis_total, dataframes);
disp('Complete') ; disp(' ')



%%% MACHINE LEARNING

if machine_learning == "yes"

for ext_idx = extremity_idx

names = [repmat({"max_Acc_X"},4,1) ; repmat({"max_AL_Acc_X"},2,1)];
    
D = dataframes{ext_idx,1};                               % Picks the dataframe belonging to the extremity index the code is looping over
nan = isnan(D{:,end});                                   % Finds the NaN values in the clinical scores
D = D(~nan,:);                                           % Remove the rows with NaN values as a clinical score
D.Properties.VariableNames{size(D,2)} = 'class';
D = movevars(D, 'class', 'Before', names{ext_idx,1});    % Puts the clinical score ('class') as the first column
D.class = categorical(D.class);                          % Transforms the clinical score to a categorical array


% Define input (X), output (y) and whole dataframe (D) for training and testing data 
for k = 1:5
    
    D_train{k}     =  D(trainInd{k,ext_idx},:);
    D_test{k}      =  D(testInd{k,ext_idx},:);
    X_train{k}     =  D{trainInd{k,ext_idx},2:end};
    X_test{k}      =  D{testInd{k,ext_idx},2:end};
    y_train{k}     =  D.class(trainInd{k,ext_idx},:);
    y_test{k}      =  D.class(testInd{k,ext_idx},:);
    size_y_test(k) =  size(y_test{k},1);

    if oversampling == "yes"
        % Oversample training data. This piece of code oversamples classes with
        % a low number of samples, making sure all classes will have an equal amount of
        % samples in the training set. (function_oversample.m)
        [X_train, y_train, D_train] = function_oversample(X_train, y_train, D_train,k);
    end
end


% Training the machine learning models and determining the F1 scores and accuracies. (function_machine_learning.m)
[results_ML_features, results_ML_accuracy_val_total, results_ML_accuracy_test_total, results_ML_recall_val_total, results_ML_recall_test_total, results_ML_precision_val_total, results_ML_precision_test_total, results_ML_F1_val_total, results_ML_F1_test_total, toKeep_save, model_ALL, model_ALL_HYP, model_SFS, model_SFS_HYP] = function_machine_learning(ext_idx, D_train, X_train, y_train, D_test, X_test, y_test)

% Determine best Machine Learning model (function_determine_best_ML_model.m)
disp('Determining the best machine learning model...')
[best_ML_model, best_ML_model_type, best_ML_model_specs, best_ML_model_accuracy_val, best_ML_model_accuracy_test, best_ML_model_recall_val, best_ML_model_precision_val, best_ML_model_recall_test, best_ML_model_precision_test, best_ML_model_F1_val, best_ML_model_F1_test]  =   function_determine_best_ML_model(model_ALL, model_ALL_HYP, model_SFS, model_SFS_HYP, results_ML_accuracy_val_total, results_ML_accuracy_test_total, results_ML_F1_val_total, results_ML_F1_test_total, results_ML_recall_val_total, results_ML_precision_val_total, results_ML_recall_test_total, results_ML_precision_test_total, ext_idx, size_y_test)
disp('Complete') ; disp(' ')
disp("The best ML model is a " + string(best_ML_model_type{1,ext_idx}) + " model (" + string(best_ML_model_specs{1,ext_idx})  + ") with an F1 score of " + string(best_ML_model_F1_val{1,ext_idx}) + " and a validation accuracy of " + string(best_ML_model_accuracy_val{1,ext_idx}) + "%")
disp("The test accuracy of this model is " + string(best_ML_model_accuracy_test{1,ext_idx}) + "%")

% Save in workspace
for i = 1:6
    results_ML_accuracy_val_total_save{i,ext_idx}     = results_ML_accuracy_val_total{i,ext_idx};
    results_ML_accuracy_test_total_save{i,ext_idx}    = results_ML_accuracy_test_total{i,ext_idx};
    results_ML_F1_val_total_save{i,ext_idx}           = results_ML_F1_val_total{i,ext_idx};
    results_ML_F1_test_total_save{i,ext_idx}          = results_ML_F1_test_total{i,ext_idx};
    results_features{i,ext_idx}                       = sum(toKeep_save{i,ext_idx});
end
best_ML_model_save{1,ext_idx}                = best_ML_model{1,ext_idx};
best_ML_model_type_save{1,ext_idx}           = best_ML_model_type{1,ext_idx};
best_ML_model_specs_save{1,ext_idx}          = best_ML_model_specs{1,ext_idx};
best_ML_model_accuracy_val_save{1,ext_idx}   = best_ML_model_accuracy_val{1,ext_idx};
best_ML_model_accuracy_test_save{1,ext_idx}  = best_ML_model_accuracy_test{1,ext_idx};
best_ML_model_recall_val_save{1,ext_idx}     = best_ML_model_recall_val{1,ext_idx};
best_ML_model_precision_val_save{1,ext_idx}  = best_ML_model_precision_val{1,ext_idx};
best_ML_model_recall_test_save{1,ext_idx}    = best_ML_model_recall_test{1,ext_idx};
best_ML_model_precision_test_save{1,ext_idx} = best_ML_model_precision_test{1,ext_idx};
best_ML_model_F1_val_save{1,ext_idx}         = best_ML_model_F1_val{1,ext_idx};
best_ML_model_F1_test_save{1,ext_idx}        = best_ML_model_F1_test{1,ext_idx};

end

% Display and save the results. (function_display_ML_results.m)
[TABLES_ML_accuracy, TABLES_ML_F1] = function_display_ML_results(results_ML_accuracy_val_total_save, results_ML_accuracy_test_total_save, results_features, results_ML_F1_val_total_save, results_ML_F1_test_total_save);

end



%%% DEEP LEARNING

if deep_learning == "yes"
    
for ext_idx = extremity_idx     % Run the deep learning code for each dataset

clear x class F1_val_kfold F1_test_kfold accuracy_val_kfold accuracy_test_kfold net_save optimal_hiddenlayers_val

% Prepare data of the considered extremity for deep learning. (function_prepare_DL_data.m)
[X, class]  = function_prepare_DL_data(ext_idx, TABLE_ML_TOTAL, TABLE_ML_TOTAL_upper, TABLE_ML_TOTAL_lower, Clinical_Scores_TOTAL_median);

% Train the deep learning models and determine the F1 score and accuracy. (function_deep_learning.m)
[F1_val_kfold, F1_test_kfold, accuracy_val_kfold, accuracy_test_kfold, recall_val_mean_kfold, precision_val_mean_kfold, recall_test_mean_kfold, precision_test_mean_kfold,  net_save, optimal_hiddenlayers_val]   = function_deep_learning(ext_idx, X, class, trainInd_DL, valInd_DL, testInd_DL)

% Calculate the results of the 5 folds combined
best_DL_model_F1_val{1,ext_idx}          = (F1_val_kfold(1)*size_y_val_DL(1,ext_idx)   + F1_val_kfold(2)*size_y_val_DL(2,ext_idx)   + F1_val_kfold(3)*size_y_val_DL(3,ext_idx)   + F1_val_kfold(4)*size_y_val_DL(4,ext_idx)   + F1_val_kfold(5)*size_y_val_DL(5,ext_idx))   / sum(size_y_val_DL(:,ext_idx))
best_DL_model_F1_test{1,ext_idx}         = (F1_test_kfold(1)*size_y_test_DL(1,ext_idx) + F1_test_kfold(2)*size_y_test_DL(2,ext_idx) + F1_test_kfold(3)*size_y_test_DL(3,ext_idx) + F1_test_kfold(4)*size_y_test_DL(4,ext_idx) + F1_test_kfold(5)*size_y_test_DL(5,ext_idx)) / sum(size_y_test_DL(:,ext_idx))
best_DL_model_accuracy_val{1,ext_idx}    = (accuracy_val_kfold(1)*size_y_val_DL(1,ext_idx)   + accuracy_val_kfold(2)*size_y_val_DL(2,ext_idx)   + accuracy_val_kfold(3)*size_y_val_DL(3,ext_idx)   + accuracy_val_kfold(4)*size_y_val_DL(4,ext_idx)   + accuracy_val_kfold(5)*size_y_val_DL(5,ext_idx))   / sum(size_y_val_DL(:,ext_idx))
best_DL_model_accuracy_test{1,ext_idx}   = (accuracy_test_kfold(1)*size_y_test_DL(1,ext_idx) + accuracy_test_kfold(2)*size_y_test_DL(2,ext_idx) + accuracy_test_kfold(3)*size_y_test_DL(3,ext_idx) + accuracy_test_kfold(4)*size_y_test_DL(4,ext_idx) + accuracy_test_kfold(5)*size_y_test_DL(5,ext_idx)) / sum(size_y_test_DL(:,ext_idx))
best_DL_model_recall_val{1,ext_idx}      = (recall_val_mean_kfold(1)*size_y_val_DL(1,ext_idx)  +  recall_val_mean_kfold(2)*size_y_val_DL(2,ext_idx)  +  recall_val_mean_kfold(3)*size_y_val_DL(3,ext_idx)  +  recall_val_mean_kfold(4)*size_y_val_DL(4,ext_idx)  +  recall_val_mean_kfold(5)*size_y_val_DL(5,ext_idx))  / sum(size_y_val_DL(:,ext_idx))
best_DL_model_precision_val{1,ext_idx}   = (precision_val_mean_kfold(1)*size_y_val_DL(1,ext_idx)  +  precision_val_mean_kfold(2)*size_y_val_DL(2,ext_idx)  +  precision_val_mean_kfold(3)*size_y_val_DL(3,ext_idx)  +  precision_val_mean_kfold(4)*size_y_val_DL(4,ext_idx)  +  precision_val_mean_kfold(5)*size_y_val_DL(5,ext_idx))  / sum(size_y_val_DL(:,ext_idx))
best_DL_model_recall_test{1,ext_idx}     = (recall_test_mean_kfold(1)*size_y_test_DL(1,ext_idx)  +  recall_test_mean_kfold(2)*size_y_test_DL(2,ext_idx)  +  recall_test_mean_kfold(3)*size_y_test_DL(3,ext_idx)  +  recall_test_mean_kfold(4)*size_y_test_DL(4,ext_idx)  +  recall_test_mean_kfold(5)*size_y_test_DL(5,ext_idx))  / sum(size_y_test_DL(:,ext_idx))
best_DL_model_precision_test{1,ext_idx}  = (precision_test_mean_kfold(1)*size_y_test_DL(1,ext_idx)  +  precision_test_mean_kfold(2)*size_y_test_DL(2,ext_idx)  +  precision_test_mean_kfold(3)*size_y_test_DL(3,ext_idx)  +  precision_test_mean_kfold(4)*size_y_test_DL(4,ext_idx)  +  precision_test_mean_kfold(5)*size_y_test_DL(5,ext_idx))  / sum(size_y_test_DL(:,ext_idx))
best_DL_model_hidden_layers{1,ext_idx}   = optimal_hiddenlayers_val{1,ext_idx}(1);
best_DL_model{1,ext_idx}                 = net_save;

end

% Display the Deep Learning results. (function_display_DL_results.m)
[TABLES_DL]    = function_display_DL_results(extremity_idx, best_DL_model_hidden_layers, best_DL_model_accuracy_val, best_DL_model_accuracy_test, best_DL_model_F1_val, best_DL_model_F1_test, dataframeTrain_dystonia_lower, dataframeTrain_dystonia_upper, dataframeTrain_choreoathetosis_lower, dataframeTrain_choreoathetosis_upper, dataframeTrain_dystonia_total, dataframeTrain_choreoathetosis_total); 

end



%%% Pick the best models (ML vs DL)

if machine_learning == "yes" && deep_learning == "yes"

% Pick the best models. (function_pick_best_models.m)
[table_best_models_performance, best_model_dys_lower, best_model_dys_upper, best_model_cho_lower, best_model_cho_upper, best_model_dys_total, best_model_cho_total]     = function_pick_best_models(extremity_idx, best_ML_model_save, best_DL_model, best_ML_model_F1_val_save, best_DL_model_F1_val, best_ML_model_F1_test_save, best_DL_model_F1_test, best_ML_model_accuracy_val_save, best_DL_model_accuracy_val, best_ML_model_accuracy_test_save, best_DL_model_accuracy_test, best_ML_model_recall_val_save, best_DL_model_recall_val, best_ML_model_precision_val_save, best_DL_model_precision_val, best_ML_model_recall_test_save, best_DL_model_recall_test, best_ML_model_precision_test_save, best_DL_model_precision_test)

% Save the models & results
DIR_save = string(DIR(1:end-15)) + "results\" + Patient + "\models_results_" + string(Patient);
save(DIR_save, 'best_model_dys_lower','best_model_dys_upper','best_model_cho_lower','best_model_cho_upper','best_model_dys_total','best_model_cho_total','toKeep_save','table_best_models_performance','TABLES_ML_accuracy','TABLES_ML_F1','TABLES_DL')
save(Patient)

table_best_models_performance

end