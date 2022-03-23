function [results_ML_features, results_ML_accuracy_val_total, results_ML_accuracy_test_total, results_ML_recall_val_total, results_ML_recall_test_total, results_ML_precision_val_total, results_ML_precision_test_total, results_ML_F1_val_total, results_ML_F1_test_total, toKeep_save, model_ALL, model_ALL_HYP, model_SFS, model_SFS_HYP] = function_machine_learning(ext_idx, D_train, X_train, y_train, D_test, X_test, y_test)

for k = 1:5
    
% Define data partition settings (5-fold cross validation) and maximum objective evaluations (20)
cvpt = cvpartition(y_train{k},"KFold",5);
opt  = struct("CVPartition",cvpt,"MaxObjectiveEvaluations",15,"ShowPlots",false);
    
% Train machine learning models (KNN, DT, SVM, NB, DA, Ensemble) with all
% features, without hyperparameter optimization
model_ALL  = {fitcknn(D_train{k},"class","CVPartition",cvpt) fitctree(D_train{k},"class","CVPartition",cvpt) fitcecoc(D_train{k},"class","CVPartition",cvpt) fitcnb(D_train{k},"class","CVPartition",cvpt) fitcdiscr(D_train{k},"class","CVPartition",cvpt) fitcensemble(D_train{k},"class","CVPartition",cvpt)};
model_ALL_kfold{k} = model_ALL;

if k == 1
    % Train machine learning models (KNN, DT, SVM, NB, DA, Ensemble) with all
    % features, with hyperparameter optimization
    model_knn_ALL_HYP1      = fitcknn(X_train{k},y_train{k},"OptimizeHyperparameters","auto","HyperparameterOptimizationOptions",opt)
    model_tree_ALL_HYP1     = fitctree(X_train{k},y_train{k},"OptimizeHyperparameters","auto","HyperparameterOptimizationOptions",opt)
    model_svm_ALL_HYP1      = fitcecoc(X_train{k},y_train{k},"OptimizeHyperparameters","auto","HyperparameterOptimizationOptions",opt)
    model_nb_ALL_HYP1       = fitcnb(X_train{k},y_train{k},"OptimizeHyperparameters","auto","HyperparameterOptimizationOptions",opt)
    model_discr_ALL_HYP1    = fitcdiscr(X_train{k},y_train{k},"OptimizeHyperparameters","auto","HyperparameterOptimizationOptions",opt)
    model_ensemble_ALL_HYP1 = fitcensemble(X_train{k},y_train{k},"Method","RUSBoost","OptimizeHyperparameters",["NumLearningCycles" "LearnRate" "MinLeafSize"],"HyperparameterOptimizationOptions",opt)
    
    model_ALL_HYP1 = {model_knn_ALL_HYP1 model_tree_ALL_HYP1  model_svm_ALL_HYP1  model_nb_ALL_HYP1  model_discr_ALL_HYP1  model_ensemble_ALL_HYP1};
    model_ALL_HYP  = model_ALL_HYP1;
    
    % Save the hyperparameters
    for p = 1:6
        hyperparameters_ALL_HYP{p} = model_ALL_HYP1{1,p}.HyperparameterOptimizationResults.XAtMinEstimatedObjective  
    end
else
    model_knn_ALL_HYP2      = fitcknn(D_train{k},"class","CVPartition",cvpt,"NumNeighbors",hyperparameters_ALL_HYP{1}.NumNeighbors,'Distance',string(hyperparameters_ALL_HYP{1}.Distance))
    model_tree_ALL_HYP2     = fitctree(D_train{k},"class","CVPartition",cvpt,"MinLeafSize",hyperparameters_ALL_HYP{2}.MinLeafSize)
    model_svm_ALL_HYP2      = fitcecoc(D_train{k},"class","CVPartition",cvpt,"Coding",string(hyperparameters_ALL_HYP{3}.Coding),"Learners",templateSVM("BoxConstraint",hyperparameters_ALL_HYP{3}.BoxConstraint,"KernelScale",hyperparameters_ALL_HYP{3}.KernelScale)) 
    model_nb_ALL_HYP2       = fitcnb(D_train{k},"class","CVPartition",cvpt,"DistributionNames",string(hyperparameters_ALL_HYP{4}.DistributionNames),"Width",hyperparameters_ALL_HYP{4}.Width)
    model_discr_ALL_HYP2    = fitcdiscr(D_train{k},"class","CVPartition",cvpt,"Delta",hyperparameters_ALL_HYP{5}.Delta,"Gamma",hyperparameters_ALL_HYP{5}.Gamma)
    model_ensemble_ALL_HYP2 = fitcensemble(D_train{k},"class","CVPartition",cvpt,"Method","RUSBoost","NumLearningCycles",hyperparameters_ALL_HYP{6}.NumLearningCycles, "LearnRate" , hyperparameters_ALL_HYP{6}.LearnRate, "Learners", templateTree("MinLeafSize", hyperparameters_ALL_HYP{6}.MinLeafSize))
    
    model_ALL_HYP2 = {model_knn_ALL_HYP2  model_tree_ALL_HYP2  model_svm_ALL_HYP2  model_nb_ALL_HYP2  model_discr_ALL_HYP2  model_ensemble_ALL_HYP2};
    model_ALL_HYP = model_ALL_HYP2;
end
    model_ALL_HYP_kfold{k} = model_ALL_HYP;

    
% Perform forward Sequential feature selection for KNN, DT, SVM and DA
% NB and Ensemble are excluded due to high computation costs
if k == 1
    X_train_temp = X_train{k};
    y_train_temp = y_train{k};
    fmodel_all = {@(X_train_temp,y_train_temp) fitcknn(X_train_temp,y_train_temp) ; @(X_train_temp,y_train_temp) fitctree(X_train_temp,y_train_temp) ; @(X_train_temp,y_train_temp) fitcecoc(X_train_temp,y_train_temp) ; @(X_train_temp,y_train_temp) fitcnb(X_train_temp,y_train_temp) ; @(X_train_temp,y_train_temp) fitcdiscr(X_train_temp,y_train_temp) ; @(X_train_temp,y_train_temp) fitcecoc(X_train_temp,y_train_temp)}
    for j = 1:6 % indices of KNN, DT, SVM and DA, respectively
        fmodel = fmodel_all{j};
        ferror = @(Xtrain,ytrain,Xtest,ytest) nnz(predict(fmodel(Xtrain,ytrain),Xtest) ~= ytest);
        toKeep = sequentialfs(ferror,X_train_temp,y_train_temp,"cv",cvpt,"options",statset("Display","iter"));
        toKeep_save{j,ext_idx} = toKeep;     
        idx_toKeep = find(toKeep_save{j,ext_idx}==1);   % indices of the kept features
    end
    clear X_train_temp y_train_temp
end

for C = 1:6
    
    True_scores_test                           = categorical(D_test{k}.class);
    True_scores_test_save{C,ext_idx,k}         = True_scores_test;
    True_scores_test_ALL_HYP                   = categorical(D_test{k}.class);
    True_scores_test_ALL_HYP_save{C,ext_idx,k} = True_scores_test_ALL_HYP;
    True_scores_test_SFS_HYP                   = categorical(D_test{k}.class);
    True_scores_test_SFS_HYP_save{C,ext_idx,k} = True_scores_test_SFS_HYP;
    True_scores_val                            = categorical(D_train{k}.class);
    True_scores_val_save{C,ext_idx,k}          = True_scores_val;
    True_scores_val_ALL_HYP                    = categorical(D_train{k}.class);
    True_scores_val_ALL_HYP_save{C,ext_idx,k}  = True_scores_val_ALL_HYP;
    True_scores_val_SFS_HYP                    = categorical(D_train{k}.class);
    True_scores_val_SFS_HYP_save{C,ext_idx,k}  = True_scores_val_SFS_HYP;
    
    % Predict scores with models (ALL & ALL+HYP) 
    Predicted_scores_ALL_test                   = predict(model_ALL{C}.Trained{end, 1} , table2array(D_test{k}(:,2:end)));
    Predicted_scores_ALL_test_save{C,ext_idx,k} = Predicted_scores_ALL_test;
    Predicted_scores_ALL_val                    = kfoldPredict(model_ALL{C});
    Predicted_scores_ALL_val_save{C,ext_idx,k}  = Predicted_scores_ALL_val;

    if k == 1
        Predicted_scores_ALL_HYP_test   = predict(model_ALL_HYP{C} , table2array(D_test{k}(:,2:end))); 
        Predicted_scores_ALL_HYP_val    = kfoldPredict(crossval(model_ALL_HYP{C},'kfold',5));
    else
        if size(model_ALL_HYP2{C}.Trained,1) ~= 0
        Predicted_scores_ALL_HYP_test   = predict(model_ALL_HYP{C}.Trained{end, 1} , table2array(D_test{k}(:,2:end)));
        Predicted_scores_ALL_HYP_val    = kfoldPredict(model_ALL_HYP{C});
        else
            Predicted_scores_ALL_HYP_test = [];
            Predicted_scores_ALL_HYP_val  = [];
            
            True_scores_test_ALL_HYP                   = [];
            True_scores_test_ALL_HYP_save{C,ext_idx,k} = True_scores_test_ALL_HYP;
            True_scores_val_ALL_HYP                    = [];
            True_scores_val_ALL_HYP_save{C,ext_idx,k}   = True_scores_val_ALL_HYP;
        end
    end
    % Save
    Predicted_scores_ALL_HYP_test_save{C,ext_idx,k} = Predicted_scores_ALL_HYP_test;
    Predicted_scores_ALL_HYP_val_save{C,ext_idx,k} = Predicted_scores_ALL_HYP_val;

    
    if k == 1
            clear toKeep_idx D_test_SFS
            toKeep_idx = [logical(0) toKeep_save{C,ext_idx}];
            features_kept = D_train{k}.Properties.VariableNames(toKeep_idx)
            features_kept2{C,ext_idx} = features_kept;
    end
    
    % Models with Sequential feature selection        
        clear D_test_SFS
        
        % Test on holdout data (SFS)
        D_test_SFS = D_test{k}(:,[logical(0) toKeep_save{C,ext_idx}]);
        D_test_SFS.class = D_test{k}.class;
    
        % Fit models (except naïve bayes & ensemble) with just the given variables
        model_SFS = {fitcknn(X_train{k}(:,toKeep_save{1,ext_idx}),y_train{k},"CVPartition",cvpt)  fitctree(X_train{k}(:,toKeep_save{2,ext_idx}),y_train{k},"CVPartition",cvpt)  fitcecoc(X_train{k}(:,toKeep_save{3,ext_idx}),y_train{k},"CVPartition",cvpt)  fitcnb(X_train{k}(:,toKeep_save{4,ext_idx}),y_train{k},"CVPartition",cvpt)  fitcdiscr(X_train{k}(:,toKeep_save{5,ext_idx}),y_train{k},"CVPartition",cvpt)  fitcensemble(X_train{k}(:,toKeep_save{6,ext_idx}),y_train{k},"CVPartition",cvpt)};
        model_SFS_kfold{k} = model_SFS;
           
    if k == 1
    % Train machine learning models (KNN, DT, SVM, NB, DA, Ensemble) with all
    % features, with hyperparameter optimization
    
        model_knn_SFS_HYP1      = fitcknn(X_train{k}(:,toKeep_save{1,ext_idx}),y_train{k},"OptimizeHyperparameters","auto","HyperparameterOptimizationOptions",opt)
        model_tree_SFS_HYP1     = fitctree(X_train{k}(:,toKeep_save{2,ext_idx}),y_train{k},"OptimizeHyperparameters","auto","HyperparameterOptimizationOptions",opt)
        model_svm_SFS_HYP1      = fitcecoc(X_train{k}(:,toKeep_save{3,ext_idx}),y_train{k},"OptimizeHyperparameters","auto","HyperparameterOptimizationOptions",opt)
        model_nb_SFS_HYP1       = fitcnb(X_train{k}(:,toKeep_save{4,ext_idx}),y_train{k},"OptimizeHyperparameters","auto","HyperparameterOptimizationOptions",opt)
        model_discr_SFS_HYP1    = fitcdiscr(X_train{k}(:,toKeep_save{5,ext_idx}),y_train{k},"OptimizeHyperparameters","auto","HyperparameterOptimizationOptions",opt)
        model_ensemble_SFS_HYP1 = fitcensemble(X_train{k}(:,toKeep_save{6,ext_idx}),y_train{k},"Method","RUSBoost","OptimizeHyperparameters",["NumLearningCycles" "LearnRate" "MinLeafSize"],"HyperparameterOptimizationOptions",opt)

        model_SFS_HYP1 = {model_knn_SFS_HYP1  model_tree_SFS_HYP1  model_svm_SFS_HYP1  model_nb_SFS_HYP1  model_discr_SFS_HYP1  model_ensemble_SFS_HYP1};
        model_SFS_HYP  = model_SFS_HYP1;

        Predicted_scores_SFS_HYP_test   = predict(model_SFS_HYP{C} ,table2array(D_test_SFS(:,1:end-1))); 
        Predicted_scores_SFS_HYP_val    = kfoldPredict(crossval(model_SFS_HYP{C},'kfold',5));
        
    % Save the hyperparameters
    for p = 1:6
        hyperparameters_SFS_HYP{p} = model_SFS_HYP{1,p}.HyperparameterOptimizationResults.XAtMinEstimatedObjective ; 
    end
    else
        model_knn_SFS_HYP2      =  fitcknn(X_train{k}(:,toKeep_save{1,ext_idx}),y_train{k},"CVPartition",cvpt,"NumNeighbors",hyperparameters_SFS_HYP{1}.NumNeighbors,'Distance',string(hyperparameters_SFS_HYP{1}.Distance))
        model_tree_SFS_HYP2     =  fitctree(X_train{k}(:,toKeep_save{2,ext_idx}),y_train{k},"CVPartition",cvpt,"MinLeafSize",hyperparameters_SFS_HYP{2}.MinLeafSize)
        model_svm_SFS_HYP2      =  fitcecoc(X_train{k}(:,toKeep_save{3,ext_idx}),y_train{k},"CVPartition",cvpt,"Coding",string(hyperparameters_SFS_HYP{3}.Coding),"Learners",templateSVM("BoxConstraint",hyperparameters_SFS_HYP{3}.BoxConstraint,"KernelScale",hyperparameters_SFS_HYP{3}.KernelScale))
        model_nb_SFS_HYP2       =  fitcnb(X_train{k}(:,toKeep_save{4,ext_idx}),y_train{k},"CVPartition",cvpt,"DistributionNames",string(hyperparameters_SFS_HYP{4}.DistributionNames),"Width",hyperparameters_SFS_HYP{4}.Width)
        model_discr_SFS_HYP2    =  fitcdiscr(X_train{k}(:,toKeep_save{5,ext_idx}),y_train{k},"CVPartition",cvpt,"Delta",hyperparameters_SFS_HYP{5}.Delta,"Gamma",hyperparameters_SFS_HYP{5}.Gamma)
        model_ensemble_SFS_HYP2 =  fitcensemble(X_train{k}(:,toKeep_save{6,ext_idx}),y_train{k},"CVPartition",cvpt,"Method","RUSBoost","NumLearningCycles",hyperparameters_SFS_HYP{6}.NumLearningCycles, "LearnRate" , hyperparameters_SFS_HYP{6}.LearnRate, "Learners", templateTree("MinLeafSize", hyperparameters_SFS_HYP{6}.MinLeafSize))
          
        model_SFS_HYP2 = {model_knn_SFS_HYP2  model_tree_SFS_HYP2  model_svm_SFS_HYP2  model_nb_SFS_HYP2  model_discr_SFS_HYP2  model_ensemble_SFS_HYP2};
        model_SFS_HYP  = model_SFS_HYP2;
        
        if size(model_SFS_HYP2{C}.Trained,1) ~= 0
            Predicted_scores_SFS_HYP_test  = predict(model_SFS_HYP{C}.Trained{end, 1},table2array(D_test_SFS(:,1:end-1)));
            Predicted_scores_SFS_HYP_val   = kfoldPredict(model_SFS_HYP{C});
            
            True_scores_test_SFS_HYP                   = categorical(D_test{k}.class);
            True_scores_test_SFS_HYP_save{C,ext_idx,k} = True_scores_test_SFS_HYP;
            True_scores_val_SFS_HYP                    = categorical(D_train{k}.class);
            True_scores_val_SFS_HYP_save{C,ext_idx,k}  = True_scores_val_SFS_HYP;
        else 
            Predicted_scores_SFS_HYP_test  = [];
            Predicted_scores_SFS_HYP_val   = [];
            
            True_scores_test_SFS_HYP                   = [];
            True_scores_test_SFS_HYP_save{C,ext_idx,k} = True_scores_test_SFS_HYP;
            True_scores_val_SFS_HYP                    = [];
            True_scores_val_SFS_HYP_save{C,ext_idx,k}  = True_scores_val_SFS_HYP;
        end
    end
    model_SFS_HYP_kfold{k} = model_SFS_HYP;
    
        % Predicted 
        Predicted_scores_SFS_test                       = predict(model_SFS{C}.Trained{end, 1}  ,table2array(D_test_SFS(:,1:end-1)));
        Predicted_scores_SFS_test_save{C,ext_idx,k}     = Predicted_scores_SFS_test;
        Predicted_scores_SFS_val                        = kfoldPredict(model_SFS{C});
        Predicted_scores_SFS_val_save{C,ext_idx,k}      = Predicted_scores_SFS_val;
        Predicted_scores_SFS_HYP_test_save{C,ext_idx,k} = Predicted_scores_SFS_HYP_test;
        Predicted_scores_SFS_HYP_val_save{C,ext_idx,k}  = Predicted_scores_SFS_HYP_val;
        
end 

end



% Predicted & true scores of 5-folds combined
for C = 1:6
    Predicted_scores_ALL_val_kfold{C,ext_idx} = [];    
    Predicted_scores_ALL_test_kfold{C,ext_idx} = [];
    Predicted_scores_ALL_HYP_val_kfold{C,ext_idx} = [];
    Predicted_scores_ALL_HYP_test_kfold{C,ext_idx} = [];
    Predicted_scores_SFS_val_kfold{C,ext_idx} = [];    
    Predicted_scores_SFS_test_kfold{C,ext_idx} = [];
    Predicted_scores_SFS_HYP_val_kfold{C,ext_idx} = [];
    Predicted_scores_SFS_HYP_test_kfold{C,ext_idx} = [];
    True_scores_val_kfold{C,ext_idx} = [];
    True_scores_val_ALL_HYP_kfold{C,ext_idx} = [];
    True_scores_val_SFS_HYP_kfold{C,ext_idx} = [];
    True_scores_test_kfold{C,ext_idx} = [];
    True_scores_test_ALL_HYP_kfold{C,ext_idx} = [];
    True_scores_test_SFS_HYP_kfold{C,ext_idx} = [];

    for k = 1:5
        Predicted_scores_ALL_val_kfold{C,ext_idx}      = [Predicted_scores_ALL_val_kfold{C,ext_idx} ; Predicted_scores_ALL_val_save{C,ext_idx,k}];
        Predicted_scores_ALL_test_kfold{C,ext_idx}     = [Predicted_scores_ALL_test_kfold{C,ext_idx} ; Predicted_scores_ALL_test_save{C,ext_idx,k}];
        Predicted_scores_ALL_HYP_val_kfold{C,ext_idx}  = [Predicted_scores_ALL_HYP_val_kfold{C,ext_idx} ; Predicted_scores_ALL_HYP_val_save{C,ext_idx,k}];
        Predicted_scores_ALL_HYP_test_kfold{C,ext_idx} = [Predicted_scores_ALL_HYP_test_kfold{C,ext_idx} ; Predicted_scores_ALL_HYP_test_save{C,ext_idx,k}];
        Predicted_scores_SFS_val_kfold{C,ext_idx}      = [Predicted_scores_SFS_val_kfold{C,ext_idx} ; Predicted_scores_SFS_val_save{C,ext_idx,k}];
        Predicted_scores_SFS_test_kfold{C,ext_idx}     = [Predicted_scores_SFS_test_kfold{C,ext_idx} ; Predicted_scores_SFS_test_save{C,ext_idx,k}];
        Predicted_scores_SFS_HYP_val_kfold{C,ext_idx}  = [Predicted_scores_SFS_HYP_val_kfold{C,ext_idx} ; Predicted_scores_SFS_HYP_val_save{C,ext_idx,k}];
        Predicted_scores_SFS_HYP_test_kfold{C,ext_idx} = [Predicted_scores_SFS_HYP_test_kfold{C,ext_idx} ; Predicted_scores_SFS_HYP_test_save{C,ext_idx,k}];
        True_scores_val_kfold{C,ext_idx}               = [ True_scores_val_kfold{C,ext_idx} ; True_scores_val_save{C,ext_idx,k}];
        True_scores_val_ALL_HYP_kfold{C,ext_idx}       = [ True_scores_val_ALL_HYP_kfold{C,ext_idx} ; True_scores_val_ALL_HYP_save{C,ext_idx,k}];
        True_scores_val_SFS_HYP_kfold{C,ext_idx}       = [ True_scores_val_SFS_HYP_kfold{C,ext_idx} ; True_scores_val_SFS_HYP_save{C,ext_idx,k}];
        True_scores_test_kfold{C,ext_idx}              = [ True_scores_test_kfold{C,ext_idx} ; True_scores_test_save{C,ext_idx,k}];
        True_scores_test_ALL_HYP_kfold{C,ext_idx}      = [ True_scores_test_ALL_HYP_kfold{C,ext_idx} ; True_scores_test_ALL_HYP_save{C,ext_idx,k}];
        True_scores_test_SFS_HYP_kfold{C,ext_idx}      = [ True_scores_test_SFS_HYP_kfold{C,ext_idx} ; True_scores_test_SFS_HYP_save{C,ext_idx,k}];
    end
end

% Calculate metrics
for C = 1:6

% accuracies validation set
accuracy_ALL_val{C,ext_idx}     = round(sum(Predicted_scores_ALL_val_kfold{C,ext_idx}     == True_scores_val_kfold{C,ext_idx})         / numel(True_scores_val_kfold{C,ext_idx}) *100,1);
accuracy_ALL_HYP_val{C,ext_idx} = round(sum(Predicted_scores_ALL_HYP_val_kfold{C,ext_idx} == True_scores_val_ALL_HYP_kfold{C,ext_idx}) / numel(True_scores_val_ALL_HYP_kfold{C,ext_idx}) *100,1);
accuracy_SFS_val{C,ext_idx}     = round(sum(Predicted_scores_SFS_val_kfold{C,ext_idx}     == True_scores_val_kfold{C,ext_idx})         / numel(True_scores_val_kfold{C,ext_idx}) *100,1);
accuracy_SFS_HYP_val{C,ext_idx} = round(sum(Predicted_scores_SFS_HYP_val_kfold{C,ext_idx} == True_scores_val_SFS_HYP_kfold{C,ext_idx}) / numel(True_scores_val_SFS_HYP_kfold{C,ext_idx}) *100,1);

% accuracies test set
accuracy_ALL_test{C,ext_idx}     = round(sum(Predicted_scores_ALL_test_kfold{C,ext_idx}     == True_scores_test_kfold{C,ext_idx})         / numel(True_scores_test_kfold{C,ext_idx}) *100,1);
accuracy_ALL_HYP_test{C,ext_idx} = round(sum(Predicted_scores_ALL_HYP_test_kfold{C,ext_idx} == True_scores_test_ALL_HYP_kfold{C,ext_idx}) / numel(True_scores_test_ALL_HYP_kfold{C,ext_idx}) *100,1);
accuracy_SFS_test{C,ext_idx}     = round(sum(Predicted_scores_SFS_test_kfold{C,ext_idx}     == True_scores_test_kfold{C,ext_idx})         / numel(True_scores_test_kfold{C,ext_idx}) *100,1);
accuracy_SFS_HYP_test{C,ext_idx} = round(sum(Predicted_scores_SFS_HYP_test_kfold{C,ext_idx} == True_scores_test_SFS_HYP_kfold{C,ext_idx}) / numel(True_scores_test_SFS_HYP_kfold{C,ext_idx}) *100,1);

% recall test set
for i = 0:4

% recall validation set
recall_ALL_val{C,ext_idx}(i+1)       =  numel(find(double(string(True_scores_val_kfold{C,ext_idx}))         == i & double(string(Predicted_scores_ALL_val_kfold{C,ext_idx})) == i))      / numel(find(double(string(True_scores_val_kfold{C,ext_idx})) == i));
recall_ALL_HYP_val{C,ext_idx}(i+1)   =  numel(find(double(string(True_scores_val_ALL_HYP_kfold{C,ext_idx})) == i & double(string(Predicted_scores_ALL_HYP_val_kfold{C,ext_idx})) == i))  / numel(find(double(string(True_scores_val_ALL_HYP_kfold{C,ext_idx})) == i));
recall_SFS_val{C,ext_idx}(i+1)       =  numel(find(double(string(True_scores_val_kfold{C,ext_idx}))         == i & double(string(Predicted_scores_SFS_val_kfold{C,ext_idx})) == i))      / numel(find(double(string(True_scores_val_kfold{C,ext_idx})) == i));
recall_SFS_HYP_val{C,ext_idx}(i+1)   =  numel(find(double(string(True_scores_val_SFS_HYP_kfold{C,ext_idx})) == i & double(string(Predicted_scores_SFS_HYP_val_kfold{C,ext_idx})) == i))  / numel(find(double(string(True_scores_val_SFS_HYP_kfold{C,ext_idx})) == i));

% recall test set
recall_ALL_test{C,ext_idx}(i+1)        =  numel(find(double(string(True_scores_test_kfold{C,ext_idx}))         == i & double(string(Predicted_scores_ALL_test_kfold{C,ext_idx})) == i))      / numel(find(double(string(True_scores_test_kfold{C,ext_idx})) == i));
recall_ALL_HYP_test{C,ext_idx}(i+1)    =  numel(find(double(string(True_scores_test_ALL_HYP_kfold{C,ext_idx})) == i & double(string(Predicted_scores_ALL_HYP_test_kfold{C,ext_idx})) == i))  / numel(find(double(string(True_scores_test_ALL_HYP_kfold{C,ext_idx})) == i));
recall_SFS_test{C,ext_idx}(i+1)        =  numel(find(double(string(True_scores_test_kfold{C,ext_idx}))         == i & double(string(Predicted_scores_SFS_test_kfold{C,ext_idx})) == i))      / numel(find(double(string(True_scores_test_kfold{C,ext_idx})) == i));
recall_SFS_HYP_test{C,ext_idx}(i+1)    =  numel(find(double(string(True_scores_test_SFS_HYP_kfold{C,ext_idx})) == i & double(string(Predicted_scores_SFS_HYP_test_kfold{C,ext_idx})) == i))  / numel(find(double(string(True_scores_test_SFS_HYP_kfold{C,ext_idx})) == i));

% precision validation set
precision_ALL_val{C,ext_idx}(i+1)     =  numel(find(double(string(True_scores_val_kfold{C,ext_idx}))         == i & double(string(Predicted_scores_ALL_val_kfold{C,ext_idx})) == i))      / numel(find(double(string(Predicted_scores_ALL_val_kfold{C,ext_idx})) == i));
precision_ALL_HYP_val{C,ext_idx}(i+1) =  numel(find(double(string(True_scores_val_ALL_HYP_kfold{C,ext_idx})) == i & double(string(Predicted_scores_ALL_HYP_val_kfold{C,ext_idx})) == i))  / numel(find(double(string(Predicted_scores_ALL_HYP_val_kfold{C,ext_idx})) == i));
precision_SFS_val{C,ext_idx}(i+1)     =  numel(find(double(string(True_scores_val_kfold{C,ext_idx}))         == i & double(string(Predicted_scores_SFS_val_kfold{C,ext_idx})) == i))      / numel(find(double(string(Predicted_scores_SFS_val_kfold{C,ext_idx})) == i));
precision_SFS_HYP_val{C,ext_idx}(i+1) =  numel(find(double(string(True_scores_val_SFS_HYP_kfold{C,ext_idx})) == i & double(string(Predicted_scores_SFS_HYP_val_kfold{C,ext_idx})) == i))  / numel(find(double(string(Predicted_scores_SFS_HYP_val_kfold{C,ext_idx})) == i));

% precision test set
precision_ALL_test{C,ext_idx}(i+1)     =  numel(find(double(string(True_scores_test_kfold{C,ext_idx}))         == i & double(string(Predicted_scores_ALL_test_kfold{C,ext_idx})) == i))      / numel(find(double(string(Predicted_scores_ALL_test_kfold{C,ext_idx})) == i));
precision_ALL_HYP_test{C,ext_idx}(i+1) =  numel(find(double(string(True_scores_test_ALL_HYP_kfold{C,ext_idx})) == i & double(string(Predicted_scores_ALL_HYP_test_kfold{C,ext_idx})) == i))  / numel(find(double(string(Predicted_scores_ALL_HYP_test_kfold{C,ext_idx})) == i));
precision_SFS_test{C,ext_idx}(i+1)     =  numel(find(double(string(True_scores_test_kfold{C,ext_idx}))         == i & double(string(Predicted_scores_SFS_test_kfold{C,ext_idx})) == i))      / numel(find(double(string(Predicted_scores_SFS_test_kfold{C,ext_idx})) == i));
precision_SFS_HYP_test{C,ext_idx}(i+1) =  numel(find(double(string(True_scores_test_SFS_HYP_kfold{C,ext_idx})) == i & double(string(Predicted_scores_SFS_HYP_test_kfold{C,ext_idx})) == i))  / numel(find(double(string(Predicted_scores_SFS_HYP_test_kfold{C,ext_idx})) == i));
end

% Calculate mean recall and precision (val data set)
recall_mean_ALL_val{C,ext_idx}         = mean(recall_ALL_val{C,ext_idx}(~isnan(recall_ALL_val{C,ext_idx})));
recall_mean_ALL_HYP_val{C,ext_idx}     = mean(recall_ALL_HYP_val{C,ext_idx}(~isnan(recall_ALL_HYP_val{C,ext_idx})));
recall_mean_SFS_val{C,ext_idx}         = mean(recall_SFS_val{C,ext_idx}(~isnan(recall_SFS_val{C,ext_idx})));
recall_mean_SFS_HYP_val{C,ext_idx}     = mean(recall_SFS_HYP_val{C,ext_idx}(~isnan(recall_SFS_HYP_val{C,ext_idx})));

precision_mean_ALL_val{C,ext_idx}      = mean(precision_ALL_val{C,ext_idx}(~isnan(precision_ALL_val{C,ext_idx})));
precision_mean_ALL_HYP_val{C,ext_idx}  = mean(precision_ALL_HYP_val{C,ext_idx}(~isnan(precision_ALL_HYP_val{C,ext_idx})));
precision_mean_SFS_val{C,ext_idx}      = mean(precision_SFS_val{C,ext_idx}(~isnan(precision_SFS_val{C,ext_idx})));
precision_mean_SFS_HYP_val{C,ext_idx}  = mean(precision_SFS_HYP_val{C,ext_idx}(~isnan(precision_SFS_HYP_val{C,ext_idx})));

% Calculate mean recall and precision (test data set)
recall_mean_ALL_test{C,ext_idx}         = mean(recall_ALL_test{C,ext_idx}(~isnan(recall_ALL_test{C,ext_idx})));
recall_mean_ALL_HYP_test{C,ext_idx}     = mean(recall_ALL_HYP_test{C,ext_idx}(~isnan(recall_ALL_HYP_test{C,ext_idx})));
recall_mean_SFS_test{C,ext_idx}         = mean(recall_SFS_test{C,ext_idx}(~isnan(recall_SFS_test{C,ext_idx})));
recall_mean_SFS_HYP_test{C,ext_idx}     = mean(recall_SFS_HYP_test{C,ext_idx}(~isnan(recall_SFS_HYP_test{C,ext_idx})));

precision_mean_ALL_test{C,ext_idx}      = mean(precision_ALL_test{C,ext_idx}(~isnan(precision_ALL_test{C,ext_idx})));
precision_mean_ALL_HYP_test{C,ext_idx}  = mean(precision_ALL_HYP_test{C,ext_idx}(~isnan(precision_ALL_HYP_test{C,ext_idx})));
precision_mean_SFS_test{C,ext_idx}      = mean(precision_SFS_test{C,ext_idx}(~isnan(precision_SFS_test{C,ext_idx})));
precision_mean_SFS_HYP_test{C,ext_idx}  = mean(precision_SFS_HYP_test{C,ext_idx}(~isnan(precision_SFS_HYP_test{C,ext_idx})));
 
% Calculate F1 scores (test data set)
F1_ALL_test{C,ext_idx}      = 2 * precision_mean_ALL_test{C,ext_idx}      * recall_mean_ALL_test{C,ext_idx}      / (precision_mean_ALL_test{C,ext_idx}     + recall_mean_ALL_test{C,ext_idx});
F1_ALL_HYP_test{C,ext_idx}  = 2 * precision_mean_ALL_HYP_test{C,ext_idx}  * recall_mean_ALL_HYP_test{C,ext_idx}  / (precision_mean_ALL_HYP_test{C,ext_idx} + recall_mean_ALL_HYP_test{C,ext_idx});
F1_SFS_test{C,ext_idx}      = 2 * precision_mean_SFS_test{C,ext_idx}      * recall_mean_SFS_test{C,ext_idx}      / (precision_mean_SFS_test{C,ext_idx}     + recall_mean_SFS_test{C,ext_idx});
F1_SFS_HYP_test{C,ext_idx}  = 2 * precision_mean_SFS_HYP_test{C,ext_idx}  * recall_mean_SFS_HYP_test{C,ext_idx}  / (precision_mean_SFS_HYP_test{C,ext_idx} + recall_mean_SFS_HYP_test{C,ext_idx});

% Calculate F1 scores (val data set)
F1_ALL_val{C,ext_idx}      = 2 * precision_mean_ALL_val{C,ext_idx}      * recall_mean_ALL_val{C,ext_idx}      / (precision_mean_ALL_val{C,ext_idx}     + recall_mean_ALL_val{C,ext_idx});
F1_ALL_HYP_val{C,ext_idx}  = 2 * precision_mean_ALL_HYP_val{C,ext_idx}  * recall_mean_ALL_HYP_val{C,ext_idx}  / (precision_mean_ALL_HYP_val{C,ext_idx} + recall_mean_ALL_HYP_val{C,ext_idx});
F1_SFS_val{C,ext_idx}      = 2 * precision_mean_SFS_val{C,ext_idx}      * recall_mean_SFS_val{C,ext_idx}      / (precision_mean_SFS_val{C,ext_idx}     + recall_mean_SFS_val{C,ext_idx});
F1_SFS_HYP_val{C,ext_idx}  = 2 * precision_mean_SFS_HYP_val{C,ext_idx}  * recall_mean_SFS_HYP_val{C,ext_idx}  / (precision_mean_SFS_HYP_val{C,ext_idx} + recall_mean_SFS_HYP_val{C,ext_idx});

results_ML_features{C,ext_idx}              =  size(features_kept2{C,ext_idx},2);
results_ML_accuracy_val_total{C,ext_idx}    =  [accuracy_ALL_val{C,ext_idx}    accuracy_ALL_HYP_val{C,ext_idx}    accuracy_SFS_val{C,ext_idx}    accuracy_SFS_HYP_val{C,ext_idx}];
results_ML_accuracy_test_total{C,ext_idx}   =  [accuracy_ALL_test{C,ext_idx}    accuracy_ALL_HYP_test{C,ext_idx}    accuracy_SFS_test{C,ext_idx}    accuracy_SFS_HYP_test{C,ext_idx}];
results_ML_recall_val_total{C,ext_idx}      =  [recall_ALL_val{C,ext_idx}  ;  recall_ALL_HYP_val{C,ext_idx}  ;  recall_SFS_val{C,ext_idx}  ;  recall_SFS_HYP_val{C,ext_idx} ];
results_ML_recall_test_total{C,ext_idx}     =  [recall_ALL_test{C,ext_idx}  ;  recall_ALL_HYP_test{C,ext_idx}  ;  recall_SFS_test{C,ext_idx}  ;  recall_SFS_HYP_test{C,ext_idx} ];
results_ML_precision_val_total{C,ext_idx}   =  [precision_ALL_val{C,ext_idx}  ;  precision_ALL_HYP_val{C,ext_idx}  ;  precision_SFS_val{C,ext_idx}  ;  precision_SFS_HYP_val{C,ext_idx} ];
results_ML_precision_test_total{C,ext_idx}  =  [precision_ALL_test{C,ext_idx}  ;  precision_ALL_HYP_test{C,ext_idx}  ;  precision_SFS_test{C,ext_idx}  ;  precision_SFS_HYP_test{C,ext_idx} ];
results_ML_F1_val_total{C,ext_idx}          =  [F1_ALL_val{C,ext_idx}    F1_ALL_HYP_val{C,ext_idx}    F1_SFS_val{C,ext_idx}    F1_SFS_HYP_val{C,ext_idx} ];
results_ML_F1_test_total{C,ext_idx}         =  [F1_ALL_test{C,ext_idx}    F1_ALL_HYP_test{C,ext_idx}    F1_SFS_test{C,ext_idx}    F1_SFS_HYP_test{C,ext_idx} ];

end

end

