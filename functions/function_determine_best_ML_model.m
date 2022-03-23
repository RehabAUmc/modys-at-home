function [best_ML_model, best_ML_model_type, best_ML_model_specs, best_ML_model_accuracy_val, best_ML_model_accuracy_test, best_ML_model_recall_val, best_ML_model_precision_val, best_ML_model_recall_test, best_ML_model_precision_test, best_ML_model_F1_val, best_ML_model_F1_test]  =   function_determine_best_ML_model(model_ALL, model_ALL_HYP, model_SFS, model_SFS_HYP, results_ML_accuracy_val_total, results_ML_accuracy_test_total, results_ML_F1_val_total, results_ML_F1_test_total, results_ML_recall_val_total, results_ML_precision_val_total, results_ML_recall_test_total, results_ML_precision_test_total, ext_idx, size_y_test)

% Calculate the results of the 5 folds combined
% for i = 1:6
%     results_ML_accuracy_total{i,ext_idx}                = (results_ML_accuracy{i,ext_idx,1}*size_y_test(1) + results_ML_accuracy{i,ext_idx,2}*size_y_test(2) + results_ML_accuracy{i,ext_idx,3}*size_y_test(3) + results_ML_accuracy{i,ext_idx,4}*size_y_test(4) + results_ML_accuracy{i,ext_idx,5}*size_y_test(5)) / sum(size_y_test);
%     results_ML_F1_val_total{i,ext_idx}         = (results_ML_F1_val{i,ext_idx,1}*size_y_test(1) + results_ML_F1_val{i,ext_idx,2}*size_y_test(2) + results_ML_F1_val{i,ext_idx,3}*size_y_test(3) + results_ML_F1_val{i,ext_idx,4}*size_y_test(4) + results_ML_F1_val{i,ext_idx,5}*size_y_test(5)) / sum(size_y_test);
%     results_ML_F1_test_total{i,ext_idx}        = (results_ML_F1_test{i,ext_idx,1}*size_y_test(1) + results_ML_F1_test{i,ext_idx,2}*size_y_test(2) + results_ML_F1_test{i,ext_idx,3}*size_y_test(3) + results_ML_F1_test{i,ext_idx,4}*size_y_test(4) + results_ML_F1_test{i,ext_idx,5}*size_y_test(5)) / sum(size_y_test);
%     results_ML_recall_val_total{i,ext_idx}     = (results_ML_recall_val{i,ext_idx,1}*size_y_test(1) + results_ML_recall_val{i,ext_idx,2}*size_y_test(2) + results_ML_recall_val{i,ext_idx,3}*size_y_test(3) + results_ML_recall_val{i,ext_idx,4}*size_y_test(4) + results_ML_recall_val{i,ext_idx,5}*size_y_test(5)) / sum(size_y_test);
%     results_ML_precision_val_total{i,ext_idx}  = (results_ML_precision_val{i,ext_idx,1}*size_y_test(1) + results_ML_precision_val{i,ext_idx,2}*size_y_test(2) + results_ML_precision_val{i,ext_idx,3}*size_y_test(3) + results_ML_precision_val{i,ext_idx,4}*size_y_test(4) + results_ML_precision_val{i,ext_idx,5}*size_y_test(5)) / sum(size_y_test);
%     results_ML_recall_test_total{i,ext_idx}     = (results_ML_recall_test{i,ext_idx,1}*size_y_test(1) + results_ML_recall_test{i,ext_idx,2}*size_y_test(2) + results_ML_recall_test{i,ext_idx,3}*size_y_test(3) + results_ML_recall_test{i,ext_idx,4}*size_y_test(4) + results_ML_recall_test{i,ext_idx,5}*size_y_test(5)) / sum(size_y_test);
%     results_ML_precision_test_total{i,ext_idx}  = (results_ML_precision_test{i,ext_idx,1}*size_y_test(1) + results_ML_precision_test{i,ext_idx,2}*size_y_test(2) + results_ML_precision_test{i,ext_idx,3}*size_y_test(3) + results_ML_precision_test{i,ext_idx,4}*size_y_test(4) + results_ML_precision_test{i,ext_idx,5}*size_y_test(5)) / sum(size_y_test);
% end

%%% determine best Machine Learning model %%%
models = [model_ALL' model_ALL_HYP' model_SFS' model_SFS_HYP'];
%models = [model_ALL' model_ALL_HYP' [model_SFS NaN]' [model_SFS_HYP]'];

% all results (validation & test set) for each dataset
results_all_accuracy_val{1,ext_idx}    = [results_ML_accuracy_val_total{1,ext_idx} ; results_ML_accuracy_val_total{2,ext_idx} ; results_ML_accuracy_val_total{3,ext_idx} ; results_ML_accuracy_val_total{4,ext_idx} ; results_ML_accuracy_val_total{5,ext_idx} ; results_ML_accuracy_val_total{6,ext_idx} ]; 
results_all_accuracy_test{1,ext_idx}   = [results_ML_accuracy_test_total{1,ext_idx} ; results_ML_accuracy_test_total{2,ext_idx} ; results_ML_accuracy_test_total{3,ext_idx} ; results_ML_accuracy_test_total{4,ext_idx} ; results_ML_accuracy_test_total{5,ext_idx} ; results_ML_accuracy_test_total{6,ext_idx} ]; 
results_all_F1_val{1,ext_idx}          = [results_ML_F1_val_total{1,ext_idx}   ; results_ML_F1_val_total{2,ext_idx}   ; results_ML_F1_val_total{3,ext_idx}   ; results_ML_F1_val_total{4,ext_idx}   ; results_ML_F1_val_total{5,ext_idx}   ; results_ML_F1_val_total{6,ext_idx}   ];
results_all_F1_test{1,ext_idx}         = [results_ML_F1_test_total{1,ext_idx}  ; results_ML_F1_test_total{2,ext_idx}  ; results_ML_F1_test_total{3,ext_idx}  ; results_ML_F1_test_total{4,ext_idx}  ; results_ML_F1_test_total{5,ext_idx}  ; results_ML_F1_test_total{6,ext_idx}  ];

% pick the model with the highest F1 score
maximum_F1 = max(max(results_all_F1_val{1,ext_idx}));                     
[row_model,col_model]= find(results_all_F1_val{1,ext_idx}==maximum_F1);   % row_model = algorithm index (1=KNN, 2=DT, 3=SVM, 4=NB, 5=DA, 6=Ensemble) ,  col_model = specifications (1=ALL, 2=ALL+HYP, 3=SFS, 4=SFS+HYP)

% if multiple models have the highest F1, pick the model
% with the highest validation accuracy
if size(row_model,1) > 1
    clear results_all_accuracy_val_temp
    for i = 1:size(row_model,1)
        results_all_accuracy_val_temp(i) = results_all_accuracy_val{1,ext_idx}(row_model(i),col_model(i));
    end
    [~,M] = max(results_all_accuracy_val_temp); 
else M = 1;
end

% best ML model for each dataset
best_ML_model{1,ext_idx} = models{row_model(M),col_model(M)};

% best ML model type, specifications, validation accuracy, test accuracy,
% recall and precision for each dataset
model_type                             = [string('KNN') ; string('Decision tree') ; string('SVM') ; string('Naïve Bayes') ; string('Discriminant Analysis') ; string('Ensemble')];
best_ML_model_type{1,ext_idx}          = model_type(row_model(M));
model_specs                            = [string('ALL') ; string('ALL + HYP') ; string('SFS') ; string('SFS + HYP')];
best_ML_model_specs{1,ext_idx}         = model_specs(col_model(M));
best_ML_model_accuracy_val{1,ext_idx}  = results_all_accuracy_val{1,ext_idx}(row_model(M),col_model(M));
best_ML_model_accuracy_test{1,ext_idx} = results_all_accuracy_test{1,ext_idx}(row_model(M),col_model(M));
best_ML_model_recall_val{1,ext_idx}    = results_ML_recall_val_total{row_model(M),ext_idx}(col_model(M),:);
best_ML_model_precision_val{1,ext_idx} = results_ML_precision_val_total{row_model(M),ext_idx}(col_model(M),:);
best_ML_model_recall_test{1,ext_idx}   = results_ML_recall_test_total{row_model(M),ext_idx}(col_model(M),:);
best_ML_model_precision_test{1,ext_idx}= results_ML_precision_test_total{row_model(M),ext_idx}(col_model(M),:);
best_ML_model_F1_val{1,ext_idx}        = results_all_F1_val{1,ext_idx}(row_model(M),col_model(M));
best_ML_model_F1_test{1,ext_idx}       = results_all_F1_test{1,ext_idx}(row_model(M),col_model(M));

end
