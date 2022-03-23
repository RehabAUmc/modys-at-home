function [table_best_models_performance, best_model_dys_lower, best_model_dys_upper, best_model_cho_lower, best_model_cho_upper, best_model_dys_total, best_model_cho_total]     = function_pick_best_models(extremity_idx, best_ML_model, best_DL_model, best_ML_model_F1_val_save, best_DL_model_F1_val, best_ML_model_F1_test_save, best_DL_model_F1_test, best_ML_model_accuracy_val_save, best_DL_model_accuracy_val, best_ML_model_accuracy_test_save, best_DL_model_accuracy_test, best_ML_model_recall_val_save, best_DL_model_recall_val, best_ML_model_precision_val_save, best_DL_model_precision_val, best_ML_model_recall_test_save, best_DL_model_recall_test, best_ML_model_precision_test_save, best_DL_model_precision_test)

extremity_fullnames = ["dystonia of the lower extremities";"dystonia of the upper extremities";"choreaothetosis of the lower extremities";"choreaothetosis of the upper extremities";"dystonia total";"choreoathetosis total"];

best_model_dys_lower =[];
best_model_dys_upper =[];
best_model_cho_lower =[];
best_model_cho_upper =[];
best_model_dys_total =[];
best_model_cho_total =[];

% For each extremity, find which model is the best (best ML
% model vs best DL model)
for i = extremity_idx
        
    if best_ML_model_F1_val_save{1,i} > best_DL_model_F1_val{1,i}
        idx_algorithm_type(i) = 1;
        disp(char("Best ML model has higher F1 score (" + string(best_ML_model_F1_val_save{1,i}) + ") than DL model (" + string(best_DL_model_F1_val{1,i}) + "%) for " + extremity_fullnames(i)))    
        if i == 1 && isempty(best_ML_model{1,i}) == 0
            best_model_dys_lower  = best_ML_model{1,i};
        end
        if i == 2 && isempty(best_ML_model{1,i}) == 0
            best_model_dys_upper  = best_ML_model{1,i};
        end       
        if i == 3 && isempty(best_ML_model{1,i}) == 0
            best_model_cho_lower  = best_ML_model{1,i};
        end
        if i == 4 && isempty(best_ML_model{1,i}) == 0
            best_model_cho_upper  = best_ML_model{1,i};
        end       
        if i == 5 && isempty(best_ML_model{1,i}) == 0
            best_model_dys_total  = best_ML_model{1,i};
        end
        if i == 6 && isempty(best_ML_model{1,i}) == 0
            best_model_cho_total  = best_ML_model{1,i};
        end      
    end 
    
    if best_ML_model_F1_val_save{1,i} < best_DL_model_F1_val{1,i}
        idx_algorithm_type(i) = 2;
        disp(char("DL model has higher F1 score (" + string(best_DL_model_F1_val{1,i}) + ") than best ML model (" + string(best_ML_model_F1_val_save{1,i}) + "%) for " + extremity_fullnames(i)))    
        if i == 1 && isempty(best_DL_model{1,i}) == 0
            best_model_dys_lower  = best_DL_model{1,i};
        end
        if i == 2 && isempty(best_DL_model{1,i}) == 0
            best_model_dys_upper  = best_DL_model{1,i};
        end       
        if i == 3 && isempty(best_DL_model{1,i}) == 0
            best_model_cho_lower  = best_DL_model{1,i};
        end
        if i == 4 && isempty(best_DL_model{1,i}) == 0
            best_model_cho_upper  = best_DL_model{1,i};
        end       
        if i == 5 && isempty(best_DL_model{1,i}) == 0
            best_model_dys_total  = best_DL_model{1,i};
        end
        if i == 6 && isempty(best_DL_model{1,i}) == 0
            best_model_cho_total  = best_DL_model{1,i};
        end    
    end
    
    if best_ML_model_F1_val_save{1,i} == best_DL_model_F1_val{1,i}
        if best_ML_model_accuracy_val_save{1,i} < best_DL_model_accuracy_val{1,i}
            idx_algorithm_type(i) = 2;
            disp(char("Best ML model and DL model F1 score (" + string(best_DL_model_F1_val{1,i}) + ") are equal for " + extremity_fullnames(i) + ", DL model is chosen since the DL model has a higher validation accuracy"))
            if i == 1 && isempty(best_DL_model{1,i}) == 0
                best_model_dys_lower  = best_DL_model{1,i};
            end
            if i == 2 && isempty(best_DL_model{1,i}) == 0
                best_model_dys_upper  = best_DL_model{1,i};
            end       
            if i == 3 && isempty(best_DL_model{1,i}) == 0
                best_model_cho_lower  = best_DL_model{1,i};
            end
            if i == 4 && isempty(best_DL_model{1,i}) == 0
                best_model_cho_upper  = best_DL_model{1,i};
            end       
            if i == 5 && isempty(best_DL_model{1,i}) == 0
                best_model_dys_total  = best_DL_model{1,i};
            end
            if i == 6 && isempty(best_DL_model{1,i}) == 0
                best_model_cho_total  = best_DL_model{1,i};
            end 
        else
            idx_algorithm_type(i) = 1;
            disp(char("Best ML model and DL model F1 score (" + string(best_ML_model_F1_val_save{1,i}) + ") are equal for " + extremity_fullnames(i) + ", ML model is chosen since the ML model has a higher validation accuracy"))
            if i == 1 && isempty(best_ML_model{1,i}) == 0
                best_model_dys_lower  = best_ML_model{1,i};
            end
            if i == 2 && isempty(best_ML_model{1,i}) == 0
                best_model_dys_upper  = best_ML_model{1,i};
            end       
            if i == 3 && isempty(best_ML_model{1,i}) == 0
                best_model_cho_lower  = best_ML_model{1,i};
            end
            if i == 4 && isempty(best_ML_model{1,i}) == 0
                best_model_cho_upper  = best_ML_model{1,i};
            end       
            if i == 5 && isempty(best_ML_model{1,i}) == 0
                best_model_dys_total  = best_ML_model{1,i};
            end
            if i == 6 && isempty(best_ML_model{1,i}) == 0
                best_model_cho_total  = best_ML_model{1,i};
            end 
        end
    end 
    
end

% Overview results
for i = extremity_idx
    if idx_algorithm_type(i) == 1
        best_model_val_F1(i)         = best_ML_model_F1_val_save{i};
        best_model_test_F1(i)        = best_ML_model_F1_test_save{i};
        best_model_val_accuracy(i)   = best_ML_model_accuracy_val_save{i};
        best_model_test_accuracy(i)  = best_ML_model_accuracy_test_save{i};
%         best_model_val_recall{i}     = best_ML_model_recall_val(i);
%         best_model_val_precision{i}  = best_ML_model_precision_val(i);
%         best_model_test_recall{i}    = best_ML_model_recall_test(i);
%         best_model_test_precision{i} = best_ML_model_precision_test(i);
    elseif idx_algorithm_type(i) == 2
        best_model_val_F1(i)         = best_DL_model_F1_val{i};
        best_model_test_F1(i)        = best_DL_model_F1_test{i};
        best_model_val_accuracy(i)   = best_DL_model_accuracy_val{i};
        best_model_test_accuracy(i)  = best_DL_model_accuracy_test{i};
%         best_model_val_recall{i}     = best_DL_model_recall_val(i);
%         best_model_val_precision{i}  = best_DL_model_precision_val(i);
%         best_model_test_recall{i}    = best_DL_model_recall_test(i);
%         best_model_test_precision{i} = best_DL_model_precision_test(i);
    end
end   
   
    
    %table_best_models_performance = [table({'F1 validadion';'F1 test';'accuracy validation';'accuracy test';'recall test 1';'recall test 2';'recall test 3';'recall test 4';'recall test 5';'precision test 1';'precision test 2';'precision test 3';'precision test 4';'precision test 5'},'VariableNames',{'score'})  table(repmat(NaN,14,1),'VariableNames',{'dystonia_lower'})  table(repmat(NaN,14,1),'VariableNames',{'dystonia_upper'})  table(repmat(NaN,14,1),'VariableNames',{'choreoathetosis_lower'})  table(repmat(NaN,14,1),'VariableNames',{'choreoathetosis_upper'})  table(repmat(NaN,14,1),'VariableNames',{'dystonia_total'}) table(repmat(NaN,14,1),'VariableNames',{'choreoathetosis_total'})];
    table_best_models_performance = [table({'F1 validadion';'F1 test';'accuracy validation';'accuracy test'},'VariableNames',{'score'})  table(repmat(NaN,4,1),'VariableNames',{'dystonia_lower'})  table(repmat(NaN,4,1),'VariableNames',{'dystonia_upper'})  table(repmat(NaN,4,1),'VariableNames',{'choreoathetosis_lower'})  table(repmat(NaN,4,1),'VariableNames',{'choreoathetosis_upper'})  table(repmat(NaN,4,1),'VariableNames',{'dystonia_total'}) table(repmat(NaN,4,1),'VariableNames',{'choreoathetosis_total'})];

    for j = extremity_idx
        table_best_models_performance{1,j+1} = best_model_val_F1(j);
        table_best_models_performance{2,j+1} = best_model_test_F1(j);
        table_best_models_performance{3,j+1} = best_model_val_accuracy(j);
        table_best_models_performance{4,j+1} = best_model_test_accuracy(j);
%         table_best_models_performance{5,j+1} = double(best_model_test_recall{1,j}{1,1}(1));
%         table_best_models_performance{6,j+1} = double(best_model_test_recall{1,j}{1,1}(2));
%         table_best_models_performance{7,j+1} = double(best_model_test_recall{1,j}{1,1}(3));
%         table_best_models_performance{8,j+1} = double(best_model_test_recall{1,j}{1,1}(4));
%         table_best_models_performance{9,j+1} = double(best_model_test_recall{1,j}{1,1}(5));
%         table_best_models_performance{10,j+1}= double(best_model_test_precision{1,j}{1,1}(1));
%         table_best_models_performance{11,j+1}= double(best_model_test_precision{1,j}{1,1}(2));
%         table_best_models_performance{12,j+1}= double(best_model_test_precision{1,j}{1,1}(3));
%         table_best_models_performance{13,j+1}= double(best_model_test_precision{1,j}{1,1}(4));
%         table_best_models_performance{14,j+1}= double(best_model_test_precision{1,j}{1,1}(5));

        table_best_models_performance
        %table_best_models_performance{6,j+1} = best_model_val_precision{1,j};
    end
    
%     best_val_F1        = table_template;
%     best_test_F1       = table_template;
%     best_val_accuracy  = table_template;
%     best_test_accuracy = table_template;
%     best_val_recall    = table_template;
%     best_val_precision = table_template;

%     for j = extremity_idx
%         best_val_F1{1,j}        = best_model_val_F1(j);
%         best_test_F1{1,j}       = best_model_test_F1(j);
%         best_val_accuracy{1,j}  = best_model_val_accuracy(j);
%         best_test_accuracy{1,j} = best_model_test_accuracy(j);
%         best_val_recall{1,j}    = best_model_val_recall{1,j};
%         best_val_precision{1,j} = best_model_val_precision{1,j};
%     end

end