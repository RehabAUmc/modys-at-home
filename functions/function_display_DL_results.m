function [TABLES_DL]    = function_display_DL_results(extremity_idx, best_DL_model_hidden_layers, best_DL_model_accuracy_val, best_DL_model_accuracy_test, best_DL_model_F1_val, best_DL_model_F1_test, dataframeTrain_dystonia_lower, dataframeTrain_dystonia_upper, dataframeTrain_choreoathetosis_lower, dataframeTrain_choreoathetosis_upper, dataframeTrain_dystonia_total, dataframeTrain_choreoathetosis_total)

for i = extremity_idx
    hidden_layers = best_DL_model_hidden_layers{1,i};
    accuracy_val  = best_DL_model_accuracy_val{1,i};
    accuracy_test = best_DL_model_accuracy_test{1,i};
    F1_val        = best_DL_model_F1_val{1,i};
    F1_test       = best_DL_model_F1_test{1,i};
     
    if i == 1
        samples = sum(~isnan(dataframeTrain_dystonia_lower.dystonia_lower_extemity));
        TABLE_DL_dys_lower = table(samples, hidden_layers , F1_val, F1_test, accuracy_val , accuracy_test)
        TABLES_DL{i} = TABLE_DL_dys_lower;
    elseif i == 2
        samples = sum(~isnan(dataframeTrain_dystonia_upper.dystonia_upper_extemity));
        TABLE_DL_dys_upper = table(samples, hidden_layers , F1_val, F1_test, accuracy_val , accuracy_test)
        TABLES_DL{i} = TABLE_DL_dys_upper;
    elseif i == 3
        samples = sum(~isnan(dataframeTrain_choreoathetosis_lower.choreoathetosis_lower_extemity));
        TABLE_DL_cho_lower = table(samples, hidden_layers , F1_val, F1_test, accuracy_val , accuracy_test)
        TABLES_DL{i} = TABLE_DL_cho_lower;
    elseif i == 4
        samples = sum(~isnan(dataframeTrain_choreoathetosis_upper.choreoathetosis_upper_extemity));
        TABLE_DL_cho_upper = table(samples, hidden_layers , F1_val, F1_test, accuracy_val , accuracy_test)
        TABLES_DL{i} = TABLE_DL_cho_upper;
    elseif i == 5
        samples = sum(~isnan(dataframeTrain_dystonia_total.dystonia_total));
        TABLE_DL_dys_Total = table(samples, hidden_layers , F1_val, F1_test, accuracy_val , accuracy_test)
        TABLES_DL{i} = TABLE_DL_dys_Total;
    elseif i == 6
        samples = sum(~isnan(dataframeTrain_choreoathetosis_total.choreoathetosis_total));
        TABLE_DL_cho_Total = table(samples, hidden_layers , F1_val, F1_test, accuracy_val , accuracy_test)
        TABLES_DL{i} = TABLE_DL_cho_Total;
    end
end

end
