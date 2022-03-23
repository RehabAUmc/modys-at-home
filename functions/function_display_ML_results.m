function [TABLES_ML_accuracy, TABLES_ML_F1] = function_display_ML_results(results_ML_accuracy_val_total_save, results_ML_accuracy_test_total_save, results_features, results_ML_F1_val_total_save, results_ML_F1_test_total_save)

for i = find(~cellfun(@isempty,results_ML_accuracy_val_total_save(1,:)))
    algorithms            = ["KNN";"KNN";"Decision Tree";"Decision Tree";"SVM";"SVM";"Naïve Bayes";"Naïve Bayes";"Discriminant Analysis";"Discriminant Analysis";"Ensemble";"Ensemble"];
    algorithms_F1         = ["KNN";"Decision Tree";"SVM";"Naïve Bayes";"Discriminant Analysis";"Ensemble"];
    val_test              = ["val";"test";"val";"test";"val";"test";"val";"test";"val";"test";"val";"test"];
    number_of_features    = [results_features{1,i} ; "-" ;results_features{2,i};"-";results_features{3,i}; "-"; results_features{4,i} ; "-" ; results_features{5,i} ; "-"; results_features{6,i} ; "-" ];
    number_of_features_F1 = [results_features{1,i} ; results_features{2,i} ; results_features{3,i}; results_features{4,i} ; results_features{5,i} ; results_features{6,i}];
    results_accuracy_val  = vertcat(results_ML_accuracy_val_total_save{:,i});
    results_accuracy_test = vertcat(results_ML_accuracy_test_total_save{:,i});
    results_F1_val        = vertcat(results_ML_F1_val_total_save{:,i});
    results_F1_test       = vertcat(results_ML_F1_test_total_save{:,i});

    ALL     = [results_accuracy_val(1,1) ; results_accuracy_test(1,1) ; results_accuracy_val(2,1) ; results_accuracy_test(2,1) ; results_accuracy_val(3,1) ; results_accuracy_test(3,1) ; results_accuracy_val(4,1) ; results_accuracy_test(4,1) ; results_accuracy_val(5,1) ; results_accuracy_test(5,1) ; results_accuracy_val(6,1) ; results_accuracy_test(6,1)];
    ALL_HYP = [results_accuracy_val(1,2) ; results_accuracy_test(1,2) ; results_accuracy_val(2,2) ; results_accuracy_test(2,2) ; results_accuracy_val(3,2) ; results_accuracy_test(3,2) ; results_accuracy_val(4,2) ; results_accuracy_test(4,2) ; results_accuracy_val(5,2) ; results_accuracy_test(5,2) ; results_accuracy_val(6,2) ; results_accuracy_test(6,2)];
    SFS     = [results_accuracy_val(1,3) ; results_accuracy_test(1,3) ; results_accuracy_val(2,3) ; results_accuracy_test(2,3) ; results_accuracy_val(3,3) ; results_accuracy_test(3,3) ; results_accuracy_val(4,3) ; results_accuracy_test(4,3) ; results_accuracy_val(5,3) ; results_accuracy_test(5,3) ; results_accuracy_val(6,3) ; results_accuracy_test(6,3)];
    SFS_HYP = [results_accuracy_val(1,4) ; results_accuracy_test(1,4) ; results_accuracy_val(2,4) ; results_accuracy_test(2,4) ; results_accuracy_val(3,4) ; results_accuracy_test(3,4) ; results_accuracy_val(4,4) ; results_accuracy_test(4,4) ; results_accuracy_val(5,4) ; results_accuracy_test(5,4) ; results_accuracy_val(6,4) ; results_accuracy_test(6,4)];
    
    ALL_F1_val       = round(results_F1_val(:,1),2);
    ALL_HYP_F1_val   = round(results_F1_val(:,2),2);
    SFS_F1_val       = round(results_F1_val(:,3),2);
    SFS_HYP_F1_val   = round(results_F1_val(:,4),2);
    ALL_F1_test      = round(results_F1_test(:,1),2);
    ALL_HYP_F1_test  = round(results_F1_test(:,2),2);
    SFS_F1_test      = round(results_F1_test(:,3),2);
    SFS_HYP_F1_test  = round(results_F1_test(:,4),2);
    
        if i == 1
            TABLE_ML_dys_lower      = table(algorithms, number_of_features, val_test , ALL , ALL_HYP , SFS , SFS_HYP)
            TABLE_ML_dys_lower_F1   = table(algorithms_F1, number_of_features_F1 , ALL_F1_val , ALL_F1_test, ALL_HYP_F1_val , ALL_HYP_F1_test, SFS_F1_val , SFS_F1_test,  SFS_HYP_F1_val, SFS_HYP_F1_test)
            TABLES_ML_accuracy{i}   = TABLE_ML_dys_lower;
            TABLES_ML_F1{i}         = TABLE_ML_dys_lower_F1;
        elseif i == 2
            TABLE_ML_dys_upper      = table(algorithms, number_of_features, val_test , ALL , ALL_HYP , SFS , SFS_HYP)
            TABLE_ML_dys_upper_F1   = table(algorithms_F1, number_of_features_F1 , ALL_F1_val , ALL_F1_test, ALL_HYP_F1_val , ALL_HYP_F1_test, SFS_F1_val , SFS_F1_test,  SFS_HYP_F1_val, SFS_HYP_F1_test)
            TABLES_ML_accuracy{i}   = TABLE_ML_dys_upper;
            TABLES_ML_F1{i}         = TABLE_ML_dys_upper_F1;
        elseif i == 3
            TABLE_ML_cho_lower      = table(algorithms, number_of_features, val_test , ALL , ALL_HYP , SFS , SFS_HYP)
            TABLE_ML_cho_lower_F1   = table(algorithms_F1, number_of_features_F1 , ALL_F1_val , ALL_F1_test, ALL_HYP_F1_val , ALL_HYP_F1_test, SFS_F1_val , SFS_F1_test,  SFS_HYP_F1_val, SFS_HYP_F1_test)
            TABLES_ML_accuracy{i}   = TABLE_ML_cho_lower;
            TABLES_ML_F1{i}         = TABLE_ML_cho_lower_F1;
        elseif i == 4
            TABLE_ML_cho_upper      = table(algorithms, number_of_features, val_test , ALL , ALL_HYP , SFS , SFS_HYP)
        	TABLE_ML_cho_upper_F1   = table(algorithms_F1, number_of_features_F1 , ALL_F1_val , ALL_F1_test, ALL_HYP_F1_val , ALL_HYP_F1_test, SFS_F1_val , SFS_F1_test,  SFS_HYP_F1_val, SFS_HYP_F1_test)
            TABLES_ML_accuracy{i}   = TABLE_ML_cho_upper;
            TABLES_ML_F1{i}         = TABLE_ML_cho_upper_F1;
        elseif i == 5
            TABLE_ML_dys_Total      = table(algorithms, number_of_features, val_test , ALL , ALL_HYP , SFS , SFS_HYP)
            TABLE_ML_dys_Total_F1   = table(algorithms_F1, number_of_features_F1 , ALL_F1_val , ALL_F1_test, ALL_HYP_F1_val , ALL_HYP_F1_test, SFS_F1_val , SFS_F1_test,  SFS_HYP_F1_val, SFS_HYP_F1_test)
            TABLES_ML_accuracy{i}   = TABLE_ML_dys_Total;
            TABLES_ML_F1{i}         = TABLE_ML_dys_Total_F1;
        elseif i == 6
            TABLE_ML_cho_Total      = table(algorithms, number_of_features, val_test , ALL , ALL_HYP , SFS , SFS_HYP)
            TABLE_ML_cho_Total_F1   = table(algorithms_F1, number_of_features_F1 , ALL_F1_val , ALL_F1_test, ALL_HYP_F1_val , ALL_HYP_F1_test, SFS_F1_val , SFS_F1_test,  SFS_HYP_F1_val, SFS_HYP_F1_test)
            TABLES_ML_accuracy{i}   = TABLE_ML_cho_Total;
            TABLES_ML_F1{i}         = TABLE_ML_cho_Total_F1;
        end
end

end
