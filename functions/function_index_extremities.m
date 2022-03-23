function extremity_idx = function_index_extremities(dystonia_lower, dystonia_upper, choreoathetosis_lower, choreoathetosis_upper, dystonia_total, choreoathetosis_total, dataframes)

    % index extremities to be modelled
    extremities = {dystonia_lower ; dystonia_upper ; choreoathetosis_lower ; choreoathetosis_upper ; dystonia_total ; choreoathetosis_total};
    
    for i = 1:6
        if extremities{i} == "yes" & length(unique(dataframes{i}{:,111})) > 1
            extremity_idx(i) = i;
        else extremity_idx(i) = NaN;
        end
    end
    
    extremity_idx(isnan(extremity_idx)) = [];

end
