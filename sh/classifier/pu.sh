seed=0
project="puorl_classification"

cd ../..

########################################################
# Body Mass Shift
########################################################

# PU
for shift_type in "body_mass"; do
    for env_name in "hopper" "halfcheetah" "walker2d"; do  
        for positive_data_quality in "medium_expert" "medium"; do
            for negative_data_quality in "medium_expert" "medium" "random"; do
                # if positive is medium and negative is medium_expert, skip
                if positive_data_quality == "medium" && negative_data_quality == "medium_expert"; then
                    continue
                fi
                for positive_ratio in 0.3; do
                    for labeled_ratio in 0.01 0.03; do
                        for method in "pu"; do
                            python train_classifier.py --shift_type=$shift_type \\
                            --env_name=$env_name \\
                            --data.positive_data_quality=$positive_data_quality \\
                            --data.negative_data_quality=$negative_data_quality \\
                            --data.positive_ratio=$positive_ratio \\
                            --data.labeled_ratio=$labeled_ratio \\
                            --method=$method \\
                            --seed=$seed
                        done
                    done
                done
            done
        done
    done
done

########################################################
# Halfcheetah vs Walker2d Shift
########################################################

# PU
for shift_type in "halfcheetah_vs_walker2d"; do
    for positive_data_quality in "medium_expert" "medium"; do
        for negative_data_quality in "medium_expert" "medium" "random"; do
            # if positive is medium and negative is medium_expert, skip
            if positive_data_quality == "medium" && negative_data_quality == "medium_expert"; then
                continue
            fi
            for positive_ratio in 0.3; do
                for labeled_ratio in 0.01 0.03; do
                    for method in "pu"; do
                        python train_classifier.py --shift_type=$shift_type \\
                        --env_name=$env_name \\
                        --data.positive_data_quality=$positive_data_quality \\
                        --data.negative_data_quality=$negative_data_quality \\
                        --data.positive_ratio=$positive_ratio \\
                        --data.labeled_ratio=$labeled_ratio \\
                        --method=$method
                        --seed=$seed
                    done
                done
            done
        done
    done
done
