n_seeds=10
seed=0

cd ../../..

########################################################
# Body Mass Shift
########################################################

# sharing all, unly positive
for shift in "body_mass"; do
    for env_name in "hopper" "halfcheetah" "walker2d"; do  
        # if env_name is hopper, the eval env is Hopper-v3
        # if env_name is halfcheetah, the eval env is HalfCheetah-v3
        # if env_name is walker2d, the eval env is Walker2d-v3
        if [ "$env_name" = "hopper" ]; then
            eval_env_name="Hopper-v3"
        elif [ "$env_name" = "halfcheetah" ]; then
            eval_env_name="HalfCheetah-v3"
        elif [ "$env_name" = "walker2d" ]; then
            eval_env_name="Walker2d-v3"
        fi
        for positive_data_quality in "medium_expert" "medium"; do
            for negative_data_quality in "medium_expert" "medium" "random"; do
                # if positive is medium and negative is medium_expert, skip
                if [ "$positive_data_quality" = "medium" ] && [ "$negative_data_quality" = "medium_expert" ]; then
                    continue
                fi
                for positive_ratio in 0.3; do
                    for labeled_ratio in 0.01 0.03; do
                        for method in "sharing_all" "only_p" "pu"; do
                            python train_agent.py --config_path=configs/offline/iql.yaml \
                            --data.shift=$shift \
                            --env_name=$env_name \
                            --eval_env_name=$eval_env_name \
                            --data.positive_data_quality=$positive_data_quality \
                            --data.negative_data_quality=$negative_data_quality \
                            --data.positive_ratio=$positive_ratio \
                            --data.labeled_ratio=$labeled_ratio \
                            --method=$method \
                            --n_seeds=$n_seeds \
                            --seed=$seed
                        done
                    done
                done
            done
        done
    done
done

# oracle
for shift in "body_mass"; do
    for env_name in "hopper" "halfcheetah" "walker2d"; do
        # if env_name is hopper, the eval env is Hopper-v3
        # if env_name is halfcheetah, the eval env is HalfCheetah-v3
        # if env_name is walker2d, the eval env is Walker2d-v3
        if [ "$env_name" = "hopper" ]; then
            eval_env_name="Hopper-v3"
        elif [ "$env_name" = "halfcheetah" ]; then
            eval_env_name="HalfCheetah-v3"
        elif [ "$env_name" = "walker2d" ]; then
            eval_env_name="Walker2d-v3"
        fi
        for positive_data_quality in "medium_expert" "medium"; do
            for positive_ratio in 0.3; do
                for method in "oracle"; do
                    python train_agent.py --config_path=configs/offline/iql.yaml \
                    --data.shift=$shift \
                    --env_name=$env_name \
                    --eval_env_name=$eval_env_name \
                    --data.positive_data_quality=$positive_data_quality \
                    --data.positive_ratio=$positive_ratio \
                    --method=$method \
                    --n_seeds=$n_seeds \
                    --seed=$seed    
                done
            done
        done
    done
done
