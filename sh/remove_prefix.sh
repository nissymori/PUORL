#!/bin/bash

# Function to remove prefixes from files
remove_prefixes() {
    local subdir="$1"
    local prefix="$2"
    cd "$subdir" || exit

    for file in *; do
        new_name="${file#$prefix}"
        mv "$file" "$new_name"
    done

    cd ../../ || exit
}

# Main script
cd ../ && cd dataset || exit

for sub_dir in */; do
    sub_dir=${sub_dir%/}  # Remove trailing slash
    if [[ -d "$sub_dir/body_mass" ]]; then
        remove_prefixes "$sub_dir/body_mass" "body_"
    fi
    if [[ -d "$sub_dir/joint_noise" ]]; then
        remove_prefixes "$sub_dir/joint_noise" "joint_"
    fi
done

echo "Prefixes removed from files in the dataset directory."