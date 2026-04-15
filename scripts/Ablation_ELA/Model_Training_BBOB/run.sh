#!/bin/bash
export PYTHONUNBUFFERED=1


MAX_CONCURRENT=10


# count=0
# for fid in {1..24}; do
#     for iid in {0..9}; do
#         for dim in 2 10 20; do
#             for seed in {0..9}; do
#                 for n_coef in 1000; do
#                     outfile="data/Ablation_ELA/Y/F$fid-$iid-D$dim-seed:$seed-block_coef:0.0-n_coef:$n_coef.npy"
#                     if [ -f "$outfile" ]; then
#                         echo "$outfile exists, skip"
#                         continue
#                     fi
#                     if [ $count -ge $MAX_CONCURRENT ]; then
#                         wait -n
#                         count=$((count-1))
#                     fi
#                     echo "Starting: F$fid,$iid,D$dim seed:$seed block_coef:0.0 n_coef:$n_coef"
#                     python scripts/Ablation_ELA/Model_Training_BBOB/1_sampling_points_BBOB.py $fid $iid $dim $seed 0.0 $n_coef &
#                     count=$((count+1))
#                 done
#             done
#         done
#     done
# done


# count=0
# for problem in {2..4}; do
#     for seed in {0..99}; do
#         for block_coef in 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0 4.1 4.2 4.3 4.4 4.5 4.6 4.7 4.8 4.9 5.0; do
#             outfile="data/Ablation_ELA/Y/ProblemName.${ProblemName[$problem]}-seed:$seed-block_coef:$block_coef-n_coef:0.npy"
#             if [ -f "$outfile" ]; then
#                 echo "$outfile exists, skip"
#                 continue
#             fi
#             if [ $count -ge $MAX_CONCURRENT ]; then
#                 wait -n
#                 count=$((count-1))
#             fi
#             echo "Starting: $problem seed:$seed block_coef:0.0 n_coef:$n_coef"
#             python scripts/ela_ablation/1_sampling_points.py $problem $seed $block_coef 0 &
#             count=$((count+1))
#         done
#     done
# done


count=0
cell_list=(10 11 12)
for fid in {1..24}; do
    for iid in {0..9}; do
        for dim in 2 10; do
            for ela_set_id in {0..6}; do
                for seed in {0..9}; do
                    if [[ " ${cell_list[@]} " =~ " ${ela_set_id} " ]]; then
                        for block_coef in 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0 4.1 4.2 4.3 4.4 4.5 4.6 4.7 4.8 4.9 5.0; do
                            outfile="data/Ablation_ELA/atom/ProblemName.${ProblemName[$problem]}-$ela_set_id-seed:$seed-block_coef:$block_coef-n_coef:0.csv"
                            if [ -f "$outfile" ]; then
                                echo "$outfile exists, skip"
                                continue
                            fi
                            if [ $count -ge $MAX_CONCURRENT ]; then
                                wait -n
                                count=$((count-1))
                            fi
                            echo "Starting: $problem, $ela_set_id, seed:$seed block_coef:0.0 n_coef:$n_coef"
                            python scripts/Ablation_ELA/2_ela_ablation.py $problem $ela_set_id $seed $block_coef 0 &
                            count=$((count+1))
                        done
                    else
                        for n_coef in 1000; do
                            outfile="data/Ablation_ELA/atom/F$fid-$iid-D$dim-$ela_set_id-seed:$seed-block_coef:0.0-n_coef:$n_coef.csv"
                            if [ -f "$outfile" ]; then
                                echo "$outfile exists, skip"
                                continue
                            fi
                            if [ $count -ge $MAX_CONCURRENT ]; then
                                wait -n
                                count=$((count-1))
                            fi
                            echo "Starting: F$fid,$iid,D$dim, $ela_set_id, seed:$seed block_coef:0.0 n_coef:$n_coef"
                            python scripts/Ablation_ELA/Model_Training_BBOB/2_ela_calculation_BBOB.py $fid $iid $dim $ela_set_id $seed 0.0 $n_coef &
                            count=$((count+1))
                        done
                    fi
                done
            done
        done
    done
done

wait
echo "All jobs submitted."
