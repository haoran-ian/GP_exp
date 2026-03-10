#!/bin/bash
export PYTHONUNBUFFERED=1

ProblemName=("META_SURFACE" "META_SURFACE_SOLVER" "PHOTONIC_2LAYERS_ELLIPSOMETRY" "PHOTONIC_10LAYERS_BRAGG" "PHOTONIC_20LAYERS_BRAGG" "PHOTONIC_10LAYERS_PHOTOVOLTAIC")


MAX_CONCURRENT=20


count=0
for problem in {2..4}; do
    for seed in {0..99}; do
        for n_coef in 10 11 13 16 18 22 25 30 35 41 48 57 67 78 92 108 126 148 174 204 239 280 329 385 452 529 621 727 853 1000; do
            outfile="data/Ablation_ELA/Y/ProblemName.${ProblemName[$problem]}-seed:$seed-block_coef:0.0-n_coef:$n_coef.npy"
            if [ -f "$outfile" ]; then
                echo "$outfile exists, skip"
                continue
            fi
            if [ $count -ge $MAX_CONCURRENT ]; then
                wait -n
                count=$((count-1))
            fi
            echo "Starting: $problem seed:$seed block_coef:0.0 n_coef:$n_coef"
            python scripts/ela_ablation/1_sampling_points.py $problem $seed 0.0 $n_coef &
            count=$((count+1))
        done
    done
done


count=0
for problem in {2..4}; do
    for seed in {0..99}; do
        for block_coef in 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0 4.1 4.2 4.3 4.4 4.5 4.6 4.7 4.8 4.9 5.0; do
            outfile="data/Ablation_ELA/Y/ProblemName.${ProblemName[$problem]}-seed:$seed-block_coef:$block_coef-n_coef:0.npy"
            if [ -f "$outfile" ]; then
                echo "$outfile exists, skip"
                continue
            fi
            if [ $count -ge $MAX_CONCURRENT ]; then
                wait -n
                count=$((count-1))
            fi
            echo "Starting: $problem seed:$seed block_coef:0.0 n_coef:$n_coef"
            python scripts/ela_ablation/1_sampling_points.py $problem $seed $block_coef 0 &
            count=$((count+1))
        done
    done
done


count=0
cell_list=(10 11 12)
for problem in {2..4}; do
    for ela_set_id in {0..17}; do
        for seed in {0..99}; do
            if [[ " ${cell_list[@]} " =~ " ${ela_set_id} " ]]; then
                for block_coef in 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0 4.1 4.2 4.3 4.4 4.5 4.6 4.7 4.8 4.9 5.0; do
                    outfile="data/Ablation_ELA/atom/ProblemName.${ProblemName[$problem]}-$ela_set_id-$seed:$seed-block_coef:$block_coef-n_coef:0.npy"
                    if [ -f "$outfile" ]; then
                        echo "$outfile exists, skip"
                        continue
                    fi
                    if [ $count -ge $MAX_CONCURRENT ]; then
                        wait -n
                        count=$((count-1))
                    fi
                    echo "Starting: $problem seed:$seed block_coef:0.0 n_coef:$n_coef"
                    python scripts/ela_ablation/2_ela_ablation.py $problem $ela_set_id $seed $block_coef 0 &
                    count=$((count+1))
                done
            else
                for n_coef in 10 11 13 16 18 22 25 30 35 41 48 57 67 78 92 108 126 148 174 204 239 280 329 385 452 529 621 727 853 1000; do
                    outfile="data/Ablation_ELA/atom/ProblemName.${ProblemName[$problem]}-$ela_set_id-$seed:$seed-block_coef:0.0-n_coef:$n_coef.npy"
                    if [ -f "$outfile" ]; then
                        echo "$outfile exists, skip"
                        continue
                    fi
                    if [ $count -ge $MAX_CONCURRENT ]; then
                        wait -n
                        count=$((count-1))
                    fi
                    echo "Starting: $problem seed:$seed block_coef:0.0 n_coef:$n_coef"
                    python scripts/ela_ablation/2_ela_ablation.py $problem $ela_set_id $seed 0.0 $n_coef &
                    count=$((count+1))
                done
            fi
        done
    done
done

wait
echo "All jobs submitted."
