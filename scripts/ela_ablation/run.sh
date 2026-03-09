#!/bin/bash

MAX_JOBS=40

run_job () {
while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]; do
sleep 2
done

```
nohup python scripts/ela_ablation/1_sampling_points.py "$@" > /dev/null 2>&1 &
```

}

for problem in {2..4}; do
for seed in {0..99}; do
for n_coef in 10 11 13 16 18 22 25 30 35 41 48 57 67 78 92 108 126 148 174 204 239 280 329 385 452 529 621 727 853 1000; do
run_job $problem $seed 0 $n_coef
done
done
done

for problem in {2..5}; do
for seed in {0..99}; do
for block_coef in 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0 4.1 4.2 4.3 4.4 4.5 4.6 4.7 4.8 4.9 5.0; do
run_job $problem $seed $block_coef 0
done
done
done

wait
echo "All jobs submitted."
