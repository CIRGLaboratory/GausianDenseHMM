#!/bin/bash
python3 2_dense_reproducibility.py -n 2 -d 1 -T 1000
python3 2_dense_reproducibility.py -n 2 -d 2 -T 1000

python3 2_dense_reproducibility.py -n 3 -d 1 -T 1000
python3 2_dense_reproducibility.py -n 3 -d 2 -T 1000

python3 2_dense_reproducibility.py -n 5 -d 1 -T 10000
python3 2_dense_reproducibility.py -n 5 -d 2 -T 10000

python3 2_dense_reproducibility.py -n 7 -d 1 -T 10000
python3 2_dense_reproducibility.py -n 7 -d 2 -T 10000

python3 2_dense_reproducibility.py -n 10 -d 1 -T 1000000
python3 2_dense_reproducibility.py -n 10 -d 2 -T 1000000

# python3 2_dense_reproducibility.py -n 20 -d 1 -T 10000000
# python3 2_dense_reproducibility.py -n 20 -d 2 -T 10000000