# (c) 2023 - Brian Jay Tang, University of Michigan, <bjaytang@umich.edu>
#
# This file is part of Eyeshield
#
# Released under the GPL License, see included LICENSE file

python eval/compute_scores.py --dataset div2kvalid --mode blur --grid 1 --sigma 16
python eval/compute_scores.py --dataset ricovalid --mode blur --grid 1 --sigma 16

python eval/compute_scores.py --dataset div2kvalid --mode pixelate --grid 1 --blocks 16
python eval/compute_scores.py --dataset ricovalid --mode pixelate --grid 1 --blocks 16
