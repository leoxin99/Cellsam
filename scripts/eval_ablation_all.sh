#!/bin/bash -l
#SBATCH --job-name=abl_eval
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/abl_eval_%j.log
#SBATCH --error=logs/abl_eval_%j.err

# Re-evaluate all 14 ablation checkpoints with seed-separated output
# A100 checkpoints (earlier timestamp) → seed42/
# L4 checkpoints (later timestamp) → seed123/

set -o pipefail

module load CUDA/12.1.1
eval "$(conda shell.bash hook)"
conda activate cellsam
set -u

cd ~/CellSam
mkdir -p experiments/ablation_eval/seed42 experiments/ablation_eval/seed123

echo "=== Re-evaluating all ablation checkpoints ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# A100 checkpoints (seed=42) — earlier timestamps
echo ""
echo "=== A100 (seed=42) ==="
python tools/eval_ablation.py --exp-dir checkpoints/E_phase1_rebalance_l4_20260222_045141  --output experiments/ablation_eval/seed42
python tools/eval_ablation.py --exp-dir checkpoints/Ab0_bce_dice_only_20260222_064612      --output experiments/ablation_eval/seed42
python tools/eval_ablation.py --exp-dir checkpoints/Ab1_no_boundary_20260222_081441        --output experiments/ablation_eval/seed42
python tools/eval_ablation.py --exp-dir checkpoints/Ab2_no_contour_20260222_100140         --output experiments/ablation_eval/seed42
python tools/eval_ablation.py --exp-dir checkpoints/Ab3_no_aji_20260222_115004             --output experiments/ablation_eval/seed42
python tools/eval_ablation.py --exp-dir checkpoints/Ab4_no_pq_earlystop_20260222_133812    --output experiments/ablation_eval/seed42
python tools/eval_ablation.py --exp-dir checkpoints/Ab5_posw10_20260222_151355             --output experiments/ablation_eval/seed42

# L4 checkpoints (seed=123) — later timestamps
echo ""
echo "=== L4 (seed=123) ==="
python tools/eval_ablation.py --exp-dir checkpoints/E_phase1_rebalance_l4_20260222_045402  --output experiments/ablation_eval/seed123
python tools/eval_ablation.py --exp-dir checkpoints/Ab0_bce_dice_only_20260222_082648      --output experiments/ablation_eval/seed123
python tools/eval_ablation.py --exp-dir checkpoints/Ab1_no_boundary_20260222_112406        --output experiments/ablation_eval/seed123
python tools/eval_ablation.py --exp-dir checkpoints/Ab2_no_contour_20260222_131655         --output experiments/ablation_eval/seed123
python tools/eval_ablation.py --exp-dir checkpoints/Ab3_no_aji_20260222_161532             --output experiments/ablation_eval/seed123
python tools/eval_ablation.py --exp-dir checkpoints/Ab4_no_pq_earlystop_20260222_191300    --output experiments/ablation_eval/seed123
python tools/eval_ablation.py --exp-dir checkpoints/Ab5_posw10_20260222_221205             --output experiments/ablation_eval/seed123

echo ""
echo "=== All done ==="
echo "Date: $(date)"
echo "seed42:"; ls experiments/ablation_eval/seed42/
echo "seed123:"; ls experiments/ablation_eval/seed123/
