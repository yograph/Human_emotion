#!/bin/bash
#SBATCH --job-name=emotional_script       # Job name
#SBATCH --output=emotional_output.log     # Stdout log
#SBATCH --error=emotional_error.log       # Stderr log
#SBATCH --ntasks=1                        # Single task
#SBATCH --cpus-per-task=8                 # CPU cores per task
#SBATCH --mem=32GB                        # RAM
#SBATCH --time=04:00:00                   # Time limit hh:mm:ss
#SBATCH --partition=gpu                   # GPU partition name
#SBATCH --gres=gpu:2                      # Request 2 GPUs

# 1) Load required modules
module load python/3.9
module load cuda/11.7   # Adjust to whatever CUDA version is available & compatible

# 2) Activate your Python environment if you have one
#    (Replace the path below with the actual path to your venv or conda env)
# source ~/myenv/bin/activate

# 3) Go to the directory containing emotional.py
cd /home2/s5501253/human/

# 4) (Optional) Install missing packages on the fly if you haven't already
# pip install --user transformers datasets accelerate torch torchvision ...

# 5) Run the script
python emotional.py
