#!/bin/bash -l
conda activate cellsam
pip install 'numpy<2'
python -c "import numpy; print('numpy:', numpy.__version__)"
