"""Quick test: verify trainable param count with T27a config."""
import sys, yaml
from pathlib import Path
project = Path(__file__).parent.parent
sys.path.insert(0, str(project / "src"))
sys.path.insert(0, str(project / "cellSAM_source"))

import torch

with open(project / "src/config/t27a_planb_decoder.yaml") as f:
    config = yaml.safe_load(f)

from train import create_model, create_optimizer
device = torch.device("cuda")
model, adapter = create_model(config, device)
optimizer, scheduler = create_optimizer(model, config, adapter=adapter)
