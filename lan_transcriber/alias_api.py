from pathlib import Path
from typing import Dict
import yaml

# guarantee the bank always contains "S1"
def load_bank(yaml_path: Path) -> Dict[str, str]:
    if not yaml_path.exists():
        yaml_path.write_text("S1: Speaker 1\n")
    bank = yaml.safe_load(yaml_path.read_text()) or {}
    bank.setdefault("S1", "Speaker 1")
    return bank
