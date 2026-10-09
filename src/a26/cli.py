"""Public paid CLI: python -m a26.cli --stage ... --job unique-id."""
from pathlib import Path
import importlib.util

def main():
    path=Path(__file__).resolve().parents[2]/'tools/a26_driver.py'
    spec=importlib.util.spec_from_file_location('_a26_paid_driver',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.main()

if __name__=='__main__':main()
