from subprocess import check_call
from pathlib import Path
import sys

check_call([sys.executable, str(Path(__file__).parent / "FinalMaximumFunction.py")])
check_call([sys.executable, str(Path(__file__).parent / "FinalMaximumInverse.py")])