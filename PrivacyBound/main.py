from subprocess import check_call
from pathlib import Path
import sys

check_call([sys.executable, str(Path(__file__).parent / "FinalMaximumFunctionEpsilon0.py")])
check_call([sys.executable, str(Path(__file__).parent / "FinalMaximumFunctionRange1.py")])
check_call([sys.executable, str(Path(__file__).parent / "FinalMaximumFunctionRange2.py")])
check_call([sys.executable, str(Path(__file__).parent / "FinalMaximumFunctionRange3.py")])
check_call([sys.executable, str(Path(__file__).parent / "FinalMaximumInverse.py")])