import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thinruntime.cli import cmd_run

if __name__ == "__main__":
    import sys
    # Monkeypatch argv to bypass argparse parsing logic in cmd_run if needed, or just let cmd_run parse sys.argv
    # We will invoke the CLI script directly using `thintensor run` with a fixed prompt
    pass
