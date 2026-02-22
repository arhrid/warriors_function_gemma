"""Run parent benchmark.py using warriors' main.py."""
import sys, os

# Ensure warriors' main.py is found first
warriors_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(warriors_dir)
hackathon_dir = os.path.join(parent_dir, "functiongemma-hackathon")

sys.path.insert(0, warriors_dir)
sys.path.insert(0, os.path.join(hackathon_dir, "cactus", "python", "src"))
os.environ["CACTUS_NO_CLOUD_TELE"] = "1"

# Pre-load our main so benchmark.py uses it
import main  # noqa: F401

# Execute parent benchmark
exec(open(os.path.join(hackathon_dir, "benchmark.py")).read())
