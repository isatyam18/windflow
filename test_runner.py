import sys
import traceback
from pathlib import Path

log_file = Path(__file__).resolve().parent / "debug_output.txt"

with open(log_file, "w", encoding="utf-8") as f:
    f.write("Starting execution...\n")
    f.flush()
    try:
        import predict_insat
        f.write("Execution finished successfully!\n")
    except Exception as e:
        f.write(f"Exception occurred: {e}\n")
        f.write(traceback.format_exc())
    f.flush()
