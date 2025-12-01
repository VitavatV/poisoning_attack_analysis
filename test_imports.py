import sys
import os
sys.path.append(os.getcwd())
print("Starting imports...")
try:
    import models
    print("Imported models successfully")
    import experiment_runner
    print("Imported experiment_runner successfully")
except Exception as e:
    print(f"Import failed: {e}")
print("Done")
