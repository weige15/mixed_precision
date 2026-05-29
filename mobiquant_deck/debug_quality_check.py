import traceback
import sys

sys.path.insert(0, "/home/kuotzuwei15/.codex/skills/pptx-from-layouts/scripts")
from quality_check import QualityChecker

try:
    report = QualityChecker("mobiquant_deck/mobiquant_summary.pptx").check(parallel=False)
    print(report.to_dict())
except Exception:
    traceback.print_exc()
