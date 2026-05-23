# H8: Hardware-Aware Precision Policy Search

This branch extends H7 from module-risk prediction to hardware-aware policy selection.

Main files:

- `protocol.md`: locked H8 question, hypothesis, objective, and first experiment design.
- `analysis.md`: running synthesis.
- `code/build_h8_policy_candidates.py`: first conservative policy planner.
- `results/`: generated candidate policies and experiment outputs.

Current status: scaffolded. No H8 training runs have been executed yet.

First planning command:

```bash
python experiments/h8-hardware-aware-precision-search/code/build_h8_policy_candidates.py
```

