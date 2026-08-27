"""Python solver pre-pass for ARC tasks.

Loads every solver in /traces/<task_id>.py. For each ARC task, tries every
solver against the task's train pairs. If a solver matches ALL train pairs
exactly, it "claims" the task and provides test outputs that override the
LLM-derived submission slot specified by ARC_PREPASS_SLOT (default: 1).

Verified zero false positives on the full ARC-AGI-2 evaluation set
(120 tasks, 4 solvers as of 2026-04-22).

Env vars:
  ARC_DISABLE_PREPASS=1     skip prepass entirely
  ARC_PREPASS_SLOT=1|2|both which submission attempt slot to override
                            (default: 1; for unverified solvers prefer 2,
                             since kgmon ranks attempt_1 higher and
                             overriding the weaker slot bounds regression)

Usage:
    from prepass import apply_prepass_overrides
    submission = apply_prepass_overrides(submission, challenges)
"""
import importlib.util
import os


def _load_solvers(traces_dir):
    solvers = {}
    for fname in sorted(os.listdir(traces_dir)):
        if not fname.endswith(".py"):
            continue
        if fname in ("prepass.py", "verify_traces.py", "prepass_eval.py", "__init__.py", "arc_loader.py", "arc_decoder.py", "arc_solver.py", "starter.py"):
            continue
        tid = fname[:-3]
        path = os.path.join(traces_dir, fname)
        try:
            spec = importlib.util.spec_from_file_location(f"trace_{tid}", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "solve"):
                solvers[tid] = mod.solve
        except Exception as e:
            print(f"[prepass] LOAD_FAIL {fname}: {e}")
    return solvers


def _grids_equal(a, b):
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if len(ra) != len(rb):
            return False
        for x, y in zip(ra, rb):
            if x != y:
                return False
    return True


def _claim(task, solvers):
    """Return (solver_name, test_outputs, train_match_count) if some solver
    claims this task by matching all train pairs; else None."""
    train = task["train"]
    test_inputs = [t["input"] for t in task["test"]]
    for sname, solve in solvers.items():
        try:
            matched = 0
            ok = True
            for pair in train:
                out = solve(pair["input"])
                if not _grids_equal(out, pair["output"]):
                    ok = False
                    break
                matched += 1
            if not ok:
                continue
            test_outs = [solve(ti) for ti in test_inputs]
            return sname, test_outs, matched
        except Exception:
            continue
    return None


def apply_prepass_overrides(submission, challenges, traces_dir=None, verbose=True):
    """Override submission entries for tasks claimed by Python solvers.

    submission: dict {task_id: [{"attempt_1": grid, "attempt_2": grid}, ...]}
    challenges: dict {task_id: {"train": [...], "test": [...]}}

    Slot choice driven by ARC_PREPASS_SLOT env var:
      "1"    -> override attempt_1, keep model attempt_2 as fallback
      "2"    -> override attempt_2, keep model attempt_1
      "both" -> override both (max gain, max regression risk)
    Default: "1" (we have measured 0 false positives on full eval).
    """
    if traces_dir is None:
        traces_dir = os.path.dirname(os.path.abspath(__file__))
    slot_mode = os.getenv("ARC_PREPASS_SLOT", "1").strip().lower()
    if slot_mode not in ("1", "2", "both"):
        slot_mode = "1"
    solvers = _load_solvers(traces_dir)
    if verbose:
        print(f"[prepass] loaded {len(solvers)} solvers: {sorted(solvers.keys())}")
        print(f"[prepass] slot_mode={slot_mode}")
    n_overrides = 0
    n_considered = 0
    for tid, task in challenges.items():
        if tid not in submission:
            continue
        n_considered += 1
        result = _claim(task, solvers)
        if result is None:
            continue
        sname, test_outs, train_match = result
        n_train = len(task["train"])
        n_test = len(test_outs)
        new_attempts = []
        for i, ans in enumerate(test_outs):
            ans_list = [list(row) for row in ans]
            existing = submission[tid][i] if i < len(submission[tid]) else {}
            base_a1 = existing.get("attempt_1", ans_list)
            base_a2 = existing.get("attempt_2", base_a1)
            if slot_mode == "1":
                a1, a2 = ans_list, base_a2
            elif slot_mode == "2":
                a1, a2 = base_a1, ans_list
            else:  # both
                a1, a2 = ans_list, ans_list
            new_attempts.append({"attempt_1": a1, "attempt_2": a2})
        submission[tid] = new_attempts
        n_overrides += 1
        if verbose:
            print(f"[prepass] CLAIM tid={tid} solver={sname} train_match={train_match}/{n_train} test_pairs={n_test} slot={slot_mode}")
    if verbose:
        print(f"[prepass] SUMMARY considered={n_considered} overrides={n_overrides}")
    return submission
