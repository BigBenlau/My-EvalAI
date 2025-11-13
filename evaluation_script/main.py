# evaluation_script/evaluation.py
import os
import sys
import json
import tempfile
import zipfile
import importlib.util
from typing import Any, Dict, Optional


# ------------------- helpers -------------------

def _extract_submission(zip_path: str) -> str:
    """Unzip the participant submission to a temp dir and return its path."""
    tmpdir = tempfile.mkdtemp(prefix="evalai_subm_")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmpdir)
    return tmpdir


def _import_func_from(subm_dir: str,
                      module_name_candidates=("run_test_submit", "submission", "model"),
                      func_name="run_test_submit"):
    """
    Import run_test_submit(data_root, img_size, batch_size, load_path, model_cls, strict)
    from the participant's submission. We try both module import and file-based import.
    """
    sys.path.insert(0, subm_dir)  # allow "import x" where x is in submission root

    # 1) Try normal import by module name
    for mod_name in module_name_candidates:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, func_name):
                return getattr(mod, func_name)
        except Exception:
            pass

        # 2) Try file-based import if a .py exists
        py_path = os.path.join(subm_dir, f"{mod_name}.py")
        if os.path.isfile(py_path):
            spec = importlib.util.spec_from_file_location(mod_name, py_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore
                if hasattr(mod, func_name):
                    return getattr(mod, func_name)

    raise ImportError(
        "Could not find function 'run_test_submit' in the submission. "
        "Place it in 'run_test_submit.py' (or 'submission.py' / 'model.py') "
        "with signature run_test_submit(data_root, img_size, batch_size, load_path, model_cls, strict)."
    )


def _guess_checkpoint_path(subm_dir: str) -> Optional[str]:
    """
    Try to locate a checkpoint file (.npz, .pt, .pth) under the submission folder.

    Updated logic:
        - In checkpoints/: 
            * if multiple .npz files exist, choose the most recently modified one.
            * otherwise still accept .pt / .pth in that directory if present.
        - Only fallback to submission root if nothing found in checkpoints/.

    Returns:
        Path to selected checkpoint file, or None if nothing found.
    """
    patterns = [".npz", ".pt", ".pth"]
    ckpt_dir = os.path.join(subm_dir, "checkpoints")

    # ------------------------
    # 1) First: search in checkpoints/
    # ------------------------
    if os.path.isdir(ckpt_dir):
        npz_files = []
        other_ckpts = []

        for fname in os.listdir(ckpt_dir):
            full = os.path.join(ckpt_dir, fname)
            if not os.path.isfile(full):
                continue

            if fname.endswith(".npz"):
                npz_files.append(full)
            elif any(fname.endswith(ext) for ext in [".pt", ".pth"]):
                other_ckpts.append(full)

        # a) Prefer .npz files, pick newest by mtime
        if npz_files:
            npz_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return npz_files[0]

        # b) Otherwise accept .pt/.pth (first match)
        if other_ckpts:
            return other_ckpts[0]

    # ------------------------
    # 2) Fallback: search in submission root
    # ------------------------
    for fname in os.listdir(subm_dir):
        if any(fname.endswith(ext) for ext in patterns):
            return os.path.join(subm_dir, fname)

    return None




def _find_dataset_root(eval_script_dir: str) -> str:
    """
    Resolve dataset root. By default we expect a folder named 'dataset'
    next to the evaluation_script directory. Allow override via env.
    """
    default_ds = os.path.abspath(os.path.join(eval_script_dir, "..", "dataset"))
    return os.environ.get("EVAL_DATA_ROOT", default_ds)


# ------------------- required entrypoint -------------------

def evaluate(test_annotation_file: str,
             user_annotation_file: str,
             phase_codename: str,
             **kwargs) -> Dict[str, Any]:
    """
    EvalAI entrypoint. We expect participants to submit a ZIP containing:
      - a Python file that defines `run_test_submit(...)`
      - their model code and checkpoint (e.g., checkpoints/model_final.npz)

    We call run_test_submit(...) to compute accuracy on the test/val set
    and return a payload with a single metric "Accuracy" under "test_split".
    """
    print("Starting Evaluation...")
    print(f"Phase codename: {phase_codename}")
    print(f"Host annotation (test_annotation_file): {test_annotation_file}")
    print(f"User submission (user_annotation_file): {user_annotation_file}")

    # Unzip participant submission
    subm_dir = _extract_submission(user_annotation_file)
    print(f"Extracted submission to: {subm_dir}")

    # Import participant's runner function
    run_test_submit = _import_func_from(subm_dir)

    # Resolve dataset root (by default ../dataset relative to evaluation_script/)
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = _find_dataset_root(eval_dir)
    print(f"Using data_root: {data_root}")

    # Detect checkpoint path (best-effort)
    ckpt_path = _guess_checkpoint_path(subm_dir)
    if ckpt_path:
        print(f"Detected checkpoint: {ckpt_path}")
    else:
        print("No checkpoint auto-detected; relying on participant's default inside run_test_submit.")

    # Default args (participants can ignore/override inside their function)
    args = {
        "data_root": data_root,
        "img_size": 100,
        "batch_size": 64,
        "load_path": ckpt_path if ckpt_path else "checkpoints/model_final.npz",
        "model_cls": None,
        "strict": True,
    }

    # Invoke participant runner; expect a float accuracy in [0,1] (or None if unlabeled)
    try:
        acc = run_test_submit(**args)
    except TypeError:
        # Fallback in case of older signature
        acc = run_test_submit(
            data_root=data_root,
            img_size=100,
            batch_size=64,
            load_path=args["load_path"],
            model_cls=None,
            strict=True,
        )

    if acc is None:
        print("No accuracy returned (possibly unlabeled test). Setting Accuracy = 0.0")
        acc = 0.0

    # Build EvalAI-compliant output: single split "test_split" with one metric "Accuracy"
    result_payload = [{"test_split": {"Accuracy": float(acc)}}]
    output = {
        "result": result_payload,
        "submission_result": result_payload[0]["test_split"],  # appears in UI
    }

    print(f"Completed Evaluation. Accuracy = {acc:.4f}")
    return output