# evaluation_script/evaluation.py
import os
import sys
import json
import tempfile
import zipfile
import importlib.util
from typing import Any, Dict, Optional, List, Tuple
import numpy as np



# ------------------- helpers -------------------

def _extract_submission(zip_path: str) -> str:
    """Unzip the participant submission to a temp dir and return its path."""
    tmpdir = tempfile.mkdtemp(prefix="evalai_subm_")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmpdir)
    return tmpdir


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

def _import_subm_module(subm_dir: str,
                        module_name_candidates=("run_test_submit", "submission", "model")):
    """
    Try to import the participant's module from which we will fetch:
      - SmallCNN
      - load_model
      - load_batch
      - softmax
    """
    sys.path.insert(0, subm_dir)

    for mod_name in module_name_candidates:
        # 1) 普通 import
        try:
            mod = importlib.import_module(mod_name)
            return mod
        except Exception:
            pass

        # 2) 文件路径导入
        py_path = os.path.join(subm_dir, f"{mod_name}.py")
        if os.path.isfile(py_path):
            spec = importlib.util.spec_from_file_location(mod_name, py_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore
                return mod

    raise ImportError(
        "Could not import participant module. "
        "Please put your code in 'run_test_submit.py' (or 'submission.py' / 'model.py')."
    )


def _list_images(root: str) -> Tuple[List[Tuple[str, int]], List[str]]:
    """
    Simple helper to mimic list_images(root) used in the project:

    root/
      class0/*.jpg
      class1/*.jpg
      ...

    Returns:
      items: list of (path, class_index)
      classes: list of class names (sorted)
    """
    classes = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )
    items: List[Tuple[str, int]] = []
    for ci, cname in enumerate(classes):
        cdir = os.path.join(root, cname)
        for fn in os.listdir(cdir):
            fp = os.path.join(cdir, fn)
            if os.path.isfile(fp) and fp.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                items.append((fp, ci))
    return items, classes





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
    EvalAI entrypoint.

    New logic:
      - Extract submission
      - Import participant module
      - Use their:
          * SmallCNN
          * load_model
          * load_batch
          * softmax
        to build our own evaluation loop.
      - We will:
          * load data_root/val
          * build (path, label) list
          * SHUFFLE the order with a fixed random seed
          * run inference batch by batch
          * compute accuracy on our side
    """
    print("Starting Evaluation...")
    print(f"Phase codename: {phase_codename}")
    print(f"Host annotation (test_annotation_file): {test_annotation_file}")
    print(f"User submission (user_annotation_file): {user_annotation_file}")

    # 1) Unzip submission
    subm_dir = _extract_submission(user_annotation_file)
    print(f"Extracted submission to: {subm_dir}")

    # 2) Import participant module
    subm_mod = _import_subm_module(subm_dir)

    # 3) Fetch required symbols from participant code
    SmallCNN = getattr(subm_mod, "SmallCNN", None)
    load_model = getattr(subm_mod, "load_model", None)
    load_batch = getattr(subm_mod, "load_batch", None)
    softmax = getattr(subm_mod, "softmax", None)

    missing = [name for name, obj in [
        ("SmallCNN", SmallCNN),
        ("load_model", load_model),
        ("load_batch", load_batch),
        ("softmax", softmax),
    ] if obj is None]

    if missing:
        raise ImportError(
            f"Missing required symbols in submission module: {missing}. "
            f"Please make sure these are defined (they exist in the project template)."
        )

    # 4) Resolve dataset root
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = _find_dataset_root(eval_dir)
    print(f"Using data_root: {data_root}")

    # 5) Detect checkpoint
    ckpt_path = _guess_checkpoint_path(subm_dir)
    if ckpt_path:
        print(f"Detected checkpoint: {ckpt_path}")
    else:
        ckpt_path = os.path.join(subm_dir, "checkpoints", "model_final.npz")
        print(f"No checkpoint auto-detected; fallback to {ckpt_path}")

    # 6) Load model via participant's load_model
    img_size = 100
    strict = True
    model, meta, report = load_model(
        ckpt_path,
        model_ctor=SmallCNN,
        ctor_kwargs={"img_size": img_size},
        strict=strict,
    )

    # 7) Prepare test/val set (we use 'val' as test split)
    test_dir = os.path.join(data_root, "val")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test (val) directory not found: {test_dir}")

    # We assume labeled subfolders: val/classA/...
    test_items, classes_disk = _list_images(test_dir)

    if not test_items:
        raise RuntimeError(f"No images found under {test_dir}")

    # If class meta is present in checkpoint, respect its order
    classes_meta = meta.get("classes", None) if isinstance(meta, dict) else None
    if classes_meta:
        classes = classes_meta
    else:
        classes = classes_disk

    cls_to_idx = {c: i for i, c in enumerate(classes)}

    # Build y_true according to our class order
    y_true = []
    for path, _ in test_items:
        class_name = os.path.basename(os.path.dirname(path))
        if class_name not in cls_to_idx:
            raise ValueError(f"Class '{class_name}' not found in class list {classes}")
        y_true.append(cls_to_idx[class_name])
    y_true = np.array(y_true, dtype=np.int32)

    print(f"Test samples: {len(test_items)}")
    print(f"Classes ({len(classes)}): {classes}")

    # 8) Shuffle test_items + y_true with a fixed seed (to prevent order-based cheating)
    rng = np.random.RandomState(2025)
    indices = np.arange(len(test_items))
    rng.shuffle(indices)

    test_items_shuf = [test_items[i] for i in indices]
    y_true_shuf = y_true[indices]

    # 9) Inference loop using participant's load_batch + model.forward
    batch_size = 64
    logits_list = []
    for start in range(0, len(test_items_shuf), batch_size):
        end = min(start + batch_size, len(test_items_shuf))
        batch = test_items_shuf[start:end]
        # participant's load_batch(paths_labels, start, end, img_size, augment=False)
        X, _ = load_batch(batch, 0, len(batch), img_size, augment=False)
        # run forward pass
        logits, _ = model.forward(X)
        logits_list.append(logits)

    logits = np.concatenate(logits_list, axis=0)
    probs = softmax(logits)
    preds = np.argmax(probs, axis=1)

    # 10) Compute accuracy
    if len(preds) != len(y_true_shuf):
        raise RuntimeError(f"Prediction length {len(preds)} != ground truth length {len(y_true_shuf)}")

    acc = float((preds == y_true_shuf).mean())
    print(f"📊 Shuffled test accuracy: {acc:.4f}")

    # 11) Build EvalAI payload
    result_payload = [{"test_split": {"Accuracy": float(acc)}}]
    output = {
        "result": result_payload,
        "submission_result": result_payload[0]["test_split"],
    }
    print(f"Completed Evaluation. Accuracy = {acc:.4f}")
    return output
