import os, json
from evaluation_script.main import evaluate  

def test_local_evaluation(
    submission_zip="submission.zip",
    dataset_root="dataset",
    test_annotations="dataset",
    phase_codename="test",
):
    """
    Simulate EvalAI evaluation locally.
    - submission_zip: path to student's submission zip
    - dataset_root: dataset directory (e.g. dataset/)
    - test_annotations: path to dummy or real annotation JSON
    - phase_codename: "test" by default
    """
    print("=== Local EvalAI evaluation simulation ===")
    print(f"Submission file: {submission_zip}")
    print(f"Dataset root: {dataset_root}")
    print(f"Phase: {phase_codename}")

    if not os.path.exists(submission_zip):
        raise FileNotFoundError(f"Submission zip not found: {submission_zip}")
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")

    # Create a dummy test_annotation_file if missing
    if not os.path.exists(test_annotations):
        os.makedirs(os.path.dirname(test_annotations), exist_ok=True)
        dummy = {"info": "Dummy annotation file for local test"}
        with open(test_annotations, "w", encoding="utf-8") as f:
            json.dump(dummy, f)
        print(f"Created dummy annotation file at {test_annotations}")

    # Run the evaluate() function
    print("\n>>> Running evaluate() ...")
    result = evaluate(
        test_annotation_file=test_annotations,
        user_annotation_file=submission_zip,
        phase_codename=phase_codename,
    )

    print("\n=== Evaluation Output ===")
    print(json.dumps(result, indent=2))
    return result

# Example usage
res = test_local_evaluation(
    submission_zip="submission.zip",
    dataset_root="dataset",
    phase_codename="test"
)
