"""
End-to-end PS3 Door pipeline: train + validate on Train.csv, then predict
on Test.csv. Run: python run_all.py (from ps3_door/, with the repo venv
active).
"""

import pandas as pd

import config
import train_classify
import predict


def main():
    print("=" * 70)
    print("STEP 1-4: segmentation + classification training + validation")
    print("=" * 70)
    result, model, scaler, threshold = train_classify.run()

    seg = result["segmentation_eval"]
    print()
    print("--- Step 2: segmentation accuracy against Train_Segments_Answer.csv ---")
    print(f"{seg['exact_matches']}/{seg['n_true_segments']} true segments matched EXACTLY "
          f"(same start_time, end_time, row count) -- {seg['exact_match_fraction']:.1%}")
    print(f"{seg['tolerant_matches']}/{seg['n_true_segments']} matched within "
          f"{config.SEGMENTATION_TOLERANCE_SEC}s tolerance -- {seg['tolerant_match_fraction']:.1%}")
    print(f"({seg['n_detected_segments']} segments detected total, vs {seg['n_true_segments']} true)")

    print()
    print("--- Step 4: classification on held-out validation segments ---")
    print(f"held out: {result['n_val_normal']} Normal + {result['n_val_abnormal']} Abnormal resistance "
          f"(all Abnormal segments in Train, since the autoencoder never trains on them)")
    print(f"threshold: p{result['threshold_percentile']} of held-out Normal recon error = {result['threshold']:.5f}")
    print(f"precision={result['precision']:.3f}  recall={result['recall']:.3f}  f1={result['f1']:.3f}")
    cm = result["confusion_matrix"]
    print(f"confusion matrix [true \\ pred]      Normal  Abnormal")
    print(f"  Normal                            {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"  Abnormal resistance                {cm[1][0]:6d}  {cm[1][1]:6d}")

    print()
    print("=" * 70)
    print("STEP 5: predicting on Test.csv")
    print("=" * 70)
    preds = predict.run()

    print()
    print("--- Format check against 04_Example_Submission/door_predictions.csv ---")
    example = pd.read_csv(
        "/Users/gsreetham/Desktop/NebulaX-Hackathon-ProblemStatement/PS3/04_Example_Submission/door_predictions.csv"
    )
    ours_cols = pd.read_csv(config.PREDICTIONS_PATH, nrows=0).columns.tolist()
    print(f"our columns:     {ours_cols}")
    print(f"example columns: {example.columns.tolist()}")
    print(f"columns match exactly: {ours_cols == example.columns.tolist()}")
    print(f"no file_id column: {'file_id' not in ours_cols}")
    print(f"one row per predicted segment: {len(preds)} rows")
    print(f"prediction values used: {sorted(preds['prediction'].unique())}")
    print(f"output written to: {config.PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
