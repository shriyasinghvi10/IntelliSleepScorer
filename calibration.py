r"""
Calibrates a pretrained IntelliSleepScorer model to one specific recording,
using a small pre-scored (manually labeled) segment of that same recording.

WHY: the pretrained models learned decision boundaries from whatever rig
originally recorded their training data. A different recording device
(different amplifier gain, electrode placement, signal scale) can shift
where "Wake" vs "REM" actually sit in feature-space, without the underlying
sleep biology being any different. Rather than guessing at a fix, this
takes real labeled examples from THIS recording and continues training the
existing model on them -- adapting its decision boundaries to this
recording's actual signal characteristics, without discarding what it
already learned from the original (much larger) training set.

WORKFLOW:
1. You must have already scored this file once in IntelliSleepScorer (so
   the *_features.csv file already exists for it).
2. You provide a small CSV of manual scores covering some portion of the
   SAME recording -- ideally including examples of Wake, NREM, and REM.
   Required columns: "Epoch No." and "Stage_Code" (1=Wake, 2=NREM, 3=REM,
   matching parameters.csv). If you only have "Stage" (text: Wake/NREM/REM)
   that's also accepted.
3. This script continues training the pretrained model on just those
   labeled epochs, saves the calibrated model specific to this recording,
   and re-scores the entire file with it.

USAGE:
    python calibration.py "D:\path\to\recording.edf" "D:\path\to\manual_scores_segment.csv" --model 2_LightGBM-1EEG

Optional:
    --epoch_length 10       (must match what you used when scoring; auto-detected if omitted)
    --num_boost_round 50    (how many additional trees to fit on the calibration segment; default 50)
"""

import argparse
import os
import sys

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from calibrated_model import CalibratedModel


def load_parameters(parameters_csv="./parameters.csv"):
    params = pd.read_csv(parameters_csv)
    p_stage = params[params["parameter"].isin(["Wake", "NREM", "REM"])]
    stage_code = p_stage.set_index("value").to_dict()["parameter"]           # {1: "Wake", ...}
    stage_code_reverse = p_stage.set_index("parameter").to_dict()["value"]   # {"Wake": 1, ...}
    return stage_code, stage_code_reverse


def find_features_csv(folder, file_firstname, model_name, epoch_length):
    if epoch_length is not None:
        path = os.path.join(folder, f"{file_firstname}_{model_name}_epoch_length_{epoch_length}_sec_features.csv")
        if os.path.exists(path):
            return path
        return None
    import glob
    pattern = os.path.join(folder, f"{file_firstname}_{model_name}_epoch_length_*_sec_features.csv")
    matches = glob.glob(pattern)
    if not matches:
        return None
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def calibrate_and_rescore(edf_path, calibration_csv_path, model_name, epoch_length=None, num_boost_round=50, min_child_samples=None):
    stage_code, stage_code_reverse = load_parameters()

    folder = os.path.dirname(os.path.abspath(edf_path)) + os.sep
    file_firstname = os.path.basename(edf_path).split(".edf")[0]

    features_csv_path = find_features_csv(folder, file_firstname, model_name, epoch_length)
    if features_csv_path is None:
        raise FileNotFoundError(
            f"Could not find a features CSV for '{file_firstname}' with model '{model_name}' in {folder}. "
            "Score this file once in IntelliSleepScorer first (so features get extracted), then run calibration."
        )
    print(f"Using features file: {features_csv_path}")

    df_features = pd.read_csv(features_csv_path)
    # Same column slicing IntelliSleepScorer.py uses: excludes the leading
    # index column and the trailing epoch_id / subject_id / score columns.
    feature_columns = df_features.columns[1:-3].tolist()

    # --- Load the calibration (manually scored) segment ---
    df_calib = pd.read_csv(calibration_csv_path)
    if "Epoch No." not in df_calib.columns:
        raise ValueError('Calibration CSV must have an "Epoch No." column.')

    if "Stage_Code" not in df_calib.columns:
        if "Stage" in df_calib.columns:
            df_calib["Stage_Code"] = df_calib["Stage"].map(stage_code_reverse)
            if df_calib["Stage_Code"].isna().any():
                bad = df_calib.loc[df_calib["Stage_Code"].isna(), "Stage"].unique()
                raise ValueError(
                    f"Some values in the 'Stage' column don't match parameters.csv (Wake/NREM/REM): {bad}"
                )
        else:
            raise ValueError('Calibration CSV must have either a "Stage_Code" or "Stage" column.')

    invalid_epochs = df_calib.loc[~df_calib["Epoch No."].isin(df_features.index), "Epoch No."]
    if len(invalid_epochs) > 0:
        raise ValueError(
            f"These epoch numbers in the calibration CSV don't exist in this recording's features "
            f"(out of range): {sorted(invalid_epochs.tolist())[:10]}{'...' if len(invalid_epochs) > 10 else ''}"
        )

    print(f"Calibration segment: {len(df_calib)} labeled epochs")
    print("Stage counts in calibration segment:")
    print(df_calib["Stage_Code"].map(stage_code).value_counts())
    missing_stages = set(stage_code.values()) - set(df_calib["Stage_Code"].map(stage_code).unique())
    if missing_stages:
        print(f"\nWARNING: your calibration segment has no examples of: {', '.join(missing_stages)}. "
              "Calibration will only be able to correct the boundaries it has examples for -- "
              "it's best if the segment includes all three stages.\n")

    X_calib = df_features.loc[df_calib["Epoch No."], feature_columns]
    y_calib_raw = df_calib["Stage_Code"].values

    # --- Load the pretrained model and continue training on the segment ---
    model_path = f"./models/{model_name}.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pretrained model not found: {model_path}")
    original_model = joblib.load(model_path)

    if not hasattr(original_model, "booster_") or not hasattr(original_model, "classes_"):
        raise TypeError(
            f"'{model_name}' doesn't look like the expected scikit-learn LightGBM classifier "
            "(missing .booster_ / .classes_). Calibration wasn't built for this model type."
        )

    original_booster = original_model.booster_
    original_classes = original_model.classes_  # e.g. array([1, 2, 3])

    class_to_index = {cls: i for i, cls in enumerate(original_classes)}
    unknown_codes = set(y_calib_raw) - set(class_to_index.keys())
    if unknown_codes:
        raise ValueError(
            f"Calibration segment has Stage_Code value(s) {unknown_codes} that the original model "
            f"doesn't recognize (it knows: {list(original_classes)}). Check parameters.csv matches."
        )
    y_calib_encoded = np.array([class_to_index[code] for code in y_calib_raw])

    print(f"\nContinuing training from the pretrained model for {num_boost_round} additional rounds...")
    # IMPORTANT: the pretrained booster's saved params include 'num_iterations'
    # (set to however many rounds it was originally trained for). If left in,
    # LightGBM treats that as the TOTAL target iteration count for training,
    # which the init_model has already reached -- so it silently adds ZERO
    # additional trees no matter what num_boost_round says. Strip it so
    # num_boost_round is what actually controls how much more training happens.
    continued_training_params = {
        k: v for k, v in original_booster.params.items()
        if k not in ("num_iterations", "n_estimators", "num_boost_round")
    }

    # IMPORTANT #2: the pretrained model's min_child_samples (commonly 20,
    # tuned for a much larger original training set) requires at least that
    # many samples per leaf. A realistic calibration segment (tens of
    # epochs, split across 3 stages) usually can't satisfy that -- LightGBM
    # then silently rejects every split ("No further splits with positive
    # gain") and adds effectively zero useful trees, with no visible error.
    # Auto-scale it to the calibration segment size unless overridden.
    if min_child_samples is None:
        smallest_class_count = pd.Series(y_calib_encoded).value_counts().min()
        min_child_samples = max(1, min(5, smallest_class_count // 2))
    continued_training_params["min_child_samples"] = min_child_samples
    continued_training_params["min_data_in_leaf"] = min_child_samples
    print(f"Using min_child_samples={min_child_samples} for calibration (scaled to your segment size)")

    train_set = lgb.Dataset(X_calib, label=y_calib_encoded, free_raw_data=False)
    calibrated_booster = lgb.train(
        params=continued_training_params,
        train_set=train_set,
        num_boost_round=num_boost_round,
        init_model=original_booster,
    )
    trees_before = original_booster.num_trees()
    trees_after = calibrated_booster.num_trees()
    print(f"Done. Tree count: {trees_before} -> {trees_after} (added {trees_after - trees_before})")
    if trees_after == trees_before:
        print("WARNING: no new trees were added -- calibration likely had no effect. "
              "Try a larger calibration segment, or pass --min_child_samples 1 to force smaller splits.")

    calibrated_model = CalibratedModel(calibrated_booster, original_classes)

    calibrated_model_path = os.path.join(folder, f"{file_firstname}_{model_name}_calibrated.pkl")
    joblib.dump(calibrated_model, calibrated_model_path)
    print(f"\nSaved calibrated model: {calibrated_model_path}")

    # --- Re-score the whole file with the calibrated model ---
    X_all = df_features[feature_columns]
    y_prediction = calibrated_model.predict(X_all)

    df_output = pd.DataFrame({
        "Epoch No.": list(range(len(y_prediction))),
        "Stage_Code": y_prediction.astype(int),
    })
    df_output["Stage"] = df_output["Stage_Code"].map(stage_code)

    epoch_length_str = features_csv_path.split("epoch_length_")[1].split("_sec_features")[0]
    scores_path = os.path.join(
        folder, f"{file_firstname}_{model_name}_calibrated_epoch_length_{epoch_length_str}_sec_scores.csv"
    )
    df_output.to_csv(scores_path, index=False)
    print(f"Saved calibrated scores: {scores_path}")

    print("\nStage distribution, calibrated vs pretrained-only:")
    original_prediction = original_model.predict(X_all)
    comparison = pd.DataFrame({
        "Pretrained": pd.Series(original_prediction).map(stage_code).value_counts(),
        "Calibrated": pd.Series(y_prediction).map(stage_code).value_counts(),
    })
    print(comparison)

    return scores_path, calibrated_model_path


def main():
    parser = argparse.ArgumentParser(description="Calibrate a model to one recording using a pre-scored segment.")
    parser.add_argument("edf_path", help="Path to the .edf file")
    parser.add_argument("calibration_csv", help="CSV with manually-scored epochs (columns: 'Epoch No.', and 'Stage_Code' or 'Stage')")
    parser.add_argument("--model", required=True, help="Model name, e.g. 2_LightGBM-1EEG")
    parser.add_argument("--epoch_length", default=None, help="Epoch length in seconds (auto-detected if omitted)")
    parser.add_argument("--num_boost_round", type=int, default=50, help="Additional boosting rounds to fit on the calibration segment (default 50)")
    parser.add_argument("--min_child_samples", type=int, default=None,
                         help="Minimum samples per leaf during calibration (default: auto, scaled to your calibration segment size)")
    args = parser.parse_args()

    try:
        calibrate_and_rescore(
            edf_path=args.edf_path,
            calibration_csv_path=args.calibration_csv,
            model_name=args.model,
            epoch_length=args.epoch_length,
            num_boost_round=args.num_boost_round,
            min_child_samples=args.min_child_samples,
        )
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
