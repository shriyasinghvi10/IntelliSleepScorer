r"""
Converts a SleepSign text/Excel export (Epoch No. + Stage columns, using
abbreviations like W / NR / R) into the CSV format calibration.py expects
(Epoch No. + Stage_Code, using the numeric codes from your parameters.csv).

USAGE:
    python convert_sleepsign_scores.py "D:\path\to\sleepsign_export.txt"

This creates a file next to it named "<original_name>_converted.csv",
ready to use directly with calibration.py -- either as-is (whole file) or
trimmed down in Excel to just the window you want to calibrate on.

Optional:
    python convert_sleepsign_scores.py "D:\path\to\sleepsign_export.txt" --output "D:\path\to\wherever.csv"
"""

import argparse
import os
import sys

import pandas as pd

# Common SleepSign stage abbreviations -> the full stage names used in
# parameters.csv. Add more entries here if your export uses different codes.
STAGE_ABBREVIATIONS = {
    "W": "Wake", "WAKE": "Wake",
    "NR": "NREM", "NREM": "NREM", "SWS": "NREM", "N": "NREM",
    "R": "REM", "REM": "REM",
}


def load_parameters(parameters_csv="./parameters.csv"):
    params = pd.read_csv(parameters_csv)
    p_stage = params[params["parameter"].isin(["Wake", "NREM", "REM"])]
    return p_stage.set_index("parameter").to_dict()["value"]  # {"Wake": 1, "NREM": 2, "REM": 3}


def _read_export_file(input_path):
    """Try a few common delimiter styles in order, since SleepSign exports
    and copy-pasted Excel content can use tabs, commas, or arbitrary runs of
    spaces for column alignment. Uses the first strategy that successfully
    parses two usable columns with no missing values."""
    strategies = [
        {"sep": "\t"},
        {"sep": ","},
        {"sep": r"\s{2,}", "engine": "python"},  # 2+ spaces: column separator, while "Epoch No." keeps its single internal space
        {"sep": r"\s+", "engine": "python"},
        {"sep": None, "engine": "python"},
    ]
    last_error = None
    for kwargs in strategies:
        try:
            df = pd.read_csv(input_path, **kwargs)
            df.columns = [str(c).strip() for c in df.columns]
            if df.shape[1] < 2:
                continue
            if df.isna().any().any():
                continue
            return df
        except Exception as e:
            last_error = e
            continue
    raise ValueError(
        f"Couldn't parse {input_path} with any recognized delimiter (tried tab, comma, "
        f"whitespace). Last error: {last_error}"
    )


def convert(input_path, output_path=None, parameters_csv="./parameters.csv"):
    stage_code_reverse = load_parameters(parameters_csv)

    df = _read_export_file(input_path)

    if "Epoch No." not in df.columns:
        # Be forgiving about slightly different header spelling/casing.
        candidates = [c for c in df.columns if "epoch" in c.lower()]
        if not candidates:
            raise ValueError(
                f"Couldn't find an 'Epoch No.' column. Found columns: {list(df.columns)}"
            )
        df = df.rename(columns={candidates[0]: "Epoch No."})

    if "Stage" not in df.columns:
        candidates = [c for c in df.columns if "stage" in c.lower()]
        if not candidates:
            raise ValueError(
                f"Couldn't find a 'Stage' column. Found columns: {list(df.columns)}"
            )
        df = df.rename(columns={candidates[0]: "Stage"})

    df["Stage"] = df["Stage"].astype(str).str.strip()
    stage_upper = df["Stage"].str.upper()

    unmapped = sorted(set(stage_upper) - set(STAGE_ABBREVIATIONS.keys()))
    if unmapped:
        raise ValueError(
            f"These stage labels in your export aren't recognized: {unmapped}\n"
            f"Recognized abbreviations: {sorted(set(STAGE_ABBREVIATIONS.keys()))}\n"
            "Add any missing ones to STAGE_ABBREVIATIONS at the top of this script."
        )

    full_stage_name = stage_upper.map(STAGE_ABBREVIATIONS)
    stage_code = full_stage_name.map(stage_code_reverse)

    if stage_code.isna().any():
        missing = sorted(set(full_stage_name[stage_code.isna()]))
        raise ValueError(
            f"These stage names aren't in your parameters.csv: {missing}. "
            "Check that parameters.csv has rows for Wake, NREM, and REM."
        )

    df_out = pd.DataFrame({
        "Epoch No.": df["Epoch No."].astype(int),
        "Stage_Code": stage_code.astype(int),
        "Stage": full_stage_name,
    })

    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_converted.csv"

    df_out.to_csv(output_path, index=False)

    print(f"Converted {len(df_out)} epochs.")
    print("\nStage counts:")
    print(df_out["Stage"].value_counts())
    print(f"\nSaved: {output_path}")
    print("\nThis file can be used directly with calibration.py, or trimmed in Excel "
          "first if you only want a specific window (e.g. one that includes REM bouts).")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert a SleepSign score export to calibration.py's CSV format.")
    parser.add_argument("input_path", help="Path to the SleepSign-exported text/csv file")
    parser.add_argument("--output", default=None, help="Output path (default: <input>_converted.csv)")
    parser.add_argument("--parameters", default="./parameters.csv", help="Path to parameters.csv (default: ./parameters.csv)")
    args = parser.parse_args()

    try:
        convert(args.input_path, args.output, args.parameters)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
