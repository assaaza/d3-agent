import sys, json
sys.path.insert(0, "tools")
from d3_preprocess import load_csv, profile, run_preprocessing, to_dictionary_md

def summarize(name, path):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    df = load_csv(path)
    prof = profile(df)
    print("row_count:", prof["row_count"], "col_count:", prof["col_count"])
    print("trimmable_headers:", prof["trimmable_headers"])
    print("empty_rows:", prof["empty_rows"], "empty_columns:", prof["empty_columns"])
    print("duplicate_rows:", prof["duplicate_rows"])
    print("constant_columns:", prof["constant_columns"])
    print("numeric_columns:", [c["col"] for c in prof["numeric_columns"]])
    print("mixed_columns:", prof["mixed_columns"])
    print("missing_cells:", prof["missing_cells"])
    print("outliers:", {k: {"lo": round(v["lo"],2), "hi": round(v["hi"],2), "n_rows": len(v["rows"])} for k,v in prof["outliers"].items()})
    print("reshape:", prof["reshape"])
    print("reshape_ambiguous:", prof["reshape_ambiguous"])

    clean_df, log, dict_notes = run_preprocessing(df, prof)
    print("\n--- log ---")
    for l in log:
        print(" -", l)
    print("\n--- clean_df ---")
    print(clean_df.head(10).to_string())
    print("shape:", clean_df.shape)
    print("\n--- dictionary notes ---")
    print(json.dumps(dict_notes, ensure_ascii=False, indent=2))
    return clean_df, log, dict_notes

summarize("HEALTH - obesity", "data/raw/health-obesity/obesity_prevalence_1998_2024.csv")
summarize("HOUSEHOLDS - one person", "data/raw/one-person-households/one_person_household_ratio_2000_2025.csv")
