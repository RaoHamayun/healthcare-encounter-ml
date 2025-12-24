import pandas as pd
from pathlib import Path

# ==================================================
# CONFIG
# ==================================================
RAW_DATA_PATH = Path("./data/raw")
OUTPUT_PATH = Path("./data/processed")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_PATH / "final_healthcare_dataset.csv"

# ==================================================
# SCHEMA
# ==================================================
SCHEMA = {
    "patient": {
        "file": "patients.csv",
        "pk": ["patient_id"],
        "columns": {
            "Id": "patient_id",
            "BIRTHDATE": "birthdate",
            "FIRST": "first_name",
            "LAST": "last_name",
            "MARITAL": "marital_status",
            "RACE": "race",
            "ETHNICITY": "ethnicity",
            "GENDER": "gender",
        },
    },
    "encounter": {
        "file": "encounters.csv",
        "pk": ["patient_id", "encounter_id"],
        "columns": {
            "Id": "encounter_id",
            "PATIENT": "patient_id",
            "ENCOUNTERCLASS": "encounter_class",
            "CODE": "encounter_code",
            "DESCRIPTION": "encounter_description",
        },
    },
    "conditions": {
        "file": "conditions.csv",
        "fk": ["patient_id", "encounter_id"],
        "columns": {
            "PATIENT": "patient_id",
            "ENCOUNTER": "encounter_id",
            "CODE": "condition_code",
            "DESCRIPTION": "condition_description",
        },
    },
    "allergies": {
        "file": "allergies.csv",
        "fk": ["patient_id", "encounter_id"],
        "columns": {
            "PATIENT": "patient_id",
            "ENCOUNTER": "encounter_id",
            "CODE": "allergy_code",
            "DESCRIPTION": "allergy_description",
            "TYPE": "allergy_type",
            "CATEGORY": "allergy_category",
        },
    },
    "observations": {
        "file": "observations.csv",
        "fk": ["patient_id", "encounter_id"],
        "columns": {
            "PATIENT": "patient_id",
            "ENCOUNTER": "encounter_id",
            "CATEGORY": "observation_category",
            "CODE": "observation_code",
            "DESCRIPTION": "observation_description",
        },
    },
    "medications": {
        "file": "medications.csv",
        "fk": ["patient_id", "encounter_id"],
        "columns": {
            "PATIENT": "patient_id",
            "ENCOUNTER": "encounter_id",
            "CODE": "medication_code",
            "DESCRIPTION": "medication_description",
        },
    },
    "procedures": {
        "file": "procedures.csv",
        "fk": ["patient_id", "encounter_id"],
        "columns": {
            "PATIENT": "patient_id",
            "ENCOUNTER": "encounter_id",
            "CODE": "procedure_code",
            "DESCRIPTION": "procedure_description",
        },
    },
    "careplans": {
        "file": "careplans.csv",
        "fk": ["patient_id", "encounter_id"],
        "columns": {
            "PATIENT": "patient_id",
            "ENCOUNTER": "encounter_id",
            "CODE": "careplan_code",
            "DESCRIPTION": "careplan_description",
        },
    },
}

# ==================================================
# FUNCTIONS
# ==================================================
def load_dataframe(name, cfg, raw_path=RAW_DATA_PATH):
    """Load CSV, select schema columns, rename, and return DataFrame"""
    path = raw_path / cfg["file"]
    df = pd.read_csv(path, usecols=cfg["columns"].keys())
    df = df.rename(columns=cfg["columns"])
    print(f"{name:<12} loaded {df.shape}")
    return df

def impute_missing(df, strategy=None):
    """Impute missing values in a DataFrame"""
    for col in df.columns:
        if strategy and col in strategy:
            df[col] = df[col].fillna(strategy[col])
        else:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].fillna(pd.Timestamp("1900-01-01"))
            else:
                df[col] = df[col].fillna("Unknown")
    return df

def check_dataframe(df, name):
    """Print basic info and sample data for a DataFrame"""
    print(f"\n=== {name.upper()} ===")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nSample rows:")
    print(df.head(3))
    print("-"*50)

def preprocess_dataframe(name, cfg, strategy=None, check=True):
    """Full preprocessing: load, impute, deduplicate"""
    df = load_dataframe(name, cfg)
    df = impute_missing(df, strategy)
    if "pk" in cfg:
        df = df.drop_duplicates(subset=cfg["pk"])
    if check:
        check_dataframe(df, name)
    return df

def aggregate_child_table(df, keys, agg_columns):
    """Aggregate child tables to have one row per encounter"""
    agg_map = {col: lambda x: ", ".join(sorted(set(x.dropna().astype(str)))) for col in agg_columns}
    return df.groupby(keys, as_index=False).agg(agg_map)

# ==================================================
# LOAD AND PREPROCESS ALL DATAFRAMES
# ==================================================
dfs = {name: preprocess_dataframe(name, cfg) for name, cfg in SCHEMA.items()}

# ==================================================
# AGGREGATE CHILD TABLES
# ==================================================
child_tables = ["conditions", "allergies", "observations", "medications", "procedures", "careplans"]
for table in child_tables:
    cfg = SCHEMA[table]
    agg_columns = list(cfg["columns"].values())
    dfs[table] = aggregate_child_table(dfs[table], keys=cfg["fk"], agg_columns=agg_columns)

# ==================================================
# MERGE CORE + CHILD TABLES
# ==================================================
final_df = dfs["encounter"].merge(dfs["patient"], on="patient_id", how="left")
for table in child_tables:
    final_df = final_df.merge(dfs[table], on=["patient_id","encounter_id"], how="left")

# ==================================================
# VALIDATION
# ==================================================
duplicates = final_df.duplicated(subset=["patient_id","encounter_id"]).sum()
print(f"\nDuplicate encounters: {duplicates}")
assert duplicates == 0, "❌ Duplicate encounters found!"

print("\n✅ Final dataset valid")
print("Final shape:", final_df.shape)

# ==================================================
# SAVE FINAL DATASET
# ==================================================
final_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n📁 Saved to: {OUTPUT_FILE}")

check_dataframe(final_df, "final_healthcare_dataset")
