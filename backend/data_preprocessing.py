import pandas as pd

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)

    # Convert birthdate
    df['birthdate'] = pd.to_datetime(df['birthdate'], errors='coerce')

    # Drop rows missing critical fields
    df = df.dropna(subset=[
        'birthdate',
        'encounter_class',
        'has_condition'
    ])

    # Save cleaned data
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess_data(
        "data/processed/actual_data.csv",
        "data/processed/clean_data.csv"
    )
