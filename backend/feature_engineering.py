import pandas as pd

def engineer_features(input_path, output_path):
    df = pd.read_csv(input_path)

    # Age feature
    today = pd.Timestamp.today()
    df['age'] = ((today - pd.to_datetime(df['birthdate']))
                 .dt.days / 365).astype(int)

    df = df[(df['age'] >= 0) & (df['age'] <= 100)]

    # Encounter type flags
    df['is_inpatient'] = (df['encounter_class'] == 'inpatient').astype(int)
    df['is_ambulatory'] = (df['encounter_class'] == 'ambulatory').astype(int)
    df['is_wellness'] = (df['encounter_class'] == 'wellness').astype(int)

    # Clinical burden
    df['clinical_burden'] = (
        df['has_allergy'] +
        df['has_observation']
    )

    # Save feature dataset
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    engineer_features(
        "data/processed/clean_data.csv",
        "data/processed/features.csv"
    )
