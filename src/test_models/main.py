import pandas as pd
from codecarbon import OfflineEmissionsTracker
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from src.data_processing.feature_engineering import data_feature_engineering
from src.test_models.test_all import test_all_models
from joblib import dump, load


def main():
    tracker = OfflineEmissionsTracker(country_iso_code="ITA")

    tracker.start()

    df = pd.read_csv('../../data/user_feedback.csv')
    df = data_feature_engineering(df, mode='fit')

    best_model, best_params, mae, qwk = test_all_models(df, test_first_params=True, repeat=2)
    if best_model == "classifier":
        model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    elif best_model == "regressor":
        model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    elif best_model == "gradient regressor":
        model = GradientBoostingRegressor(**best_params, random_state=42)

    y = df['UserInput']
    X = df.drop(columns=['UserInput'])
    model.fit(X, y)
    dump(model, "../../model/model.joblib")

    tracker.stop()
    emissions_csv = pd.read_csv("emissions.csv")
    last_emissions = emissions_csv.tail(1)

    emissions = last_emissions["emissions"] * 1000
    energy = last_emissions["energy_consumed"]
    print(f"{emissions} Grams of CO2-equivalents")
    print(f"{energy} Sum of cpu_energy, gpu_energy and ram_energy (kWh)")


if __name__ == "__main__":
    main()
