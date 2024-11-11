import pandas as pd
from codecarbon import OfflineEmissionsTracker  # We import the emission tracker
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor

from src.data_processing.feature_engineering import data_feature_engineering
from src.test_models.test_all import test_all_models
from src.test_models.test_gradient_boosting_regressor import test_gradient_boosting_regressor
from src.test_models.test_random_forest_classifier import test_random_forest_classifier
from src.test_models.test_random_forest_regressor import test_random_forest_regressor
from joblib import dump, load


def main():
    tracker = OfflineEmissionsTracker(country_iso_code="ITA")

    tracker.start()

    # Load your data
    df = pd.read_csv('../../data/user_feedback.csv')

    df = data_feature_engineering(df, mode='fit')

    best_model, best_params, mae, qwk = test_all_models(df, test_first_params=True, repeat=2)
    model = None
    #TODO test which model is best
    best_model = "regressor"

    if best_model == "classifier":
        model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    elif best_model == "regressor":
        model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    elif best_model == "gradient regressor":
        model = GradientBoostingRegressor(**best_params, random_state=42)

    y = df['UserInput']  # Target column
    X = df.drop(columns=['UserInput'])  # Feature columns
    model.fit(X, y)
    dump(model, "../../model/model.joblib")



    tracker.stop()

    emissions_csv = pd.read_csv("emissions.csv")
    last_emissions = emissions_csv.tail(1)

    emissions = last_emissions["emissions"] * 1000
    energy = last_emissions["energy_consumed"]
    cpu = last_emissions["cpu_energy"]
    gpu = last_emissions["gpu_energy"]
    ram = last_emissions["ram_energy"]

    print(f"{emissions} Grams of CO2-equivalents")
    print(f"{energy} Sum of cpu_energy, gpu_energy and ram_energy (kWh)")
    print(f"{cpu} Energy used per CPU (kWh)")
    print(f"{gpu} Energy used per GPU (kWh)")
    print(f"{ram} Energy used per RAM (kWh)")



if __name__ == "__main__":
    main()
