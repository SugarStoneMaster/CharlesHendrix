import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
from src.data_processing.feature_engineering import data_feature_engineering, split
from src.test_models.test_gradient_boosting_regressor import test_gradient_boosting_regressor
from src.test_models.test_random_forest_classifier import test_random_forest_classifier
from src.test_models.test_random_forest_regressor import test_random_forest_regressor
from src.test_models.utility import dominates


def main():
    # Load your data
    df = pd.read_csv('../data/user_feedback.csv')

    df = data_feature_engineering(df)

    #test_all_models(df, X_train, X_test, y_train, y_test)

    mae, qwk, params = test_random_forest_regressor(df, test_first_params=False, repeat=2)
    print(f"Random Forest Regressor MAE: {mae}")
    print(f"Random Forest Regressor QWK: {qwk}")
    print(f"Random Forest Regressor Params: {params}")

    mae, qwk, params = test_random_forest_classifier(df, test_first_params=False, repeat=2)
    print(f"Random Forest Classifier MAE: {mae}")
    print(f"Random Forest Classifier QWK: {qwk}")
    print(f"Random Forest Classifier Params: {params}")

    mae, qwk, params = test_gradient_boosting_regressor(df, test_first_params=False, repeat=2)
    print(f"Gradient Boosting Regressor MAE: {mae}")
    print(f"Gradient Boosting Regressor QWK: {qwk}")
    print(f"Gradient Boosting Regressor Params: {params}")





def test_all_models(df, test_first_params=True):
    best_model = "classifier"

    best_fitness, best_solution = test_random_forest_classifier(df, X_train, X_test, y_train, y_test)
    fitness, solution = test_random_forest_regressor(df, X_train, X_test, y_train, y_test)

    best_mae, best_qwk = best_fitness
    mae, qwk = fitness
    if dominates(mae, qwk, best_mae, best_qwk):
        best_model = "regressor"
        best_mae = mae
        best_qwk = qwk
        best_solution = solution

    fitness, solution = test_gradient_boosting_regressor(df, X_train, X_train, y_train, y_test)
    mae, qwk = fitness
    if dominates(mae, qwk, best_mae, best_qwk):
        best_model = "gradient regressor"
        best_mae = mae
        best_qwk = qwk
        best_solution = solution



    print(f"Best Model: {best_model}")

    return best_model, best_solution, best_mae, best_qwk





if __name__ == '__main__':
    main()