
from src.test_models.test_gradient_boosting_regressor import test_gradient_boosting_regressor
from src.test_models.test_random_forest_classifier import test_random_forest_classifier
from src.test_models.test_random_forest_regressor import test_random_forest_regressor
from src.test_models.utility import dominates


def test_all_models(df, test_first_params=True, repeat=2):
    best_model = "classifier"

    best_mae, best_qwk, best_params = test_random_forest_classifier(df, test_first_params, repeat=repeat)
    print(f"Random Forest Classifier MAE: {best_mae}")
    print(f"Random Forest Classifier QWK: {best_qwk}")
    print(f"Random Forest Classifier Params: {best_params}")
    mae, qwk, params = test_random_forest_regressor(df, test_first_params=test_first_params, repeat=repeat)
    print(f"Random Forest Regressor MAE: {mae}")
    print(f"Random Forest Regressor QWK: {qwk}")
    print(f"Random Forest Regressor Params: {params}")
    if dominates(mae, qwk, best_mae, best_qwk):
        best_model = "regressor"
        best_mae = mae
        best_qwk = qwk
        best_params = params

    mae, qwk, params = test_gradient_boosting_regressor(df, test_first_params=test_first_params, repeat=repeat)
    print(f"Gradient Boosting Regressor MAE: {mae}")
    print(f"Gradient Boosting Regressor QWK: {qwk}")
    print(f"Gradient Boosting Regressor Params: {params}")
    if dominates(mae, qwk, best_mae, best_qwk):
        best_model = "gradient regressor"
        best_mae = mae
        best_qwk = qwk
        best_params = params

    print(f"Best Model: {best_model}")

    return best_model, best_params, best_mae, best_qwk





