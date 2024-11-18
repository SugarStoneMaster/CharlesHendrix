from sklearn.ensemble import RandomForestRegressor
from src.data_processing.feature_engineering import split
from src.test_models.utility import run_ga, evaluate_model, cross_validation, repeat_cross_validation


def test_random_forest_regressor(df, test_first_params=True, repeat=2):
    if test_first_params:
        X_train, X_test, y_train, y_test = split(df, smote=True)
        fitness_function = set_fitness_function(X_train, X_test, y_train, y_test)
        fitness, solution = run_ga(gene_space, fitness_function)
        params = map_solution_to_parameters(solution)
        best_params = params
    else:
        params = None

    final_avg_mae, final_avg_qwk, best_params_avg = repeat_cross_validation(df, n_splits=10, repeat=repeat,
                                                           test_first_params=test_first_params,
                                                           set_fitness_function=set_fitness_function,
                                                           gene_space=gene_space,
                                                           mapping=map_solution_to_parameters,
                                                           model_type=RandomForestRegressor,
                                                           params=params)

    print(f"\nFinal Average MAE across 10 repeats: {final_avg_mae}")
    print(f"Final Average QWK across 10 repeats: {final_avg_qwk}")

    if best_params_avg is not None:
        best_params = best_params_avg

    return final_avg_mae, final_avg_qwk, best_params





gene_space = [
    {'low': 50, 'high': 200},     # n_estimators
    {'low': 0, 'high': 3},        # criterion
    {'low': 0, 'high': 25},       # max_depth (0 represents None)
    {'low': 2, 'high': 10},       # min_samples_split
    {'low': 1, 'high': 5},        # min_samples_leaf
    {'low': 0, 'high': 3},        # max_features option
    {'low': 0.1, 'high': 1.0},    # max_features float (used if option == 3)
    {'low': 0, 'high': 1}         # bootstrap
]
criterion_options = ["squared_error", "absolute_error", "friedman_mse", "poisson"]
max_features_options = ["sqrt", "log2", None]

def set_fitness_function(X_train, X_test, y_train, y_test):
    def fitness_function(ga_instance, solution, solution_idx):
        n_estimators = int(solution[0])

        criterion = criterion_options[int(solution[1])]

        max_depth_value = int(solution[2])
        max_depth = None if max_depth_value == 0 else max_depth_value + 5  # Offset to start from 5

        min_samples_split = int(solution[3])
        min_samples_leaf = int(solution[4])

        # Handle max_features (0: "sqrt", 1: "log2", 2: None, 3: float)
        max_features_option = int(solution[5])
        if max_features_option == 3:
            max_features = solution[6]  # Float between 0.1 and 1.0
        else:
            max_features = max_features_options[max_features_option]

        bootstrap = bool(solution[7])

        reg = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=42,
            n_jobs=-1
        )

        # Train the regressor on the training data
        reg.fit(X_train, y_train)

        # Predict on the validation set
        y_pred = reg.predict(X_test)


        # Evaluate the model
        mae, qwk = evaluate_model(y_test, y_pred, model_type="regressor")
        negate_mae = -mae

        #multi-objective opt
        fitness = [negate_mae, qwk]

        return fitness

    return fitness_function


def map_solution_to_parameters(solution):
    # Map solution values to hyperparameters
    n_estimators = int(solution[0])
    criterion = criterion_options[int(solution[1])]

    max_depth_value = int(solution[2])
    max_depth = None if max_depth_value == 0 else max_depth_value + 5  # Offset to start from 5

    min_samples_split = int(solution[3])
    min_samples_leaf = int(solution[4])

    # Handle max_features (0: "sqrt", 1: "log2", 2: None, 3: float)
    max_features_option = int(solution[5])
    if max_features_option == 3:
        max_features = solution[6]  # Float between 0.1 and 1.0
    else:
        max_features = max_features_options[max_features_option]

    bootstrap = bool(solution[7])

    # Return mapped parameters as a dictionary
    return {
        'n_estimators': n_estimators,
        'criterion': criterion,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
        'bootstrap': bootstrap
    }








