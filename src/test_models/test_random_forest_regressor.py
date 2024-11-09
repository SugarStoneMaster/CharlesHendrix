from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from src.test_models.utility import run_ga, evaluate_model


def test_random_forest_regressor(X_train, y_train, X_test, y_test):
    fitness_function = set_fitness_function(X_train, y_train, X_test, y_test)

    run_ga(gene_space, fitness_function)








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

def set_fitness_function(X_train, y_train, X_test, y_test):
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
        mae, qwk = evaluate_model(y_test, y_pred, model_type="regression")
        negate_mae = -mae

        #multi-objective opt
        fitness = [negate_mae, qwk]


        """
        global best_fitness, no_improvement_count

        # Check if there’s an improvement
        if best_fitness is None or fitness > best_fitness:
            best_fitness = fitness  # Update the best fitness
            no_improvement_count = 0  # Reset stagnation counter
        else:
            no_improvement_count += 1  # Increment stagnation counter if no improvement
        """
        return fitness

    return fitness_function







