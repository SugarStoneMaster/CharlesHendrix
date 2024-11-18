import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from src.data_processing.feature_engineering import split
from src.test_models.utility import run_ga, evaluate_model, repeat_cross_validation
from sklearn.ensemble import GradientBoostingRegressor


def test_gradient_boosting_regressor(df, test_first_params=True, repeat=2):
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
                                                                            model_type=GradientBoostingRegressor,
                                                                            params=params)

    print(f"\nFinal Average MAE across 10 repeats: {final_avg_mae}")
    print(f"Final Average QWK across 10 repeats: {final_avg_qwk}")

    if best_params_avg is not None:
        best_params = best_params_avg

    return final_avg_mae, final_avg_qwk, best_params




# Discrete values for float parameters
learning_rate_values = np.round(np.linspace(0.01, 0.3, num=50), 3).tolist()
subsample_values = np.round(np.linspace(0.5, 1.0, num=50), 2).tolist()
alpha_values = np.round(np.linspace(0.1, 0.9, num=50), 2).tolist()
gene_space = [
    {'low': 0, 'high': 3},                # loss
    {'low': 0, 'high': len(learning_rate_values) - 1},  # learning_rate (index)
    {'low': 50, 'high': 500},             # n_estimators
    {'low': 0, 'high': len(subsample_values) - 1},      # subsample (index)
    {'low': 0, 'high': 1},                # criterion
    {'low': 2, 'high': 10},               # min_samples_split
    {'low': 1, 'high': 10},               # min_samples_leaf
    {'low': 0, 'high': 7},                # max_depth (0 represents None)
    {'low': 0, 'high': 2},                # max_features
    {'low': 0, 'high': len(alpha_values) - 1}          # alpha (index)
]
loss_options = ['squared_error', 'absolute_error', 'huber', 'quantile']
criterion_options = ['friedman_mse', 'squared_error']
max_features_options = ['sqrt', 'log2', None]
def set_fitness_function(X_train, X_test, y_train, y_test):
    def fitness_function(ga_instance, solution, solution_idx):
        loss = loss_options[int(solution[0])]

        learning_rate = learning_rate_values[solution[1]]
        n_estimators = int(solution[2])
        subsample = subsample_values[solution[3]]

        criterion = criterion_options[int(solution[4])]

        min_samples_split = int(solution[5])
        min_samples_leaf = int(solution[6])

        max_depth_value = int(solution[7])
        max_depth = None if max_depth_value == 0 else max_depth_value + 2  # Offset to start from 3

        max_features = max_features_options[int(solution[8])]

        # Handle alpha parameter
        if loss in ['huber', 'quantile']:
            alpha = alpha_values[solution[9]]
        else:
            alpha = 0.9  # Default value

        # Initialize the regressor with these hyperparameters
        reg = GradientBoostingRegressor(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            max_features=max_features,
            alpha=alpha,
            random_state=42
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
    # Extract each parameter from the solution array and map it to its meaning

    # Loss function: map index to one of the loss options
    loss = loss_options[int(solution[0])]

    # Learning rate: use index to select from predefined values in learning_rate_values
    learning_rate = learning_rate_values[int(solution[1])]

    # n_estimators: directly mapped as an integer from solution
    n_estimators = int(solution[2])

    # Subsample: use index to select from predefined values in subsample_values
    subsample = subsample_values[int(solution[3])]

    # Criterion: map index to one of the criterion options
    criterion = criterion_options[int(solution[4])]

    # min_samples_split and min_samples_leaf are directly taken from solution
    min_samples_split = int(solution[5])
    min_samples_leaf = int(solution[6])

    # max_depth: handle 0 as None and shift other values to represent 1-8 as depths 2-10
    max_depth_value = int(solution[7])
    max_depth = None if max_depth_value == 0 else max_depth_value + 2  # Offset by 2 to range 2-10

    # max_features: map index to options in max_features_options
    max_features = max_features_options[int(solution[8])]

    # Alpha: use index to select from predefined values in alpha_values
    alpha = alpha_values[int(solution[9])]

    # Return parameters as a dictionary
    return {
        'loss': loss,
        'learning_rate': learning_rate,
        'n_estimators': n_estimators,
        'subsample': subsample,
        'criterion': criterion,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_depth': max_depth,
        'max_features': max_features,
        'alpha': alpha
    }





