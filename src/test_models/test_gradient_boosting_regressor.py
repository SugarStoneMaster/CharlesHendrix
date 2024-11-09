import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from src.test_models.utility import run_ga, evaluate_model
from sklearn.ensemble import GradientBoostingRegressor

def test_gradient_boosting_regressor(X_train, y_train, X_test, y_test):
    fitness_function = set_fitness_function(X_train, y_train, X_test, y_test)

    return run_ga(gene_space, fitness_function)





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
def set_fitness_function(X_train, y_train, X_test, y_test):
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







