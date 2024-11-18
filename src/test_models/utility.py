from collections import defaultdict

import numpy as np
import pygad
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
from sklearn.model_selection import KFold



def evaluate_model(y_test, y_pred, model_type="regressor"):
    if model_type == "regressor":
        # Round regression predictions to nearest integer between 1 and 5
        y_pred_rounded = np.clip(np.rint(y_pred), 1, 5).astype(int)
    else:
        y_pred_rounded = y_pred.astype(int)

    mae = mean_absolute_error(y_test, y_pred_rounded)

    qwk = cohen_kappa_score(y_test, y_pred_rounded, weights='quadratic')

    return mae, qwk


def repeat_cross_validation(df, n_splits=10, repeat=10, test_first_params=True, set_fitness_function=None,
                            gene_space=None, mapping=None, model_type=None, params=None):
    all_averages_mae_scores = []
    all_averages_qwk_scores = []
    all_params = []  # Store all parameters from each fold and repetition

    for random_state in range(repeat):
        #print(f"Repeat {random_state + 1}")
        avg_mae, avg_qwk, fold_params = cross_validation(
            df, n_splits=n_splits, random_state=random_state, test_first_params=test_first_params,
            set_fitness_function=set_fitness_function, gene_space=gene_space, model_type=model_type,
            mapping=mapping, params=params
        )

        all_averages_mae_scores.append(avg_mae)
        all_averages_qwk_scores.append(avg_qwk)

        # Collect parameters from this fold if tuning was done
        if not test_first_params:
            all_params.extend(fold_params)

        print(f"  Repeat {random_state + 1} Average MAE: {avg_mae}, Average QWK: {avg_qwk}")

    # Final average across all repetitions
    final_avg_mae = np.mean(all_averages_mae_scores)
    final_avg_qwk = np.mean(all_averages_qwk_scores)

    # Aggregate the final best parameters across all folds and repetitions
    best_params = aggregate_parameters(all_params) if not test_first_params else params

    return final_avg_mae, final_avg_qwk, best_params if not test_first_params else params


def cross_validation(df, n_splits=10, random_state=42, test_first_params=True, set_fitness_function=None,
                     gene_space=None, mapping=None, model_type=None, params=None):
    X = df.drop(columns=['UserInput'])
    y = df['UserInput']

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    all_mae_scores = []
    all_qwk_scores = []
    fold_params = []  # Store parameters for each fold if tuning is done
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        #print(f"Fold {fold + 1}")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        # Only tune hyperparameters if test_first_params is False
        if not test_first_params:
            fitness_function = set_fitness_function(X_train, X_test, y_train, y_test)
            fitness, solution = run_ga(gene_space, fitness_function)
            params = mapping(solution)  # Get the tuned parameters from the genetic algorithm
            fold_params.append(params)  # Collect tuned parameters for aggregation

        if model_type is RandomForestRegressor:
            model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        elif model_type is RandomForestClassifier:
            model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        elif model_type is GradientBoostingRegressor:
            model = GradientBoostingRegressor(**params, random_state=42)

        model.fit(X_train, y_train)

        # Predict on the test set for this fold
        y_pred = model.predict(X_test)

        if model_type is RandomForestRegressor or model_type is GradientBoostingRegressor:
            model_type_string = "regression"
        else:
            model_type_string = "classifier"
        # Evaluate and store results
        mae, qwk = evaluate_model(y_test, y_pred, model_type=model_type_string)
        all_mae_scores.append(-mae)
        all_qwk_scores.append(qwk)


    # Calculate average scores across all folds
    avg_mae = np.mean(all_mae_scores)
    avg_qwk = np.mean(all_qwk_scores)

    # Return average scores and fold-specific parameters if tuning was done
    return avg_mae, avg_qwk, fold_params if not test_first_params else params


def run_ga(gene_space, fitness_function):
    num_genes = len(gene_space)

    ga_instance = pygad.GA(
        num_generations=100,
        num_parents_mating=10,
        fitness_func=fitness_function,
        sol_per_pop=10,
        num_genes=num_genes,
        gene_space=gene_space,
        gene_type=int,
        parent_selection_type="nsga2",
        keep_parents=2,
        crossover_type="uniform",
        mutation_type="random",
        mutation_percent_genes=80,
        random_mutation_min_val=0,
        random_mutation_max_val=1,
        suppress_warnings=True,
        parallel_processing=10,
        keep_elitism=3,
    )
    ga_instance.on_generation = on_generation

    ga_instance.run()

    #ga_instance.plot_fitness(label=['MAE', 'QWK'])

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Best Hyperparameters Found: {solution}")
    print(f"Best Fitness Value: {solution_fitness}")

    return solution_fitness, solution


# Initialize tracking variables
no_improvement_count = 0
max_no_improvement_generations = 50
best_mae = -1000000
best_qwk = -1000000
def on_generation(ga_instance: pygad.GA):
    global no_improvement_count, best_mae, best_qwk

    # Get the best solution’s fitness (tuple of MAE, QWK)
    best_solution, best_fitness, _ = ga_instance.best_solution()
    mae, qwk = best_fitness

    # Check if the current solution dominates the previous best
    if dominates(mae, qwk, best_mae, best_qwk):
        best_mae, best_qwk = mae, qwk
        no_improvement_count = 0  # Reset if there's improvement
    else:
        no_improvement_count += 1

    # Stop if no improvement for specified generations
    if no_improvement_count >= max_no_improvement_generations:
        print(f"No improvement for {max_no_improvement_generations} generations. Stopping GA.")
        return "stop"

    print(f"Generation {ga_instance.generations_completed}: MAE = {mae:.4f}, QWK = {qwk:.4f}")


def dominates(negative_mae, qwk, best_mae, best_qwk):
    # Returns True if (mae, qwk) is better in at least one objective and no worse in the other
    return (negative_mae > best_mae and qwk >= best_qwk) or (negative_mae >= best_mae and qwk > best_qwk)


def aggregate_parameters(all_params):
    aggregated_params = defaultdict(list)

    # Collect each parameter across all repetitions and folds
    for params in all_params:
        for key, value in params.items():
            aggregated_params[key].append(value)

    # Calculate the final parameters based on mean (continuous) or mode (categorical)
    final_params = {}
    for key, values in aggregated_params.items():
        # Filter out None values
        filtered_values = [v for v in values if v is not None]

        if not filtered_values:
            final_params[key] = None  # Assign None if no valid values remain
            continue

        if isinstance(filtered_values[0], (int, float)) and key != 'bootstrap':  # Continuous parameters
            # Calculate mean and round if integer
            mean_value = np.mean(filtered_values)
            final_params[key] = int(round(mean_value)) if isinstance(filtered_values[0], int) else mean_value
        else:  # Categorical parameters (mode)
            # Calculate the mode after filtering out None values
            unique_values, counts = np.unique(filtered_values, return_counts=True)
            mode_value = unique_values[np.argmax(counts)]
            # Convert 0 and 1 to False and True specifically for 'bootstrap' or boolean-like keys
            final_params[key] = bool(mode_value) if key == 'bootstrap' else mode_value

    return final_params



