import numpy as np
import pygad
from sklearn.metrics import mean_absolute_error, cohen_kappa_score


def evaluate_model(y_test, y_pred, model_type='regression'):
    if model_type == 'regression':
        # Round regression predictions to nearest integer between 1 and 5
        y_pred_rounded = np.clip(np.rint(y_pred), 1, 5).astype(int)
    else:
        y_pred_rounded = y_pred.astype(int)

    mae = mean_absolute_error(y_test, y_pred_rounded)

    qwk = cohen_kappa_score(y_test, y_pred_rounded, weights='quadratic')

    return mae, qwk


def run_ga(gene_space, fitness_function):
    num_genes = len(gene_space)

    ga_instance = pygad.GA(
        num_generations=200,
        num_parents_mating=10,
        fitness_func=fitness_function,
        sol_per_pop=25,
        num_genes=num_genes,
        gene_space=gene_space,
        gene_type=int,
        parent_selection_type="nsga2",
        keep_parents=2,
        crossover_type="uniform",
        mutation_type="random",
        mutation_percent_genes=40,
        random_mutation_min_val=0,
        random_mutation_max_val=1,
        suppress_warnings=True,
        parallel_processing=10,
        keep_elitism=5,
    )
    ga_instance.on_generation = on_generation

    ga_instance.run()

    ga_instance.plot_fitness(label=['MAE', 'QWK'])

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Best Hyperparameters Found: {solution}")
    print(f"Best Fitness Value: {solution_fitness}")

    return solution, solution_fitness


# Initialize tracking variables
no_improvement_count = 0
max_no_improvement_generations = 50
best_mae = float("-inf")  # Start with worst possible MAE
best_qwk = float("-inf")  # Start with worst possible QWK
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


def dominates(mae, qwk, best_mae, best_qwk):
    # Returns True if (mae, qwk) is better in at least one objective and no worse in the other
    return (mae < best_mae and qwk >= best_qwk) or (mae <= best_mae and qwk > best_qwk)



