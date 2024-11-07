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
        parallel_processing=20,
        keep_elitism=5,
    )
    ga_instance.on_generation = on_generation

    ga_instance.run()

    ga_instance.plot_fitness(label=['MAE', 'QWK'])

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Best Hyperparameters Found: {solution}")
    print(f"Best Fitness Value: {solution_fitness}")

    return solution, solution_fitness


def on_generation(ga_instance):
    current_generation = ga_instance.generations_completed
    print(f"Generation {current_generation}")




