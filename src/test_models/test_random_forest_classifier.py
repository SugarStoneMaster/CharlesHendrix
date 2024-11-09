from sklearn.ensemble import RandomForestClassifier
from src.test_models.utility import run_ga, evaluate_model


def test_random_forest_classifier(X_train, y_train, X_test, y_test):
    fitness_function = set_fitness_function(X_train, y_train, X_test, y_test)

    return run_ga(gene_space, fitness_function)





gene_space = [
    {'low': 50, 'high': 200},     # n_estimators
    {'low': 0, 'high': 2},        # criterion (0: "gini", 1: "entropy", 2: "log_loss")
    {'low': 0, 'high': 25},       # max_depth (0 represents None, else 5 to 30)
    {'low': 2, 'high': 10},       # min_samples_split
    {'low': 1, 'high': 5},        # min_samples_leaf
    {'low': 0, 'high': 2},        # max_features (0: "sqrt", 1: "log2", 2: None)
    {'low': 0, 'high': 1}         # bootstrap (0: False, 1: True)
]
criterion_options = ["gini", "entropy", "log_loss"]
max_features_options = ["sqrt", "log2", None]
def set_fitness_function(X_train, y_train, X_test, y_test):
    def fitness_function(ga_instance, solution, solution_idx):
        n_estimators = int(solution[0])

        criterion = criterion_options[int(solution[1])]

        # max_depth: 0 represents None, else actual depth value
        max_depth_value = int(solution[2])
        max_depth = None if max_depth_value == 0 else max_depth_value + 5  # Offset to start from 5

        min_samples_split = int(solution[3])
        min_samples_leaf = int(solution[4])

        max_features = max_features_options[int(solution[5])]

        bootstrap = bool(solution[6])

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        mae, qwk = evaluate_model(y_test, y_pred, model_type="classifier")
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




"""
# Track the number of generations without improvement
no_improvement_count = 0
max_no_improvement_generations = 20  # Trigger adaptation if no improvement in 20 generations
base_mutation_percent = 20           # Base mutation rate (as percentage of genes)
high_mutation_percent = 80           # Increased mutation rate

def adaptive_mutation(ga_instance, offspring):
    global no_improvement_count, base_mutation_percent, high_mutation_percent

    # Determine the current mutation rate based on the stagnation count
    mutation_percent = high_mutation_percent if no_improvement_count >= max_no_improvement_generations else base_mutation_percent

    # Apply mutation to the offspring
    for chromosome in offspring:
        # Randomly mutate each gene with a probability based on the mutation rate
        for gene_idx in range(len(chromosome)):
            if np.random.rand() < (mutation_percent / 100.0):  # Convert percentage to probability
                # Mutate by adding or subtracting a random value (or you can use the existing mutation range)
                mutation_value = np.random.choice([-1, 1]) * np.random.randint(1, 3)
                chromosome[gene_idx] += mutation_value

    return offspring
"""
