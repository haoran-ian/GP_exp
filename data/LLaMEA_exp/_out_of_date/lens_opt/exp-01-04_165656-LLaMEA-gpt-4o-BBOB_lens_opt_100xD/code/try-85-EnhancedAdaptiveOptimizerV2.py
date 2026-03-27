import numpy as np

class EnhancedAdaptiveOptimizerV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20 * dim
        self.learning_rate = 0.1
        self.energy_threshold = 0.1

    def local_search(self, individual, lb, ub):
        step_size = 0.05 * (ub - lb)
        perturbation = np.random.normal(0, step_size, self.dim)
        return np.clip(individual + perturbation, lb, ub)

    def dynamic_population_resizing(self, evaluations):
        return max(5, int(self.initial_population_size * (1 - (evaluations / self.budget)**0.5)))

    def adaptive_learning_rate(self, progress):
        return min(1.0, self.learning_rate + 0.9 * (1 - progress))

    def stochastic_rank_selection(self, population, fitness):
        probabilities = np.exp(-np.argsort(np.argsort(fitness)))
        probabilities /= probabilities.sum()
        selected_idx = np.random.choice(range(len(population)), p=probabilities)
        return population[selected_idx]

    def energy_based_reinitialization(self, population, fitness, lb, ub):
        energy = np.std(fitness) / np.mean(fitness)
        if energy < self.energy_threshold:
            return np.random.uniform(lb, ub, population.shape), np.apply_along_axis(func, 1, np.random.uniform(lb, ub, population.shape))
        return population, fitness

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        global_best = population[best_idx]
        global_best_fitness = fitness[best_idx]

        evaluations = population_size

        while evaluations < self.budget:
            population_size = self.dynamic_population_resizing(evaluations)
            new_population = []
            progress = evaluations / self.budget
            learning_rate = self.adaptive_learning_rate(progress)
            for _ in range(population_size):
                a = self.stochastic_rank_selection(population, fitness)
                b = self.stochastic_rank_selection(population, fitness)
                c = self.stochastic_rank_selection(population, fitness)
                mutation_factor = 0.6 + np.random.rand() * 0.4
                mutant = np.clip(a + learning_rate * mutation_factor * (b - c) + (1 - learning_rate) * (global_best - a), lb, ub)
                trial = self.local_search(mutant, lb, ub)
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[np.argmin(fitness)]:
                    new_population.append(trial)
                else:
                    new_population.append(population[np.argmin(fitness)])

                if trial_fitness < global_best_fitness:
                    global_best = trial
                    global_best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

            population, fitness = self.energy_based_reinitialization(np.array(new_population), fitness, lb, ub)
            fitness = np.apply_along_axis(func, 1, population)

        return global_best