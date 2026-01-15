import numpy as np

class EnhancedAdaptiveOptimizerPlus:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20 * dim
        self.temp_schedule = lambda t: max(0.01, 1.0 - t / self.budget)
        self.learning_rate = 0.1

    def local_search(self, individual, lb, ub):
        step_size = 0.05 * (ub - lb)
        perturbation = np.random.normal(0, step_size, self.dim)
        return np.clip(individual + perturbation, lb, ub)

    def dynamic_population_resizing(self, evaluations):
        return max(5, int(self.initial_population_size * (1 - (evaluations / self.budget)**0.5)))

    def adaptive_learning_rate(self, progress):
        return min(1.0, self.learning_rate + 0.9 * (1 - progress))

    def multi_modal_mutation(self, a, b, c, global_best, lb, ub, learning_rate):
        mutation_strategy = np.random.choice(["p-best", "rand", "current-to-best"], p=[0.3, 0.4, 0.3])
        mutation_factor = 0.6 + np.random.rand() * 0.4
        if mutation_strategy == "p-best":
            return np.clip(a + mutation_factor * (b - c) + (1 - learning_rate) * (global_best - a), lb, ub)
        elif mutation_strategy == "rand":
            return np.clip(a + mutation_factor * (b - c), lb, ub)
        else:
            return np.clip(a + mutation_factor * (global_best - a) + (b - c), lb, ub)

    def adaptive_crossover_probability(self, evaluations, diversity):
        return 0.5 + 0.5 * (1 - np.tanh((evaluations / self.budget - 0.5) * 5)) + 0.5 * diversity

    def elitist_selection(self, new_population, new_fitness, old_population, old_fitness):
        combined_population = np.vstack((new_population, old_population))
        combined_fitness = np.hstack((new_fitness, old_fitness))
        indices = np.argsort(combined_fitness)
        return combined_population[indices[:len(old_population)]], combined_fitness[indices[:len(old_population)]]

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
            new_population = []
            new_fitness = []
            diversity = np.mean(np.std(population, axis=0))
            dynamic_crossover_prob = self.adaptive_crossover_probability(evaluations, diversity)
            progress = evaluations / self.budget
            learning_rate = self.adaptive_learning_rate(progress)
            for i in range(population_size):
                idxs = [idx for idx in range(len(population)) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = self.multi_modal_mutation(a, b, c, global_best, lb, ub, learning_rate)
                crossover = np.random.rand(self.dim) < dynamic_crossover_prob
                trial = np.where(crossover, mutant, population[i % len(population)])

                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i % len(fitness)] or np.random.rand() < np.exp((fitness[i % len(fitness)] - trial_fitness) / self.temp_schedule(evaluations)):
                    new_population.append(self.local_search(trial, lb, ub))
                    new_fitness.append(trial_fitness)
                else:
                    new_population.append(population[i % len(population)])
                    new_fitness.append(fitness[i % len(fitness)])

                if trial_fitness < global_best_fitness:
                    global_best = trial
                    global_best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

            population, fitness = self.elitist_selection(np.array(new_population), np.array(new_fitness), population, fitness)

        return global_best