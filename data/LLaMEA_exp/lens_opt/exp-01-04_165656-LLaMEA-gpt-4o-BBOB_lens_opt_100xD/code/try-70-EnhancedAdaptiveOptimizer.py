import numpy as np

class EnhancedAdaptiveOptimizer:
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

    def differential_mutation(self, a, b, c, factor, lb, ub):
        return np.clip(a + factor * (b - c), lb, ub)

    def self_adaptive_parameters(self):
        factor = np.random.rand() * 0.5 + 0.5
        crossover_rate = np.random.rand() * 0.5 + 0.5
        return factor, crossover_rate

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
            for i in range(population_size):
                factor, crossover_rate = self.self_adaptive_parameters()
                idxs = [idx for idx in range(len(population)) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = self.differential_mutation(a, b, c, factor, lb, ub)
                crossover = np.random.rand(self.dim) < crossover_rate
                trial = np.where(crossover, mutant, population[i % len(population)])

                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i % len(fitness)] or np.random.rand() < np.exp((fitness[i % len(fitness)] - trial_fitness) / self.temp_schedule(evaluations)):
                    new_population.append(self.local_search(trial, lb, ub))
                    if len(fitness) > i:
                        fitness[i] = trial_fitness
                else:
                    new_population.append(population[i % len(population)])

                if trial_fitness < global_best_fitness:
                    global_best = trial
                    global_best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

            population = np.array(new_population)
            fitness = fitness[:population_size]

        return global_best