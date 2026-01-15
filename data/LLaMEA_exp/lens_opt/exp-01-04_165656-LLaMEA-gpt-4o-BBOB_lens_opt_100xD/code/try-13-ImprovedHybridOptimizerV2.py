import numpy as np
from concurrent.futures import ThreadPoolExecutor

class ImprovedHybridOptimizerV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 15 * dim
        self.temp_schedule = lambda t: max(0.01, 1.0 - t / self.budget)

    def local_search(self, individual, lb, ub, learning_rate):
        step_size = learning_rate * (ub - lb)
        perturbation = np.random.normal(0, step_size, self.dim)
        return np.clip(individual + perturbation, lb, ub)

    def dynamic_population_resizing(self, evaluations):
        return max(4, int(self.initial_population_size * (1 - (evaluations / self.budget)**0.5)))

    def adaptive_learning_rate(self, initial_rate, evaluations):
        return initial_rate * (1 - (evaluations / self.budget))

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        global_best = population[best_idx]
        global_best_fitness = fitness[best_idx]

        evaluations = population_size
        initial_learning_rate = 0.05

        while evaluations < self.budget:
            population_size = self.dynamic_population_resizing(evaluations)
            new_population = np.empty((population_size, self.dim))
            fitness = np.empty(population_size)
            learning_rate = self.adaptive_learning_rate(initial_learning_rate, evaluations)

            def optimize_individual(i):
                idxs = [idx for idx in range(len(population)) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutation_factor = 0.6 + np.random.rand() * 0.4
                mutant = np.clip(a + mutation_factor * (b - c), lb, ub)
                diversity = np.mean(np.std(population, axis=0))
                dynamic_crossover_prob = max(0.5, min(1.0, 1.5 * diversity))
                crossover = np.random.rand(self.dim) < dynamic_crossover_prob
                trial = np.where(crossover, mutant, population[i % len(population)])

                trial = self.local_search(trial, lb, ub, learning_rate)
                trial_fitness = func(trial)

                nonlocal global_best, global_best_fitness

                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temp_schedule(evaluations)):
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                else:
                    new_population[i] = population[i % len(population)]

                if trial_fitness < global_best_fitness:
                    global_best = trial
                    global_best_fitness = trial_fitness

            with ThreadPoolExecutor() as executor:
                executor.map(optimize_individual, range(population_size))
            
            evaluations += population_size
            population = new_population

        return global_best