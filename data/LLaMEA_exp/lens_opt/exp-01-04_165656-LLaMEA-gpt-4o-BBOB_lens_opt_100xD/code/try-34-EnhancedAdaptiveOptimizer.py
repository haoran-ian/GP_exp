import numpy as np

class EnhancedAdaptiveOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20 * dim
        self.temp_schedule = lambda t: max(0.01, 1.0 - t / self.budget)
        self.learning_rate = 0.1
        self.elitism_fraction = 0.1

    def local_search(self, individual, lb, ub):
        step_size = 0.05 * (ub - lb)
        perturbation = np.random.normal(0, step_size, self.dim)
        return np.clip(individual + perturbation, lb, ub)

    def dynamic_population_resizing(self, evaluations):
        return max(5, int(self.initial_population_size * (1 - (evaluations / self.budget)**0.5)))

    def adaptive_learning_rate(self, progress):
        return min(1.0, self.learning_rate + 0.9 * (1 - progress))
    
    def stochastic_sort(self, fitness, penalty_weight):
        penalties = penalty_weight * np.random.rand(len(fitness))
        indices = np.argsort(fitness + penalties)
        return indices

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
            diversity = np.mean(np.std(population, axis=0))
            dynamic_crossover_prob = max(0.5, min(1.0, 1.5 * diversity))
            progress = evaluations / self.budget
            learning_rate = self.adaptive_learning_rate(progress)
            penalty_weight = self.temp_schedule(evaluations)

            sorted_indices = self.stochastic_sort(fitness, penalty_weight)
            elitism_count = int(self.elitism_fraction * population_size)
            elite_population = [population[i] for i in sorted_indices[:elitism_count]]

            for i in range(population_size - elitism_count):
                idxs = [idx for idx in range(len(population)) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutation_factor = 0.6 + np.random.rand() * 0.4
                mutant = np.clip(a + learning_rate * mutation_factor * (b - c) + (1 - learning_rate) * (global_best - a), lb, ub)
                crossover = np.random.rand(self.dim) < dynamic_crossover_prob
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

            population = np.array(new_population + elite_population)
            fitness = np.array([func(ind) for ind in population])

        return global_best