import numpy as np

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 15 * dim
        self.temp_schedule = lambda t: max(0.01, 1.0 - t / self.budget)

    def local_search(self, individual, lb, ub):
        step_size = 0.05 * (ub - lb)
        perturbation = np.random.normal(0, step_size, self.dim)
        return np.clip(individual + perturbation, lb, ub)

    def dynamic_population_resizing(self, evaluations):
        return max(4, int(self.initial_population_size * (1 - (evaluations / self.budget)**0.5)))

    def adaptive_mutation_scale(self, fitness, global_best_fitness):
        return 0.5 + 0.5 * (1 - fitness / (global_best_fitness + 1e-8))

    def neighborhood_based_selection(self, population, fitness, lb, ub):
        selected_population = []
        for i in range(len(population)):
            neighbors = np.array([population[j] for j in range(len(population)) if j != i])
            neighbor_fitnesses = np.array([fitness[j] for j in range(len(fitness)) if j != i])
            best_neighbor_idx = np.argmin(neighbor_fitnesses)
            selected_population.append(neighbors[best_neighbor_idx])
        return np.array(selected_population)

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
            for i in range(population_size):
                idxs = [idx for idx in range(len(population)) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutation_factor = self.adaptive_mutation_scale(fitness[i % len(fitness)], global_best_fitness)
                mutant = np.clip(a + mutation_factor * (b - c), lb, ub)
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

            population = self.neighborhood_based_selection(np.array(new_population), fitness, lb, ub)
            fitness = fitness[:population_size]

        return global_best