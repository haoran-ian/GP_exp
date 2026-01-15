import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.F = 0.5
        self.CR = 0.9
        self.local_search_prob = 0.1  # Probability of applying local search
        self.max_stagnation = 50  # Stagnation threshold for adaptive population size

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.pop_size
        stagnation_counter = 0

        while num_evaluations < self.budget:
            new_population = np.copy(population)
            best_idx = np.argmin(fitness)
            best_fitness = fitness[best_idx]

            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lb, ub)

                # Adaptive crossover rate
                self.CR = 0.1 + 0.8 * (fitness[i] - best_fitness) / (np.max(fitness) - best_fitness + 1e-9)

                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Local search improvement
                if np.random.rand() < self.local_search_prob:
                    trial = trial + np.random.normal(0, 0.1, self.dim) * (ub - lb)

                trial = np.clip(trial, lb, ub)
                trial_fitness = func(trial)
                num_evaluations += 1
                
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    stagnation_counter = 0  # Reset stagnation counter
                else:
                    stagnation_counter += 1

                if num_evaluations >= self.budget:
                    break

            # Adjust population size if stuck
            if stagnation_counter > self.max_stagnation:
                self.pop_size = max(10, self.pop_size // 2)
                new_indices = np.random.choice(np.arange(len(new_population)), self.pop_size, replace=False)
                new_population = new_population[new_indices]
                fitness = fitness[new_indices]
                stagnation_counter = 0

            population = new_population

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]