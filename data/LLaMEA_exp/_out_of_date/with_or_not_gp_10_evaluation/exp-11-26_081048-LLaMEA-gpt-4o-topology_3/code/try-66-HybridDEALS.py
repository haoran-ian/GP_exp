import numpy as np

class HybridDEALS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.local_search_intensity = 4
        self.eval_count = 0

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += self.population_size
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        while self.eval_count < self.budget:
            # Adaptive Population Resizing
            self.population_size = max(4, int(10 * self.dim * (1 - self.eval_count / self.budget)))  
            population = population[:self.population_size]  # Resize population
            
            # Differential Evolution Mutation and Crossover
            for i in range(self.population_size):
                indices = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                x1, x2, x3 = population[indices]
                self.F = 0.3 + 0.3 * np.random.random() * (1 - self.eval_count / self.budget) + 0.1 * (1 - best_fitness)  # Further refined randomness
                mutant = np.clip(x1 + self.F * (x2 - x3), self.lower_bound, self.upper_bound)
                self.CR = 0.8 * (1 - np.exp(-2 * self.eval_count / self.budget) + 0.1 * (1 - best_fitness))  # Further refined dynamic crossover
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                self.eval_count += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Adaptive Local Search (if budget allows)
                dynamic_local_search_intensity = self.local_search_intensity + int((self.eval_count / self.budget) * 5)
                adaptive_search_step = 0.1 * (1 - self.eval_count / self.budget)  # New adaptive scaling factor
                if self.eval_count < self.budget and trial_fitness < best_fitness:
                    for _ in range(dynamic_local_search_intensity):
                        local_candidate = trial + np.random.normal(0, adaptive_search_step, self.dim)
                        local_candidate = np.clip(local_candidate, self.lower_bound, self.upper_bound)
                        local_fitness = func(local_candidate)
                        self.eval_count += 1
                        if local_fitness < trial_fitness:
                            trial, trial_fitness = local_candidate, local_fitness
                            population[i] = trial
                            fitness[i] = trial_fitness

                # Update global best
                if trial_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = trial_fitness

                if self.eval_count >= self.budget:
                    break

        return best_solution