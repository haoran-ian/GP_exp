import numpy as np

class EnhancedAdaptiveChaoticDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.initial_F = 0.8
        self.initial_CR = 0.9
        self.elite_rate = 0.1
        self.num_evaluations = 0

    def adapt_parameters(self, fitness, best_fitness):
        # Self-adaptive parameter adjustment based on fitness improvement
        improvement_factor = (best_fitness - np.min(fitness)) / abs(best_fitness)
        self.F = self.initial_F * (0.5 + improvement_factor * np.random.rand())
        self.CR = self.initial_CR * (0.8 + improvement_factor * np.random.rand())

    def logistic_map(self, x, r=4.0):
        # Logistic map for generating chaotic sequence
        return r * x * (1 - x)

    def calculate_diversity(self, population):
        return np.mean(np.linalg.norm(population - population.mean(axis=0), axis=1))

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        fitness = np.array([func(ind) for ind in population])
        self.num_evaluations += self.population_size
        chaos_sequence = np.random.rand(self.population_size)

        best_fitness = np.min(fitness)

        while self.num_evaluations < self.budget:
            self.adapt_parameters(fitness, best_fitness)
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            elite_size = max(1, int(self.elite_rate * self.population_size))
            
            for i in range(elite_size, self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                trial_fitness = func(trial)
                self.num_evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i], fitness[i] = trial, trial_fitness

                # Stochastic adaptive jumps
                if self.num_evaluations < self.budget:
                    jump_prob = improvement_factor * self.initial_CR
                    if np.random.rand() < jump_prob:
                        jump_vector = np.random.randn(self.dim)
                        trial += jump_vector * (ub - lb) * 0.05
                        trial = np.clip(trial, lb, ub)
                        trial_fitness = func(trial)
                        self.num_evaluations += 1
                        if trial_fitness < fitness[i]:
                            population[i], fitness[i] = trial, trial_fitness

            best_fitness = np.min(fitness)

        best_idx = np.argmin(fitness)
        best_solution, best_fitness = population[best_idx], fitness[best_idx]

        return best_solution, best_fitness