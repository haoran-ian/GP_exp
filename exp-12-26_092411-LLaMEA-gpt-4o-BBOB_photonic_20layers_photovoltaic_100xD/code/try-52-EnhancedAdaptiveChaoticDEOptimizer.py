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

    def adapt_parameters(self):
        # Adaptive parameter adjustment using chaotic sequence
        self.F = self.initial_F * (0.5 + 0.5 * np.random.rand())
        self.CR = self.initial_CR * (0.8 + 0.2 * np.random.rand())

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

        while self.num_evaluations < self.budget:
            self.adapt_parameters()
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
                
                # Introducing quadratic crossover; slight modification within allowed limit
                trial = np.clip(trial + 0.1 * (mutant - trial)**2, lb, ub)
                
                trial_fitness = func(trial)
                self.num_evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i], fitness[i] = trial, trial_fitness
                
                if self.num_evaluations < self.budget:
                    diversity = self.calculate_diversity(population)
                    sigma = 0.1 * (ub - lb) * (1 + diversity)
                    local_candidates = trial + np.random.randn(3, self.dim) * sigma
                    local_candidates = np.clip(local_candidates, lb, ub)
                    local_fitnesses = np.array([func(cand) for cand in local_candidates])
                    self.num_evaluations += 3
                    best_local_idx = np.argmin(local_fitnesses)
                    if local_fitnesses[best_local_idx] < trial_fitness:
                        population[i], fitness[i] = local_candidates[best_local_idx], local_fitnesses[best_local_idx]

            # Update chaotic sequence
            chaos_sequence = self.logistic_map(chaos_sequence)
            self.F = self.initial_F * (0.5 + 0.5 * chaos_sequence[np.argmin(fitness)])
            self.CR = self.initial_CR * (0.8 + 0.2 * chaos_sequence[np.argmin(fitness)])

        best_idx = np.argmin(fitness)
        best_solution, best_fitness = population[best_idx], fitness[best_idx]

        return best_solution, best_fitness