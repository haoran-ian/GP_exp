import numpy as np

class EnhancedDynamicDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.eval_count = 0
        self.crossover_rate = 0.9
        self.mutation_factor = 0.8
        self.p_best_rate = 0.2
        self.memory = []

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = np.random.rand(self.population_size, self.dim) * (bounds[1] - bounds[0]) + bounds[0]
        fitness = np.array([func(ind) for ind in population])
        self.eval_count = self.population_size

        while self.eval_count < self.budget:
            for i in range(self.population_size):
                indices = np.random.choice([j for j in range(self.population_size) if j != i], 3, replace=False)
                x1, x2, x3 = population[indices]
                historical_factor = self.memory[-1] if self.memory else 0
                self_adaptive_mutation = np.random.randn(self.dim) * historical_factor
                p_best = population[np.argsort(fitness)[:max(1, int(self.p_best_rate * self.population_size))]][0]
                mutant = np.clip(x1 + self.mutation_factor * (x2 - x3) + self_adaptive_mutation + 0.6 * (p_best - x1), bounds[0], bounds[1])

                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                f_trial = func(trial)
                self.eval_count += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial

            best_idx = np.argmin(fitness)
            if len(self.memory) >= 5:
                self.memory.pop(0)
            self.memory.append(fitness[best_idx])

            ranked_indices = np.argsort(fitness)
            self.crossover_rate = 0.3 + 0.5 * np.sin(4 * np.pi * self.eval_count / self.budget)
            self.mutation_factor = 0.5 + 0.4 * np.cos(4 * np.pi * (self.eval_count/self.budget) * (fitness[ranked_indices[:5]].mean() / (fitness.min() + 1e-8)))

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]