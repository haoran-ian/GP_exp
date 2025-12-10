import numpy as np

class EnhancedASMDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(4, 10 * dim // 2)  
        self.base_mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.success_mutation_factors = []
        self.eval_limit_factor = 0.5

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                indices = list(range(0, i)) + list(range(i+1, self.population_size))
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                F = self.base_mutation_factor if not self.success_mutation_factors else np.median(self.success_mutation_factors) * np.random.uniform(0.8, 1.2)
                mutant = np.clip(a + F * (b - c), lb, ub)

                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover, mutant, pop[i])

                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    self.success_mutation_factors.append(F)

                if eval_count % (self.population_size * 2) == 0:
                    diversity = np.mean([np.linalg.norm(p1 - p2) for p1 in pop for p2 in pop]) / self.dim
                    self.base_mutation_factor = 0.3 + 0.7 * diversity
                    self.crossover_rate = 0.1 + 0.8 * (1 - diversity)

                    # Time-varying population size based on convergence speed
                    convergence_speed = np.std(fitness) / np.mean(fitness)
                    if diversity < 0.1 or convergence_speed < 0.01:
                        self.population_size = min(max(self.population_size // 2, 4), 10 * self.dim)
                    else:
                        self.population_size = min(self.population_size * 2, 10 * self.dim)

                if eval_count >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]