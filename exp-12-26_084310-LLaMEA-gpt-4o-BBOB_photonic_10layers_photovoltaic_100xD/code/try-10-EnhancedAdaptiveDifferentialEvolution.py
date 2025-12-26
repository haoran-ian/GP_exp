import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.F = 0.5
        self.CR = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best_individual = pop[best_idx]
        best_fitness = fitness[best_idx]
        eval_count = self.pop_size
        
        # Self-adaptive parameters
        F_mutate_rate = np.ones(self.pop_size) * self.F
        CR_mutate_rate = np.ones(self.pop_size) * self.CR

        while eval_count < self.budget:
            for i in range(self.pop_size):
                if eval_count >= self.budget:
                    break
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                F = F_mutate_rate[i]
                mutant = np.clip(a + F * (b - c), lb, ub)
                
                CR = CR_mutate_rate[i]
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                
                trial_fitness = func(trial)
                eval_count += 1
                
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial
                    # Adjust F and CR towards successful values
                    F_mutate_rate[i] = min(1.0, F + 0.1 * np.random.rand() * (1 - F))
                    CR_mutate_rate[i] = min(1.0, CR + 0.1 * np.random.rand() * (1 - CR))
                else:
                    # Adjust F and CR towards exploration
                    F_mutate_rate[i] = max(0.1, F - 0.1 * np.random.rand() * F)
                    CR_mutate_rate[i] = max(0.1, CR - 0.1 * np.random.rand() * CR)

        return best_individual, best_fitness