import numpy as np

class DualPhaseAdaptiveMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.F = 0.8
        self.CR = 0.9
        self.ensemble_factor = 0.2
        self.dynamic_adjustment_rate = 0.1  # Dynamic adjustment rate for parameters

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.asarray([func(ind) for ind in population])
        budget_spent = self.population_size
        
        while budget_spent < self.budget:
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = np.clip(x0 + self.F * (x1 - x2), lb, ub)
                
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                if np.random.rand() < self.ensemble_factor:
                    closest_idx = np.argmin([np.linalg.norm(trial - p) for p in population])
                    trial += self.dynamic_adjustment_rate * (population[closest_idx] - trial)
                
                trial_fitness = func(trial)
                budget_spent += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                
                if budget_spent >= self.budget:
                    break

            if np.random.rand() < self.ensemble_factor:
                best_indices = np.argsort(fitness)[:self.population_size // 2]
                worst_indices = np.argsort(fitness)[self.population_size // 2:]
                population[worst_indices] = np.random.uniform(lb, ub, (len(worst_indices), self.dim))
                fitness[worst_indices] = [func(ind) for ind in population[worst_indices]]
                budget_spent += len(worst_indices)

            # Dynamic adjustment of parameters based on diversity
            diversity = np.std(population, axis=0).mean()
            if diversity < 0.1:
                self.F = min(1.0, self.F + 0.05)
                self.CR = max(0.5, self.CR - 0.05)
            else:
                self.F = max(0.5, self.F - 0.05)
                self.CR = min(1.0, self.CR + 0.05)

        best_index = np.argmin(fitness)
        return population[best_index]