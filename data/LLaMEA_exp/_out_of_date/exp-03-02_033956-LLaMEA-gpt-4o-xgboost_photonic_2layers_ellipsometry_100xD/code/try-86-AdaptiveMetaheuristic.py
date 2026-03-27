import numpy as np

class AdaptiveMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.base_F = 0.8
        self.base_CR = 0.9
        self.min_F = 0.4
        self.max_F = 1.2
        self.min_CR = 0.1
        self.max_CR = 1.0
        self.device_factor = 0.3
        self.reshape_probability = 0.25
        self.entropy_threshold = 0.5

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.asarray([func(ind) for ind in population])
        budget_spent = self.population_size

        while budget_spent < self.budget:
            diversity = np.std(population, axis=0).mean()
            F = np.clip(self.base_F * (1 + self.device_factor * (0.5 - diversity)), self.min_F, self.max_F)
            CR = np.clip(self.base_CR * (1 - self.device_factor * diversity), self.min_CR, self.max_CR)
            
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = np.clip(x0 + F * (x1 - x2), lb, ub)

                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                dynamic_adjustment = self.device_factor * (1 - fitness[i] / max(fitness))
                if np.random.rand() < self.reshape_probability:
                    trial += dynamic_adjustment * np.random.normal(0, 0.1, self.dim)
                
                entropy_measure = -np.sum(np.log(np.abs(fitness - np.mean(fitness)) + 1e-5))
                if entropy_measure < self.entropy_threshold:
                    trial += np.random.normal(0, 0.05, self.dim)

                trial_fitness = func(trial)
                budget_spent += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if budget_spent >= self.budget:
                    break

            if np.random.rand() < self.reshape_probability:
                best_indices = np.argsort(fitness)[:self.population_size // 2]
                worst_indices = np.argsort(fitness)[self.population_size // 2:]
                population[worst_indices] = np.random.uniform(lb, ub, (len(worst_indices), self.dim))
                fitness[worst_indices] = [func(ind) for ind in population[worst_indices]]
                budget_spent += len(worst_indices)

        best_index = np.argmin(fitness)
        return population[best_index]