import numpy as np

class ImprovedDMDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # population size
        self.CR = 0.9  # initial crossover probability
        self.F = 0.5  # initial mutation factor
        self.current_evaluations = 0

    def generate_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.pop_size, self.dim))

    def mutate(self, idx, population, F):
        indices = np.random.choice(self.pop_size, 3, replace=False)
        while idx in indices:
            indices = np.random.choice(self.pop_size, 3, replace=False)
        a, b, c = population[indices]
        return a + F * (b - c)

    def crossover(self, target, donor, CR):
        crossover_mask = np.random.rand(self.dim) < CR
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        return np.where(crossover_mask, donor, target)

    def select(self, candidate, target, func):
        candidate_fitness = func(candidate)
        target_fitness = func(target)
        if candidate_fitness < target_fitness:
            return candidate, candidate_fitness
        return target, target_fitness

    def adapt_parameters(self, success_rate):
        if success_rate > 0.2:
            self.F = min(self.F * 1.1, 1.0)
            self.CR = max(self.CR * 0.9, 0.1)
        else:
            self.F = max(self.F * 0.9, 0.1)
            self.CR = min(self.CR * 1.1, 1.0)

    def __call__(self, func):
        bounds = func.bounds
        population = self.generate_population(bounds)
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += self.pop_size

        while self.current_evaluations < self.budget:
            successful_trials = 0
            for i in range(self.pop_size):
                F_adapted = self.F * (1 + np.random.uniform(-0.1, 0.1))
                CR_adapted = self.CR * (1 + np.random.uniform(-0.1, 0.1))
                
                donor_vector = self.mutate(i, population, F_adapted)
                trial_vector = self.crossover(population[i], donor_vector, CR_adapted)
                trial_vector = np.clip(trial_vector, bounds.lb, bounds.ub)
                
                candidate, candidate_fitness = self.select(trial_vector, population[i], func)
                
                if candidate_fitness < fitness[i]:
                    successful_trials += 1
                
                population[i] = candidate
                fitness[i] = candidate_fitness
                self.current_evaluations += 1
                
                if self.current_evaluations >= self.budget:
                    break
            
            self.adapt_parameters(successful_trials / self.pop_size)

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]