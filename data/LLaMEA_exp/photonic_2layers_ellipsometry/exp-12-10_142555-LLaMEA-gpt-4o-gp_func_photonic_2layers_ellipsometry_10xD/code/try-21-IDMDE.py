import numpy as np

class IDMDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # population size
        self.CR = 0.9  # initial crossover probability
        self.F = 0.5  # initial mutation factor
        self.current_evaluations = 0
        self.CR_min = 0.1
        self.CR_max = 0.9
        self.F_min = 0.4
        self.F_max = 0.9

    def generate_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.pop_size, self.dim))

    def mutate(self, idx, population):
        indices = np.random.choice(self.pop_size, 3, replace=False)
        while idx in indices:
            indices = np.random.choice(self.pop_size, 3, replace=False)
        a, b, c = population[indices]
        return a + self.F * (b - c)

    def crossover(self, target, donor):
        crossover_mask = np.random.rand(self.dim) < self.CR
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
        self.CR = self.CR_min + (self.CR_max - self.CR_min) * success_rate
        self.F = self.F_min + (self.F_max - self.F_min) * success_rate

    def __call__(self, func):
        bounds = func.bounds
        population = self.generate_population(bounds)
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += self.pop_size
        successful_mutations = 0

        while self.current_evaluations < self.budget:
            for i in range(self.pop_size):
                previous_fitness = fitness[i]
                donor_vector = self.mutate(i, population)
                trial_vector = self.crossover(population[i], donor_vector)
                trial_vector = np.clip(trial_vector, bounds.lb, bounds.ub)
                
                population[i], fitness[i] = self.select(trial_vector, population[i], func)
                if fitness[i] < previous_fitness:
                    successful_mutations += 1
                
                self.current_evaluations += 1
                if self.current_evaluations >= self.budget:
                    break
            
            success_rate = successful_mutations / self.pop_size
            self.adapt_parameters(success_rate)
            successful_mutations = 0

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]