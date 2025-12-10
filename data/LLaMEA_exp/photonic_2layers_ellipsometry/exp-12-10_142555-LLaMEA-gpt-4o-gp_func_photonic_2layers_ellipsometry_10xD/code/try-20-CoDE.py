import numpy as np

class CoDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = max(20, 7 * dim)  # Larger population for diverse exploration
        self.CR = 0.9  # Crossover probability
        self.F = 0.5  # Mutation factor
        self.current_evaluations = 0

    def generate_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.pop_size, self.dim))

    def mutate(self, idx, population, bounds):
        # Select three random distinct indices different from idx
        indices = np.random.choice(self.pop_size, 3, replace=False)
        while idx in indices:
            indices = np.random.choice(self.pop_size, 3, replace=False)
        a, b, c = population[indices]
        donor = a + self.F * (b - c)
        
        # Coevolutionary strategy: create a hybrid donor by blending with a random individual
        rand_indiv_idx = np.random.choice(self.pop_size)
        rand_indiv = population[rand_indiv_idx]
        hybrid_donor = np.random.uniform(0.4, 0.6) * donor + np.random.uniform(0.4, 0.6) * rand_indiv
        return np.clip(hybrid_donor, bounds.lb, bounds.ub)

    def crossover(self, target, donor):
        crossover_mask = np.random.rand(self.dim) < self.CR
        if not np.any(crossover_mask):  # Ensure at least one crossover
            crossover_mask[np.random.randint(0, self.dim)] = True
        offspring = np.where(crossover_mask, donor, target)
        return offspring

    def select(self, candidate, target, func):
        candidate_fitness = func(candidate)
        target_fitness = func(target)
        if candidate_fitness < target_fitness:
            return candidate, candidate_fitness
        return target, target_fitness

    def __call__(self, func):
        bounds = func.bounds
        population = self.generate_population(bounds)
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += self.pop_size

        while self.current_evaluations < self.budget:
            for i in range(self.pop_size):
                donor_vector = self.mutate(i, population, bounds)
                trial_vector = self.crossover(population[i], donor_vector)
                trial_vector = np.clip(trial_vector, bounds.lb, bounds.ub)
                
                population[i], fitness[i] = self.select(trial_vector, population[i], func)
                self.current_evaluations += 1
                if self.current_evaluations >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]