import numpy as np

class M_DMDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # population size
        self.CR = 0.9  # crossover probability
        self.F = 0.5  # mutation factor
        self.current_evaluations = 0

    def generate_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.pop_size, self.dim))

    def mutate(self, idx, population, fitness):
        # Adapting mutation factor based on fitness variability
        if np.std(fitness) > 0.1:
            self.F = 0.5 + np.random.rand() * 0.5
        else:
            self.F = 0.5
        
        # Select three random indices different from idx
        indices = np.random.choice(self.pop_size, 3, replace=False)
        while idx in indices:
            indices = np.random.choice(self.pop_size, 3, replace=False)
        a, b, c = population[indices]
        return a + self.F * (b - c)

    def crossover(self, target, donor):
        crossover_mask = np.random.rand(self.dim) < self.CR
        if not np.any(crossover_mask):  # Guarantee at least one crossover
            crossover_mask[np.random.randint(0, self.dim)] = True
        offspring = np.where(crossover_mask, donor, target)
        return offspring

    def select(self, candidate, target, func):
        candidate_fitness = func(candidate)
        target_fitness = func(target)
        if candidate_fitness < target_fitness:
            return candidate, candidate_fitness
        return target, target_fitness

    def local_search(self, candidate, func, bounds):
        # Simple perturbation-based local search
        perturbation = np.random.normal(0, 0.1, self.dim)
        local_candidate = np.clip(candidate + perturbation, bounds.lb, bounds.ub)
        local_fitness = func(local_candidate)
        candidate_fitness = func(candidate)
        return (local_candidate, local_fitness) if local_fitness < candidate_fitness else (candidate, candidate_fitness)

    def __call__(self, func):
        bounds = func.bounds
        population = self.generate_population(bounds)
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += self.pop_size

        while self.current_evaluations < self.budget:
            for i in range(self.pop_size):
                donor_vector = self.mutate(i, population, fitness)
                trial_vector = self.crossover(population[i], donor_vector)
                trial_vector = np.clip(trial_vector, bounds.lb, bounds.ub)
                
                population[i], fitness[i] = self.select(trial_vector, population[i], func)
                self.current_evaluations += 1
                
                # Apply local search occasionally
                if np.random.rand() < 0.2:
                    population[i], fitness[i] = self.local_search(population[i], func, bounds)
                    self.current_evaluations += 1
                
                if self.current_evaluations >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]