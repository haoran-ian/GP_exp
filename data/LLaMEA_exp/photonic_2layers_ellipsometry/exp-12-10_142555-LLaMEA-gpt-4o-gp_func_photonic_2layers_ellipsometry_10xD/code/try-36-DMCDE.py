import numpy as np

class DMCDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # population size
        self.CR = 0.9  # initial crossover probability
        self.F = 0.5  # initial mutation factor
        self.current_evaluations = 0
        self.best_fitness_history = []

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
        offspring = np.where(crossover_mask, donor, target)
        return offspring

    def select(self, candidate, target, func):
        candidate_fitness = func(candidate)
        target_fitness = func(target)
        if candidate_fitness < target_fitness:
            return candidate, candidate_fitness
        return target, target_fitness

    def adjust_parameters(self):
        if len(self.best_fitness_history) < 2:
            return
        improvement = self.best_fitness_history[-2] - self.best_fitness_history[-1]
        if improvement < 0.01:  # if not improving much, enhance exploration
            self.F = min(1.0, self.F + 0.1)
            self.CR = max(0.1, self.CR - 0.1)
        else:  # if improving, enhance exploitation
            self.F = max(0.1, self.F - 0.1)
            self.CR = min(1.0, self.CR + 0.1)

    def __call__(self, func):
        bounds = func.bounds
        population = self.generate_population(bounds)
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += self.pop_size
        best_idx = np.argmin(fitness)
        self.best_fitness_history.append(fitness[best_idx])

        while self.current_evaluations < self.budget:
            for i in range(self.pop_size):
                donor_vector = self.mutate(i, population)
                trial_vector = self.crossover(population[i], donor_vector)
                trial_vector = np.clip(trial_vector, bounds.lb, bounds.ub)
                
                population[i], fitness[i] = self.select(trial_vector, population[i], func)
                self.current_evaluations += 1
                if self.current_evaluations >= self.budget:
                    break
                
            best_idx = np.argmin(fitness)
            self.best_fitness_history.append(fitness[best_idx])
            self.adjust_parameters()

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]