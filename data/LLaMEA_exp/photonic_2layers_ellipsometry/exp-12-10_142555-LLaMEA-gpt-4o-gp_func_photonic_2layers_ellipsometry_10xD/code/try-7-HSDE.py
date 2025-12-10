import numpy as np

class HSDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # population size
        self.CR = 0.9  # crossover probability
        self.F = 0.5  # mutation factor
        self.current_evaluations = 0
        # Initialize PSO-related parameters
        self.velocity = np.random.rand(self.pop_size, self.dim)
        self.w = 0.5  # inertia weight
        self.c1 = 1.0  # cognitive (personal) coefficient
        self.c2 = 1.5  # social coefficient

    def generate_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.pop_size, self.dim))

    def mutate(self, idx, population):
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

    def update_velocity(self, idx, population, personal_best, global_best):
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        cognitive = self.c1 * r1 * (personal_best[idx] - population[idx])
        social = self.c2 * r2 * (global_best - population[idx])
        self.velocity[idx] = self.w * self.velocity[idx] + cognitive + social

    def __call__(self, func):
        bounds = func.bounds
        population = self.generate_population(bounds)
        fitness = np.array([func(ind) for ind in population])
        personal_best = np.copy(population)
        personal_best_fitness = np.copy(fitness)
        global_best_idx = np.argmin(fitness)
        global_best = population[global_best_idx]
        self.current_evaluations += self.pop_size

        while self.current_evaluations < self.budget:
            for i in range(self.pop_size):
                self.update_velocity(i, population, personal_best, global_best)
                donor_vector = self.mutate(i, population)
                trial_vector = self.crossover(population[i], donor_vector)
                trial_vector = np.clip(trial_vector, bounds.lb, bounds.ub)
                
                population[i], fitness[i] = self.select(trial_vector, population[i], func)
                self.current_evaluations += 1
                if fitness[i] < personal_best_fitness[i]:
                    personal_best[i] = population[i]
                    personal_best_fitness[i] = fitness[i]
                if fitness[i] < personal_best_fitness[global_best_idx]:
                    global_best_idx = i
                    global_best = population[i]
                if self.current_evaluations >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]