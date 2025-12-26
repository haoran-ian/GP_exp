import numpy as np

class ADE_DPM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # initial population size
        self.CR = 0.9  # crossover probability
        self.F_min = 0.4  # minimum mutation factor
        self.F_max = 0.9  # maximum mutation factor
        self.current_evaluations = 0
        self.population = None
        self.fitness = None
        self.bounds = None

    def generate_population(self):
        lb, ub = self.bounds.lb, self.bounds.ub
        return np.random.uniform(lb, ub, (self.pop_size, self.dim))

    def mutate(self, idx):
        indices = np.random.choice(self.pop_size, 3, replace=False)
        while idx in indices:
            indices = np.random.choice(self.pop_size, 3, replace=False)
        a, b, c = self.population[indices]
        # Adaptive mutation factor
        F = np.random.uniform(self.F_min, self.F_max)
        return a + F * (b - c)

    def crossover(self, target, donor):
        crossover_mask = np.random.rand(self.dim) < self.CR
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        return np.where(crossover_mask, donor, target)

    def select(self, candidate, target, candidate_fitness, target_fitness):
        if candidate_fitness < target_fitness:
            return candidate, candidate_fitness
        return target, target_fitness

    def update_population_size(self):
        # Dynamically adjust population size based on convergence
        improvement_rate = np.std(self.fitness) / max(np.mean(self.fitness), 1e-9)
        if improvement_rate < 0.01:  # If convergence is too slow
            self.pop_size = min(self.pop_size + 1, 20 * self.dim)  # Increase population
        elif improvement_rate > 0.1:  # If convergence is too fast
            self.pop_size = max(self.pop_size - 1, 5 * self.dim)  # Decrease population

    def __call__(self, func):
        self.bounds = func.bounds
        self.population = self.generate_population()
        self.fitness = np.array([func(ind) for ind in self.population])
        self.current_evaluations += self.pop_size

        while self.current_evaluations < self.budget:
            self.update_population_size()
            new_population = np.zeros((self.pop_size, self.dim))
            new_fitness = np.zeros(self.pop_size)

            for i in range(self.pop_size):
                donor_vector = self.mutate(i)
                trial_vector = self.crossover(self.population[i], donor_vector)
                trial_vector = np.clip(trial_vector, self.bounds.lb, self.bounds.ub)

                candidate_fitness = func(trial_vector)
                self.current_evaluations += 1

                new_population[i], new_fitness[i] = self.select(
                    trial_vector, self.population[i], candidate_fitness, self.fitness[i]
                )

                if self.current_evaluations >= self.budget:
                    break

            self.population = new_population
            self.fitness = new_fitness

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]