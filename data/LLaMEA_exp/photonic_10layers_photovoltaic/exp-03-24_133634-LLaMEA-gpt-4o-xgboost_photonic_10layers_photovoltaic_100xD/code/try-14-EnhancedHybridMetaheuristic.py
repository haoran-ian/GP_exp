import numpy as np

class EnhancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.temperature = 1.0
        self.cooling_rate = 0.95
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.elitism_rate = 0.1  # New parameter for adaptive elitism

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size

        while budget_used < self.budget:
            elite_count = int(self.elitism_rate * self.population_size)
            sorted_indices = np.argsort(fitness)
            elites = population[sorted_indices[:elite_count]]

            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                entropy_factor = np.std(population, axis=0).mean() / (ub - lb).mean()  # Entropy-based mutation factor
                adaptive_mutation_factor = self.mutation_factor / (1 + entropy_factor * budget_used / self.budget)
                mutant = np.clip(a + adaptive_mutation_factor * (b - c), lb, ub)
                dynamic_crossover_rate = self.crossover_rate + 0.05 * np.sin(budget_used / self.budget * np.pi)
                crossover = np.random.rand(self.dim) < dynamic_crossover_rate
                trial = np.where(crossover, mutant, population[i])

                trial_fitness = func(trial)
                budget_used += 1
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

                if budget_used >= self.budget:
                    break

            self.temperature *= self.cooling_rate

            if budget_used < self.budget:
                population[sorted_indices[-elite_count:]] = elites

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]