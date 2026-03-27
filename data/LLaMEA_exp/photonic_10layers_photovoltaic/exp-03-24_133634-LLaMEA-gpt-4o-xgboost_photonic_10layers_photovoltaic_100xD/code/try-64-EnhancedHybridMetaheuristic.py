import numpy as np

class EnhancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.temperature = 1.0
        self.cooling_rate = 0.95  # Nonlinear cooling rate
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.elite_fraction = 0.1  # Fraction of elites

    def chaotic_initialization(self, lb, ub):
        x0 = np.random.uniform(0, 1, (self.population_size, self.dim))
        chaotic_sequence = 4 * x0 * (1 - x0)
        return lb + chaotic_sequence * (ub - lb)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.chaotic_initialization(lb, ub)
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size

        while budget_used < self.budget:
            elites = population[np.argsort(fitness)[:int(self.elite_fraction * self.population_size)]]
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = elites[np.random.choice(len(elites), 3, replace=False)]
                adaptive_mutation_factor = self.mutation_factor * (1 - np.cos(budget_used / self.budget * np.pi))
                mutant = np.clip(a + adaptive_mutation_factor * (b - c), lb, ub)
                dynamic_crossover_rate = self.crossover_rate + 0.05 * np.sin(budget_used / self.budget * np.pi)
                crossover = np.random.rand(self.dim) < dynamic_crossover_rate
                trial = np.where(crossover, mutant, population[i])

                # Simulated Annealing: accept based on Metropolis criterion
                trial_fitness = func(trial)
                budget_used += 1
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

                if budget_used >= self.budget:
                    break

            # Nonlinear Cool down temperature
            self.temperature *= self.cooling_rate ** (1 - self.temperature)

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]