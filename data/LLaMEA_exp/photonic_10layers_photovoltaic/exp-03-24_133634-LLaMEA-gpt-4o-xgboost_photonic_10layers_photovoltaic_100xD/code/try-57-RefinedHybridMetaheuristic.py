import numpy as np

class RefinedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.temperature = 1.0
        self.cooling_rate = 0.95  # Slightly slower cooling rate for sustained exploration
        self.mutation_factor = 0.9
        self.crossover_rate = 0.75
        self.diversity_threshold = 0.1  # Used for diversity preservation

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size

        while budget_used < self.budget:
            # Differential Evolution with stochastic tunneling
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                adaptive_mutation_factor = self.mutation_factor / (1 + 0.01 * budget_used)
                mutant = np.clip(a + adaptive_mutation_factor * (b - c), lb, ub)
                dynamic_crossover_rate = self.crossover_rate + 0.1 * np.sin(budget_used / self.budget * np.pi)
                crossover = np.random.rand(self.dim) < dynamic_crossover_rate
                trial = np.where(crossover, mutant, population[i])

                # Simulated Annealing with diversity check
                trial_fitness = func(trial)
                budget_used += 1
                energy_difference = fitness[i] - trial_fitness
                diversity = np.linalg.norm(population - np.mean(population, axis=0), axis=1).mean()
                acceptance_probability = np.exp(energy_difference / self.temperature) * (1.0 + (diversity < self.diversity_threshold))
                
                if trial_fitness < fitness[i] or np.random.rand() < acceptance_probability:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if budget_used >= self.budget:
                    break

            # Cool down temperature
            self.temperature *= self.cooling_rate

            # Adaptive mutation factor adjustment based on diversity
            if diversity < self.diversity_threshold * (ub - lb).mean():
                self.mutation_factor *= 1.1  # Adjust mutation factor adaptively

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]