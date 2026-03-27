import numpy as np

class EnhancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_populations = 3  # Use multiple populations
        self.population_size = 10 * self.dim // self.num_populations
        self.temperature = 1.0
        self.cooling_rate = 0.95  # Adjusted cooling rate
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.selection_pressure = 0.1  # Adaptive selection pressure

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        populations = [np.random.uniform(lb, ub, (self.population_size, self.dim)) for _ in range(self.num_populations)]
        fitnesses = [np.array([func(ind) for ind in pop]) for pop in populations]
        budget_used = self.population_size * self.num_populations

        while budget_used < self.budget:
            for p_idx, (population, fitness) in enumerate(zip(populations, fitnesses)):
                for i in range(self.population_size):
                    idxs = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                    adaptive_mutation_factor = self.mutation_factor / (1 + 0.01 * budget_used)
                    mutant = np.clip(a + adaptive_mutation_factor * (b - c), lb, ub)
                    dynamic_crossover_rate = self.crossover_rate + 0.05 * np.sin(budget_used / self.budget * np.pi)
                    crossover = np.random.rand(self.dim) < dynamic_crossover_rate
                    trial = np.where(crossover, mutant, population[i])

                    trial_fitness = func(trial)
                    budget_used += 1
                    acceptance_prob = np.exp((fitness[i] - trial_fitness) / self.temperature)
                    if trial_fitness < fitness[i] or np.random.rand() < acceptance_prob:
                        population[i] = trial
                        fitness[i] = trial_fitness

                    if budget_used >= self.budget:
                        break
                
                # Inter-population exploration
                if p_idx < self.num_populations - 1:
                    best_from_next = populations[p_idx + 1][np.argmin(fitnesses[p_idx + 1])]
                    swap_idx = np.random.randint(self.population_size)
                    if fitness[swap_idx] > func(best_from_next):
                        population[swap_idx] = best_from_next
                        fitness[swap_idx] = func(best_from_next)

                # Dynamic selection pressure adjustment
                if np.random.rand() < self.selection_pressure:
                    worst_idx = np.argmax(fitness)
                    median_fitness = np.median(fitness)
                    if fitness[worst_idx] > median_fitness:
                        population[worst_idx] = np.random.uniform(lb, ub, self.dim)
                        fitness[worst_idx] = func(population[worst_idx])
                        budget_used += 1

            self.temperature *= self.cooling_rate

        best_overall_idx = np.argmin([np.min(f) for f in fitnesses])
        best_population = populations[best_overall_idx]
        best_fitness = fitnesses[best_overall_idx]
        best_idx = np.argmin(best_fitness)
        return best_population[best_idx], best_fitness[best_idx]