import numpy as np

class AugmentedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.temperature = 1.0
        self.cooling_rate = 0.95  # Further refined cooling rate
        self.mutation_factor = 0.9  # Improved initial mutation factor
        self.crossover_rate = 0.7

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size

        while budget_used < self.budget:
            # Dynamic Differential Evolution: mutate and crossover with diversity consideration
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                adaptive_mutation_factor = self.mutation_factor / (1 + 0.01 * budget_used)
                mutant = np.clip(a + adaptive_mutation_factor * (b - c), lb, ub)
                dynamic_crossover_rate = self.crossover_rate + 0.05 * np.sin(budget_used / self.budget * np.pi)
                crossover = np.random.rand(self.dim) < dynamic_crossover_rate
                trial = np.where(crossover, mutant, population[i])

                # Simulated Annealing: accept based on Metropolis criterion with elitism
                trial_fitness = func(trial)
                budget_used += 1
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

                if budget_used >= self.budget:
                    break
            
            # Update temperature using refined cooling schedule
            self.temperature *= self.cooling_rate

            # Diversity-driven adjustment for mutation factor and elitism
            cluster_centers = np.mean(population, axis=0)
            diversity = np.linalg.norm(population - cluster_centers, axis=1).mean()
            if diversity < 0.1 * (ub - lb).mean():
                self.mutation_factor *= 1.1  # Adjusted adaptation factor
                # Introduce elitism by preserving top individuals
                elite_idxs = np.argsort(fitness)[:self.population_size // 5]
                elites = population[elite_idxs]
                for j in range(elites.shape[0]):
                    if budget_used < self.budget:
                        candidate = np.random.uniform(lb, ub, self.dim)
                        candidate_fitness = func(candidate)
                        budget_used += 1
                        if candidate_fitness < fitness[elite_idxs[j]]:
                            population[elite_idxs[j]] = candidate
                            fitness[elite_idxs[j]] = candidate_fitness

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]