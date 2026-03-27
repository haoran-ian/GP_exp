import numpy as np

class HybridMetaheuristicRefined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.temperature = 1.0
        self.cooling_rate = 0.92  # Slightly faster cooling
        self.mutation_factor = 0.9
        self.crossover_rate = 0.8

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size

        while budget_used < self.budget:
            # Differential Evolution with adaptive mutation
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                adaptive_mutation_factor = self.mutation_factor * np.exp(-0.01 * budget_used / self.budget)
                mutant = np.clip(a + adaptive_mutation_factor * (b - c), lb, ub)
                dynamic_crossover_rate = self.crossover_rate + 0.1 * np.random.rand()
                crossover = np.random.rand(self.dim) < dynamic_crossover_rate
                trial = np.where(crossover, mutant, population[i])

                # Local search refinement
                local_perturbation = np.random.normal(0, 0.1, self.dim) * (ub - lb) / 2
                local_variant = np.clip(trial + local_perturbation, lb, ub)
                local_variant_fitness = func(local_variant)
                budget_used += 1

                # Simulated Annealing: accept based on Metropolis criterion
                trial_fitness = func(trial)
                budget_used += 1
                if (trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temperature)) and trial_fitness <= local_variant_fitness:
                    population[i] = trial
                    fitness[i] = trial_fitness
                elif local_variant_fitness < fitness[i]:
                    population[i] = local_variant
                    fitness[i] = local_variant_fitness

                if budget_used >= self.budget:
                    break
            
            # Cool down temperature
            self.temperature *= self.cooling_rate

            # Adaptive mutation factor adjustment based on diversity
            cluster_centers = np.mean(population, axis=0)
            diversity = np.linalg.norm(population - cluster_centers, axis=1).mean()
            if diversity < 0.05 * (ub - lb).mean():
                self.mutation_factor *= 1.2  # Increased adaptation factor
            else:
                self.mutation_factor *= 0.9

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]