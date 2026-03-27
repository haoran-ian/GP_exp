import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.temperature = 1.0
        self.cooling_rate = 0.93
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.phase_switch_threshold = 0.5  # New parameter for multi-phase strategy

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size
        search_phase = 'exploration'  # New tracking variable for phase

        while budget_used < self.budget:
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

                # Incorporate dynamic phase switch based on progress
                if budget_used / self.budget > self.phase_switch_threshold:
                    search_phase = 'exploitation'

                if search_phase == 'exploitation' and trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                elif search_phase == 'exploration' and (trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temperature)):
                    population[i] = trial
                    fitness[i] = trial_fitness

                if budget_used >= self.budget:
                    break

            self.temperature *= self.cooling_rate

            cluster_centers = np.mean(population, axis=0)
            diversity = np.linalg.norm(population - cluster_centers, axis=1).mean()
            if diversity < 0.1 * (ub - lb).mean():
                self.mutation_factor *= 1.15

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]