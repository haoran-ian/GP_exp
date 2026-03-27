import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.temperature = 1.0
        self.cooling_rate = 0.91  # Further cooling for deeper exploration
        self.mutation_factor = 0.85
        self.crossover_rate = 0.65
        self.momentum = np.zeros((self.population_size, self.dim))  # Momentum for exploration

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size

        while budget_used < self.budget:
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                # Adaptive mutation factor with momentum
                momentum_contribution = 0.1 * self.momentum[i]
                adaptive_mutation_factor = self.mutation_factor / (1 + 0.01 * budget_used)
                mutant = np.clip(a + adaptive_mutation_factor * (b - c) + momentum_contribution, lb, ub)
                
                # Dynamic crossover rate
                dynamic_crossover_rate = self.crossover_rate + 0.05 * np.sin(budget_used / self.budget * np.pi)
                crossover = np.random.rand(self.dim) < dynamic_crossover_rate
                trial = np.where(crossover, mutant, population[i])

                # Simulated Annealing
                trial_fitness = func(trial)
                budget_used += 1
                accept = trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temperature)
                
                if accept:
                    self.momentum[i] = trial - population[i]  # Update momentum
                    population[i] = trial
                    fitness[i] = trial_fitness

                if budget_used >= self.budget:
                    break

            # Cool down temperature
            self.temperature *= self.cooling_rate

            # Clustering for adaptive mutation factor adjustment
            cluster_centers = np.mean(population, axis=0)
            diversity = np.linalg.norm(population - cluster_centers, axis=1).mean()
            if diversity < 0.1 * (ub - lb).mean():
                self.mutation_factor *= 1.1  # Fine-tuned adaptation factor

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]