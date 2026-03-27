import numpy as np

class ImprovedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.temperature = 1.0
        self.cooling_rate = 0.95  # Slower cooling for more exploration
        self.mutation_factor = 0.9
        self.crossover_rate = 0.8
        self.exploration_factor = 0.2  # Increased to enhance variability
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size

        while budget_used < self.budget:
            # Enhanced Differential Evolution with local search
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                # Adaptive mutation factor based on diversity
                diversity_factor = np.std(population, axis=0).mean()
                adaptive_mutation_factor = self.mutation_factor * (1 + 0.5 * (diversity_factor / (ub - lb).mean()))
                
                dynamic_exploration = self.exploration_factor * (np.random.rand(self.dim) - 0.5)
                mutant = np.clip(a + adaptive_mutation_factor * (b - c) + dynamic_exploration, lb, ub)
                crossover = np.random.rand(self.dim) < (self.crossover_rate + 0.05 * np.sin(np.pi * budget_used / self.budget))
                trial = np.where(crossover, mutant, population[i])

                # Targeted simulated annealing with adaptive acceptance
                trial_fitness = func(trial)
                budget_used += 1
                acceptance_prob = np.exp((fitness[i] - trial_fitness) / (self.temperature * (1 + diversity_factor / (ub - lb).mean())))
                if trial_fitness < fitness[i] or np.random.rand() < acceptance_prob:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if budget_used >= self.budget:
                    break

            # Gradually cool down temperature
            self.temperature *= self.cooling_rate

            # Adjust mutation and exploration factors dynamically
            cluster_centers = np.mean(population, axis=0)
            diversity = np.linalg.norm(population - cluster_centers, axis=1).mean()
            if diversity < 0.05 * (ub - lb).mean():
                self.mutation_factor *= 1.1
                self.exploration_factor *= 1.1  # Increase exploration in low-diversity scenarios

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]