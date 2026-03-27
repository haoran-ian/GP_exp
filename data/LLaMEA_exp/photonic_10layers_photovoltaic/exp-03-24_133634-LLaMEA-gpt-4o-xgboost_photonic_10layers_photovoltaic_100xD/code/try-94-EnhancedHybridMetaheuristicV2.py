import numpy as np

class EnhancedHybridMetaheuristicV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.temperature = 1.0
        self.cooling_rate = 0.95  # Refined for better cooling
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.exploration_factor = 0.1
        self.entropy_threshold = 0.15  # New parameter for entropy-based diversity assessment

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size

        while budget_used < self.budget:
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                adaptive_mutation_factor = self.mutation_factor / (1 + 0.01 * budget_used)
                dynamic_exploration = self.exploration_factor * (np.random.rand() - 0.5)
                mutant = np.clip(a + adaptive_mutation_factor * (b - c) + dynamic_exploration, lb, ub)
                dynamic_crossover_rate = self.crossover_rate + 0.05 * np.cos(budget_used / self.budget * np.pi)
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
            
            # Cool down temperature
            self.temperature *= self.cooling_rate

            # Entropy-based adjustment for mutation and exploration factors
            probabilities = np.histogram(fitness, bins='auto', density=True)[0]
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
            if entropy < self.entropy_threshold:
                self.mutation_factor *= 1.1
                self.exploration_factor *= 1.2

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]