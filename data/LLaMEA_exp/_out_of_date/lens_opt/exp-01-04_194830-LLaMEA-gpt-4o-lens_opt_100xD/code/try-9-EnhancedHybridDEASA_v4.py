import numpy as np

class EnhancedHybridDEASA_v4:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 10 * self.dim
        F = 0.8
        CR = 0.9
        T0 = 1000
        Tf = 1e-2

        # Initialize a population randomly within the bounds
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += population_size

        # Introduce memory-based archive and learning
        memory = []

        while self.evaluations < self.budget:
            # Dynamic adjustment of population size
            population_size = max(4, int(10 * self.dim * (1 - self.evaluations / self.budget)))

            # Learning rate for parameter adjustment
            learning_rate = 0.1
            diversity = np.mean(np.std(population, axis=0))
            # Added memory influence in parameter adjustment
            F = F + learning_rate * (0.5 - F) * (1 - np.tanh(diversity)) 
            CR = CR + learning_rate * (0.6 - CR) * (1 - np.tanh(diversity)) 

            # Hybrid Differential Evolution
            for i in range(population_size):
                indices = np.random.choice(population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                if np.random.rand() < 0.5:
                    mutant = np.clip(x1 + F * (x2 - x3), lb, ub)
                else:
                    mutant = np.clip(x1 - F * (x2 - x3), lb, ub)

                trial = np.where(np.random.rand(self.dim) < CR, mutant, population[i])

                trial_fitness = func(trial)
                self.evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    memory.append(trial)  # Update memory

                if self.evaluations >= self.budget:
                    break

            T = T0 * (Tf / T0) ** (self.evaluations / self.budget)
            for i in range(population_size):
                variance = max(1e-5, diversity)
                # Use memory in generating neighbors
                neighbor = (population[i] + np.random.normal(0, variance, self.dim) 
                            + np.mean(memory, axis=0, keepdims=True))
                neighbor = np.clip(neighbor, lb, ub)
                neighbor_fitness = func(neighbor)
                self.evaluations += 1

                if neighbor_fitness < fitness[i] or np.random.rand() < np.exp(-(neighbor_fitness - fitness[i]) / T):
                    population[i] = neighbor
                    fitness[i] = neighbor_fitness

                if self.evaluations >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]