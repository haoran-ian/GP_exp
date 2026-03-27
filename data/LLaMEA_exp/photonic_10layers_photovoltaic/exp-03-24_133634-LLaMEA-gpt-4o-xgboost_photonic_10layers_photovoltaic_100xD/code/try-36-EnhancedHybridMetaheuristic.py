import numpy as np

class EnhancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.temperature = 1.0
        self.cooling_rate = 0.93
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.chaotic_map = np.random.rand(self.population_size)

    def chaotic_sequence(self, x, k=3.9):
        return k * x * (1 - x)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        elite = population[best_idx]
        best_fitness = fitness[best_idx]
        budget_used = self.population_size

        while budget_used < self.budget:
            # Update chaotic sequence
            self.chaotic_map = self.chaotic_sequence(self.chaotic_map)

            # Enhanced DE with Chaotic Maps
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                chaotic_mutation_factor = self.mutation_factor * self.chaotic_map[i]
                mutant = np.clip(a + chaotic_mutation_factor * (b - c), lb, ub)
                dynamic_crossover_rate = self.crossover_rate + 0.05 * np.sin(budget_used / self.budget * np.pi)
                crossover = np.random.rand(self.dim) < dynamic_crossover_rate
                trial = np.where(crossover, mutant, population[i])

                # Metropolis criterion with elite preservation
                trial_fitness = func(trial)
                budget_used += 1
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

                if trial_fitness < best_fitness:
                    elite = trial
                    best_fitness = trial_fitness

                if budget_used >= self.budget:
                    break

            # Insert elite individual
            population[np.argmax(fitness)] = elite
            fitness[np.argmax(fitness)] = best_fitness

            # Cool down temperature
            self.temperature *= self.cooling_rate

            # Adaptive mutation factor adjustment
            cluster_centers = np.mean(population, axis=0)
            diversity = np.linalg.norm(population - cluster_centers, axis=1).mean()
            if diversity < 0.1 * (ub - lb).mean():
                self.mutation_factor *= 1.15

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]