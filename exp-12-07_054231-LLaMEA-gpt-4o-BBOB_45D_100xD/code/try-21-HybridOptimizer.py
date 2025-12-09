import numpy as np

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.mutation_factor = 0.8
        self.crossover_probability = 0.7
        self.temperature = 1.0
        self.cooling_rate = 0.99
        self.dynamic_pop_size_factor = 0.05
        self.scale_factor = 1.0

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        
        evaluations = self.population_size

        while evaluations < self.budget:
            current_pop_size = int(self.population_size * (1 + self.dynamic_pop_size_factor * evaluations / self.budget))  
            population = np.resize(population, (current_pop_size, self.dim))
            fitness = np.resize(fitness, current_pop_size)

            for i in range(current_pop_size):
                decay = (self.budget - evaluations) / self.budget  # Dynamic mutation factor decay
                self.mutation_factor = 0.5 + decay * np.random.rand() * 0.5 * (np.var(fitness) / (np.mean(fitness) + 1e-9)) * (best_fitness / (np.mean(fitness) + 1e-9)) * np.random.uniform(0.9, 1.1)
                indices = [idx for idx in range(current_pop_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * self.scale_factor * (b - c), self.lower_bound, self.upper_bound)

                self.crossover_probability = 0.6 + decay * np.random.rand() * 0.4  # Adaptive crossover probability decay
                cross_points = np.random.rand(self.dim) < self.crossover_probability
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness or np.exp((best_fitness - trial_fitness) / self.temperature) > np.random.rand():
                        best_solution = trial
                        best_fitness = trial_fitness
                        self.scale_factor = 1.0
                else:
                    self.scale_factor *= 1.05

            self.cooling_rate = 0.99 + 0.01 * (best_fitness / (1 + abs(best_fitness)))
            self.temperature *= self.cooling_rate

        return best_solution