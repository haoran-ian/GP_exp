import numpy as np

class EnhancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * self.dim
        self.crossover_rate = 0.9
        self.scaling_factor = 0.8
        self.temperature = 100.0  # Initial temperature for simulated annealing
        self.cooling_rate = 0.99  # Cooling rate for simulated annealing

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population_size = self.initial_population_size
        population = np.random.rand(population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness = np.array([func(ind) for ind in population])
        eval_count = population_size
        
        while eval_count < self.budget:
            # Dynamically adjust population size
            population_size = min(self.initial_population_size, self.budget - eval_count)
            new_population = np.zeros((population_size, self.dim))
            new_fitness = np.zeros(population_size)
            
            # Differential Evolution Mutation and Crossover
            for i in range(population_size):
                if eval_count >= self.budget:
                    break
                a, b, c = population[np.random.choice(len(population), 3, replace=False)]
                diff_vector = b - c
                scaling_factor = self.scaling_factor * np.random.rand()  # Adaptive scaling factor
                mutant = np.clip(a + scaling_factor * diff_vector, bounds[:, 0], bounds[:, 1])
                
                # Adaptive crossover rate based on diversity
                diversity = np.std(population, axis=0).mean()
                adaptive_crossover_rate = min(1.0, self.crossover_rate + diversity)
                cross_points = np.random.rand(self.dim) < adaptive_crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Simulated Annealing Acceptance Criteria
                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < fitness[i] or np.exp((fitness[i] - trial_fitness) / self.temperature) > np.random.rand():
                    new_population[i] = trial
                    new_fitness[i] = trial_fitness
                else:
                    new_population[i] = population[i]
                    new_fitness[i] = fitness[i]
            
            population = new_population
            fitness = new_fitness
            self.temperature *= self.cooling_rate
        
        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]