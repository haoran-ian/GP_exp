import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.crossover_rate = 0.9
        self.scaling_factor = 0.8
        self.temperature = 100.0  # Initial temperature for simulated annealing
        self.cooling_rate = 0.99  # Cooling rate for simulated annealing

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size
        
        while eval_count < self.budget:
            # Differential Evolution Mutation and Crossover
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                mutant = np.clip(a + self.scaling_factor * (b - c), bounds[:, 0], bounds[:, 1])
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Simulated Annealing Acceptance Criteria
                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < fitness[i] or np.exp((fitness[i] - trial_fitness) / self.temperature) > np.random.rand():
                    population[i] = trial
                    fitness[i] = trial_fitness
            
            self.temperature *= self.cooling_rate
        
        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]

# The code above defines a hybrid optimization algorithm that uses both Differential Evolution for generating
# candidate solutions and Simulated Annealing for probabilistic acceptance, which helps to escape local optima.