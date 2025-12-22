import numpy as np

class HybridBatAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.frequency_min = 0
        self.frequency_max = 2
        self.loudness = 1.0
        self.pulse_rate = 0.5
        self.alpha = 0.9
        self.gamma = 0.9
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)
        eval_count = self.population_size
        
        while eval_count < self.budget:
            for i in range(self.population_size):
                frequency = self.frequency_min + (self.frequency_max - self.frequency_min) * np.random.rand()
                velocities[i] += (population[i] - best_solution) * frequency
                candidate = population[i] + velocities[i]
                candidate = np.clip(candidate, lb, ub)
                
                if np.random.rand() > self.pulse_rate:
                    candidate = best_solution + 0.01 * np.random.randn(self.dim) * self.loudness
                
                candidate_fitness = func(candidate)
                eval_count += 1
                
                if candidate_fitness < fitness[i] and np.random.rand() < self.loudness:
                    population[i] = candidate
                    fitness[i] = candidate_fitness
                    self.loudness *= self.alpha
                    self.pulse_rate *= (1 - np.exp(-self.gamma * eval_count / self.budget))
                    
                if candidate_fitness < best_fitness:
                    best_solution = candidate
                    best_fitness = candidate_fitness
                    
                if eval_count >= self.budget:
                    break
        
        return best_solution