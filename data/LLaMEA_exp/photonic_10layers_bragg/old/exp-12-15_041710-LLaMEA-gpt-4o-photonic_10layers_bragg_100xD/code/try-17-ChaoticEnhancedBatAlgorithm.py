import numpy as np

class ChaoticEnhancedBatAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.frequency_min = 0
        self.frequency_max = 2
        self.loudness = np.random.uniform(0.5, 1.0, self.population_size)
        self.pulse_rate = np.random.uniform(0.2, 0.8, self.population_size)
        self.alpha = 0.98
        self.gamma = 0.95
        self.beta = np.random.uniform(0, 1, self.population_size)
        self.memory = np.zeros((self.population_size, self.dim))
        self.learning_rate = 0.1
        self.improvement_count = np.zeros(self.population_size)
        self.chaos_parameter = 0.7
        
    def chaotic_sequence(self, x):
        return np.sin(np.pi * x)

    def self_adaptive_mutation(self, candidate, best_solution):
        mutation_strength = np.random.uniform(0, 1, self.dim)
        return candidate + mutation_strength * (best_solution - candidate)
    
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
                self.beta[i] = self.chaotic_sequence(self.beta[i])
                frequency = self.frequency_min + (self.frequency_max - self.frequency_min) * self.beta[i]
                velocities[i] += (population[i] - best_solution) * frequency
                candidate = population[i] + velocities[i]
                candidate = np.clip(candidate, lb, ub)
                
                if np.random.rand() > self.pulse_rate[i]:
                    candidate = self.self_adaptive_mutation(best_solution, candidate) + self.learning_rate * np.random.randn(self.dim) * self.loudness[i]
                
                candidate_fitness = func(candidate)
                eval_count += 1
                
                if candidate_fitness < fitness[i]:
                    fitness_improvement = fitness[i] - candidate_fitness
                    self.improvement_count[i] += 1
                    self.memory[i] = candidate if fitness_improvement > 0 else self.memory[i]
                
                if candidate_fitness < fitness[i] and np.random.rand() < self.loudness[i]:
                    population[i] = candidate
                    fitness[i] = candidate_fitness
                    self.loudness[i] *= self.alpha
                    self.pulse_rate[i] *= (1 - np.exp(-self.gamma * self.improvement_count[i] / self.budget))
                    
                if candidate_fitness < best_fitness:
                    best_solution = candidate
                    best_fitness = candidate_fitness
                
                if eval_count >= self.budget:
                    break
        
        return best_solution