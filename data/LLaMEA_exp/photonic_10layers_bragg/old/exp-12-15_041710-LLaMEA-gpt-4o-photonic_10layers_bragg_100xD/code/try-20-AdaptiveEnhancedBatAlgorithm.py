import numpy as np

class AdaptiveEnhancedBatAlgorithm:
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
        self.prev_best_fitness = np.inf  # Added for fitness trend analysis
        self.no_improvement_steps = 0
    
    def opposition_based_learning(self, candidate, lb, ub):
        return lb + ub - candidate

    def levy_flight(self, lam=1.5):
        sigma = (np.math.gamma(1 + lam) * np.sin(np.pi * lam / 2) / 
                 (np.math.gamma((1 + lam) / 2) * lam * 2**((lam - 1) / 2)))**(1 / lam)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / np.abs(v)**(1 / lam)
        return step
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)
        eval_count = self.population_size
        
        while eval_count < self.budget:
            fitness_variance = np.var(fitness)
            self.learning_rate = 0.1 + 0.1 * fitness_variance
            for i in range(self.population_size):
                self.beta[i] = np.random.uniform(0, 1)
                frequency = self.frequency_min + (self.frequency_max - self.frequency_min) * self.beta[i]
                velocities[i] += (population[i] - best_solution) * frequency
                candidate = population[i] + velocities[i]
                candidate = np.clip(candidate, lb, ub)
                
                if np.random.rand() > self.pulse_rate[i]:
                    candidate = best_solution + self.learning_rate * self.levy_flight() * self.loudness[i]
                
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
            
            if eval_count < self.budget:
                for i in range(self.population_size):
                    if np.random.rand() < 0.1:
                        opposite_candidate = self.opposition_based_learning(population[i], lb, ub)
                        opposite_candidate_fitness = func(opposite_candidate)
                        eval_count += 1
                        
                        if opposite_candidate_fitness < fitness[i]:
                            population[i] = opposite_candidate
                            fitness[i] = opposite_candidate_fitness
                        
                        if opposite_candidate_fitness < best_fitness:
                            best_solution = opposite_candidate
                            best_fitness = opposite_candidate_fitness
                            
                        if eval_count >= self.budget:
                            break
            
            # Adaptive parameter tuning based on fitness trend
            if best_fitness == self.prev_best_fitness:
                self.no_improvement_steps += 1
            else:
                self.no_improvement_steps = 0
                self.prev_best_fitness = best_fitness
            
            if self.no_improvement_steps > 5:  # If no improvement in 5 iterations
                self.learning_rate *= 1.1  # Increase exploration
                self.loudness = np.clip(self.loudness + 0.05, 0.5, 1.0)
                self.no_improvement_steps = 0  # Reset step counter
        
        return best_solution