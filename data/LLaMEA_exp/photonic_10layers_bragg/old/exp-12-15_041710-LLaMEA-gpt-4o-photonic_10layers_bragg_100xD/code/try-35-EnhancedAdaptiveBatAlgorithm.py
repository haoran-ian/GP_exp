import numpy as np

class EnhancedAdaptiveBatAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.population_size = self.initial_population_size
        self.frequency_min = 0
        self.frequency_max = 3
        self.loudness = np.random.uniform(0.5, 1.0, self.population_size)
        self.pulse_rate = np.random.uniform(0.2, 0.8, self.population_size)
        self.alpha = 0.9
        self.gamma = 0.9
        self.beta = np.random.uniform(0, 1, self.population_size)
        self.personal_best = np.random.uniform(0, 1, (self.population_size, self.dim))
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        self.improvement_count = np.zeros(self.population_size)
        self.global_best_history = []
        self.exploration_exploitation_tradeoff = 0.5

    def levy_flight(self, lam=1.5):
        sigma = (np.math.gamma(1 + lam) * np.sin(np.pi * lam / 2) /
                 (np.math.gamma((1 + lam) / 2) * lam * 2**((lam - 1) / 2)))**(1 / lam)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / np.abs(v)**(1 / lam)
        return step
    
    def adaptive_parameters(self, improvement_rate):
        self.alpha = 0.8 + 0.2 * improvement_rate
        self.gamma = 0.7 + 0.3 * improvement_rate

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)
        eval_count = self.population_size
        self.global_best_history.append(best_fitness)
        
        while eval_count < self.budget:
            for i in range(self.population_size):
                self.beta[i] = np.random.uniform(0, 1)
                frequency = self.frequency_min + (self.frequency_max - self.frequency_min) * self.beta[i]
                velocities[i] += (population[i] - best_solution) * frequency
                candidate = population[i] + velocities[i]
                candidate = np.clip(candidate, lb, ub)
                
                if np.random.rand() > self.pulse_rate[i]:
                    candidate = best_solution + self.levy_flight() * self.loudness[i]
                
                candidate_fitness = func(candidate)
                eval_count += 1
                
                if candidate_fitness < fitness[i]:
                    self.improvement_count[i] += 1
                    if candidate_fitness < self.personal_best_fitness[i]:
                        self.personal_best[i] = candidate
                        self.personal_best_fitness[i] = candidate_fitness
                
                if candidate_fitness < fitness[i] and np.random.rand() < self.loudness[i]:
                    population[i] = candidate
                    fitness[i] = candidate_fitness
                    self.loudness[i] *= self.alpha
                    if candidate_fitness < best_fitness:
                        self.pulse_rate[i] *= self.gamma
                    
                if candidate_fitness < best_fitness:
                    best_solution = candidate
                    best_fitness = candidate_fitness
                
                if eval_count >= self.budget:
                    break
            
            improvement_rate = np.mean(self.improvement_count) / (self.population_size + 1)
            self.adaptive_parameters(improvement_rate)
            self.global_best_history.append(best_fitness)
        
        return best_solution