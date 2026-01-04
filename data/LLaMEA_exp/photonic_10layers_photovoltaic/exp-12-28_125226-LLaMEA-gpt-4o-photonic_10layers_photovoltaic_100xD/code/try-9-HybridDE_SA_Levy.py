import numpy as np

class HybridDE_SA_Levy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.cr = 0.9
        self.f = 0.8
        self.temperature = 100
        self.cooling_rate = 0.99
        self.eval_count = 0
    
    def levy_flight(self, dim, beta=1.5):
        sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u, size=dim)
        v = np.random.normal(0, 1, size=dim)
        step = u / np.abs(v) ** (1 / beta)
        return step
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += self.population_size
        
        while self.eval_count < self.budget:
            new_population = []
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                adaptive_f = self.f * (1 - self.eval_count / self.budget) + 0.1
                mutant = np.clip(a + adaptive_f * (b - c), lb, ub)
                
                # Incorporating Lévy flight
                levy_step = self.levy_flight(self.dim)
                mutant_levy = np.clip(mutant + levy_step, lb, ub)
                
                cross_points = np.random.rand(self.dim) < self.cr
                trial = np.where(cross_points, mutant_levy, population[i])
                
                trial_fitness = func(trial)
                self.eval_count += 1
                
                if trial_fitness < fitness[i] or np.random.rand() < np.exp(-(trial_fitness - fitness[i]) / self.temperature):
                    new_population.append(trial)
                    fitness[i] = trial_fitness
                else:
                    new_population.append(population[i])
                
                if self.eval_count >= self.budget:
                    break
            
            population = np.array(new_population)
            self.temperature *= self.cooling_rate
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]