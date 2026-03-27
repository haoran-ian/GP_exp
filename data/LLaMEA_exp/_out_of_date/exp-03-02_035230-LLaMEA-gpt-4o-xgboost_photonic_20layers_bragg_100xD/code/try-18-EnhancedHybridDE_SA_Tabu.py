import numpy as np

class EnhancedHybridDE_SA_Tabu:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5  # Initial Differential Evolution scaling factor
        self.CR = 0.9  # Initial Crossover probability
        self.alpha = 0.9  # Cooling rate for Simulated Annealing
        self.beta = 0.99  # Adaptive factor for DE parameters
        self.eexplore_weight = 0.1  # Weight for exploration-exploitation balance
        self.tabu_tenure = 5  # Tabu search tenure
        self.tabu_list = []

    def levy_flight(self, size):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / np.abs(v)**(1 / beta)
        return step

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_budget = self.population_size
        T = 1.0  # Initial temperature for Simulated Annealing
        success_rate = 0.2
        
        while eval_budget < self.budget:
            for i in range(self.population_size):
                # Differential Evolution mutation and crossover
                a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), bounds[:, 0], bounds[:, 1])
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, population[i])
                
                # Simulated Annealing acceptance criterion
                trial_fitness = func(trial)
                if eval_budget >= self.budget:
                    break
                eval_budget += 1
                if trial_fitness < fitness[i] and trial.tolist() not in self.tabu_list:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    self.tabu_list.append(trial.tolist())
                    if len(self.tabu_list) > self.tabu_tenure:
                        self.tabu_list.pop(0)
                    success_rate += 0.05
                else:
                    acceptance_prob = np.exp((fitness[i] - trial_fitness) / T)
                    if np.random.rand() < acceptance_prob:
                        population[i] = trial
                        fitness[i] = trial_fitness
            
            # Cooling schedule for Simulated Annealing
            T *= self.alpha * 0.95  # Slightly increase cooling rate
            
            # Adaptive parameter control
            if np.random.rand() < 0.1:  # 10% chance to adapt parameters
                self.F = self.F * self.beta + self.eexplore_weight * np.random.rand()  # Randomly tweak F
                self.CR = min(1.0, max(0.1, self.CR * success_rate))  # Adjust CR based on success rate
            
            # Dynamic exploration-exploitation adjustment with Lévy flights
            global_best_idx = np.argmin(fitness)
            global_best = population[global_best_idx]
            for j in range(self.population_size):
                if np.random.rand() < 0.1:  # 10% chance to refine based on global best
                    distance = np.linalg.norm(population[j] - global_best)
                    adjust_factor = np.exp(-self.eexplore_weight * distance)
                    population[j] = population[j] + adjust_factor * (global_best - population[j]) + self.levy_flight(self.dim)
                    population[j] = np.clip(population[j], bounds[:, 0], bounds[:, 1])
                    fitness[j] = func(population[j])
                    eval_budget += 1
                    if eval_budget >= self.budget:
                        break
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]