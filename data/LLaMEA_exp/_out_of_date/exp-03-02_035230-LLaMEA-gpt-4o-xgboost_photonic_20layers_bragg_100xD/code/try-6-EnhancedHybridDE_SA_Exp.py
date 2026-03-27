import numpy as np

class EnhancedHybridDE_SA_Exp:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50  # Fixed size for diverse exploration
        self.F = 0.6  # Adaptive Differential Evolution scaling factor
        self.CR = 0.7  # Adaptive Crossover probability
        self.alpha = 0.85  # Cooling rate for Simulated Annealing
        self.beta = 0.95  # Adaptive factor for DE parameters
        self.eexplore_weight = 0.2  # Enhanced exploration-exploitation balance

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_budget = self.population_size
        T = 1.0  # Initial temperature for Simulated Annealing
        phase_threshold = self.budget // 2  # Threshold for phase transition
        
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
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                else:
                    acceptance_prob = np.exp((fitness[i] - trial_fitness) / T)
                    if np.random.rand() < acceptance_prob:
                        population[i] = trial
                        fitness[i] = trial_fitness

            # Cooling schedule for Simulated Annealing
            T *= self.alpha
            
            # Adaptive parameter control
            self.F = max(0.4, self.F * self.beta + self.eexplore_weight * np.random.rand())
            self.CR = max(0.2, self.CR * self.beta + self.eexplore_weight * np.random.rand())
            
            # Dynamic exploration-exploitation adjustment
            global_best_idx = np.argmin(fitness)
            global_best = population[global_best_idx]
            for j in range(self.population_size):
                adaptive_explore_factor = 0.2 if eval_budget < phase_threshold else 0.05
                if np.random.rand() < adaptive_explore_factor:  # Adaptive chance to refine based on global best
                    distance = np.linalg.norm(population[j] - global_best)
                    adjust_factor = np.exp(-self.eexplore_weight * distance)
                    population[j] = population[j] + adjust_factor * (global_best - population[j])
                    population[j] = np.clip(population[j], bounds[:, 0], bounds[:, 1])
                    fitness[j] = func(population[j])
                    eval_budget += 1
                    if eval_budget >= self.budget:
                        break
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]