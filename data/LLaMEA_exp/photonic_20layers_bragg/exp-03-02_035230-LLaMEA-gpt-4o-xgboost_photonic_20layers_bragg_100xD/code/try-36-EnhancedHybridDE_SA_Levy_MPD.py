import numpy as np

class EnhancedHybridDE_SA_Levy_MPD:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5  # Initial Differential Evolution scaling factor
        self.CR = 0.9  # Crossover probability
        self.alpha = 0.9  # Cooling rate for Simulated Annealing
        self.beta = 0.99  # Adaptive factor for DE parameters
        self.eexplore_weight = 0.1  # Weight for exploration-exploitation balance
        self.num_subpopulations = 5  # Number of subpopulations
        self.subpop_size = self.population_size // self.num_subpopulations

    def levy_flight(self, size, scale=1.0):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / np.abs(v)**(1 / beta)
        return scale * step

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_budget = self.population_size
        T = 1.0  # Initial temperature for Simulated Annealing
        
        while eval_budget < self.budget:
            for sp in range(self.num_subpopulations):
                subpop = population[sp*self.subpop_size:(sp+1)*self.subpop_size]
                subfit = fitness[sp*self.subpop_size:(sp+1)*self.subpop_size]
                for i in range(self.subpop_size):
                    # Differential Evolution mutation and crossover
                    a, b, c = subpop[np.random.choice(self.subpop_size, 3, replace=False)]
                    mutant = np.clip(a + self.F * (b - c), bounds[:, 0], bounds[:, 1])
                    cross_points = np.random.rand(self.dim) < self.CR
                    trial = np.where(cross_points, mutant, subpop[i])
                    
                    # Simulated Annealing acceptance criterion
                    trial_fitness = func(trial)
                    if eval_budget >= self.budget:
                        break
                    eval_budget += 1
                    if trial_fitness < subfit[i]:
                        subpop[i] = trial
                        subfit[i] = trial_fitness
                    else:
                        acceptance_prob = np.exp((subfit[i] - trial_fitness) / T)
                        if np.random.rand() < acceptance_prob:
                            subpop[i] = trial
                            subfit[i] = trial_fitness

                population[sp*self.subpop_size:(sp+1)*self.subpop_size] = subpop
                fitness[sp*self.subpop_size:(sp+1)*self.subpop_size] = subfit

            # Cooling schedule for Simulated Annealing
            T *= self.alpha * 0.95

            # Adaptive parameter control
            if np.random.rand() < 0.1:
                self.F = self.F * self.beta + self.eexplore_weight * np.random.rand()
                self.CR = self.CR * (self.beta + 0.01) + self.eexplore_weight * np.random.rand()

            # Dynamic exploration-exploitation adjustment with Lévy flights
            global_best_idx = np.argmin(fitness)
            global_best = population[global_best_idx]
            for j in range(self.population_size):
                if np.random.rand() < 0.1:
                    distance = np.linalg.norm(population[j] - global_best)
                    adjust_factor = np.exp(-self.eexplore_weight * distance)
                    levy_scale = adjust_factor if np.random.rand() < 0.5 else 1.0
                    population[j] = population[j] + adjust_factor * (global_best - population[j]) + self.levy_flight(self.dim, scale=levy_scale)
                    population[j] = np.clip(population[j], bounds[:, 0], bounds[:, 1])
                    fitness[j] = func(population[j])
                    eval_budget += 1
                    if eval_budget >= self.budget:
                        break
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]