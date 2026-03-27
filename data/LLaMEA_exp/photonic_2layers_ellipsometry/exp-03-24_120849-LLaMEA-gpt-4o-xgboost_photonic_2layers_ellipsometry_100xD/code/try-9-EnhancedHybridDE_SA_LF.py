import numpy as np
from scipy.stats import levy

class EnhancedHybridDE_SA_LF:
    def __init__(self, budget, dim, F=0.8, CR=0.9, T0=1000, alpha=0.95, population_factor=10, elite_fraction=0.1):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.T0 = T0
        self.alpha = alpha
        self.population_factor = population_factor
        self.elite_fraction = elite_fraction

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop_size = self.population_factor * self.dim
        elite_size = int(self.elite_fraction * pop_size)
        population = lb + (ub - lb) * np.random.beta(0.5, 0.5, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = pop_size
        
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        best_fitness = fitness[best_idx]

        T = self.T0

        while eval_count < self.budget:
            order = np.argsort(fitness)
            elites = population[order[:elite_size]]
            
            for i in range(pop_size):
                if i not in order[:elite_size]:  # Skip elites in mutation
                    # Dynamic adjustment of F and CR based on current progress
                    F_dynamic = self.F * (1 + np.sin(eval_count / self.budget * np.pi))
                    CR_dynamic = self.CR * (1 - eval_count / self.budget)
                    
                    # Select parents, ensuring diversity
                    indices = np.random.choice(pop_size, 3, replace=False)
                    x0, x1, x2 = population[indices]
                    mutant = np.clip(x0 + F_dynamic * (x1 - x2), lb, ub)
                    cross_points = np.random.rand(self.dim) < CR_dynamic
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    trial = np.where(cross_points, mutant, population[i])

                    # Evaluate trial individual
                    trial_fitness = func(trial)
                    eval_count += 1

                    # Selection and Simulated Annealing acceptance
                    if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / T):
                        population[i] = trial
                        fitness[i] = trial_fitness

                        if trial_fitness < best_fitness:
                            best = trial
                            best_fitness = trial_fitness

            # Lévy Flight for exploration
            if np.random.rand() < 0.2:
                for j in range(pop_size):
                    if j not in order[:elite_size]:
                        levy_step = levy.rvs(size=self.dim)
                        new_position = np.clip(population[j] + levy_step, lb, ub)
                        new_fitness = func(new_position)
                        eval_count += 1

                        if new_fitness < fitness[j]:
                            population[j] = new_position
                            fitness[j] = new_fitness

                            if new_fitness < best_fitness:
                                best = new_position
                                best_fitness = new_fitness

            # Temperature cooling
            T *= self.alpha

        return best, best_fitness