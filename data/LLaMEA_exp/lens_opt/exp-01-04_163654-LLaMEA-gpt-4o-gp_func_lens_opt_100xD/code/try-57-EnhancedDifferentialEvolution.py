import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 20
        self.pop_size = self.initial_pop_size
        self.F_base = 0.5
        self.CR = 0.9
        self.scaling_factor = (self.budget / (self.initial_pop_size * 50))  # Dynamically adjust population size

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.pop_size

        while num_evaluations < self.budget:
            new_population = np.copy(population)
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                # Hybrid mutation strategy: Choose between standard DE and best-target mutation
                if np.random.rand() < 0.5:
                    mutant = a + self.F_base * (b - c)
                else:
                    best = population[np.argmin(fitness)]
                    mutant = best + self.F_base * (b - c)
                
                # Stochastic scaling factor for enhanced exploration
                F = self.F_base * np.random.uniform(0.5, 1.0)
                mutant = np.clip(mutant, lb, ub)
                
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                num_evaluations += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness

                if num_evaluations >= self.budget:
                    break
            
            # Adjust population size dynamically based on budget utilization
            self.pop_size = max(5, int(self.initial_pop_size * (1 - (num_evaluations / self.budget))))
            if num_evaluations < self.budget:
                additional_population_size = self.pop_size - len(new_population)
                if additional_population_size > 0:
                    additional_population = np.random.uniform(lb, ub, (additional_population_size, self.dim))
                    additional_fitness = np.array([func(ind) for ind in additional_population])
                    num_evaluations += additional_population_size
                    new_population = np.vstack((new_population, additional_population))
                    fitness = np.concatenate((fitness, additional_fitness))

            population = new_population

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]