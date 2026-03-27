import numpy as np

class ImprovedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.F = 0.5
        self.CR = 0.9
        self.global_best = None
        self.global_best_fitness = float('inf')

    def chaotic_initialization(self, lb, ub, size):
        x = np.zeros(size)
        x[0] = np.random.rand()
        for i in range(1, size[0]):
            x[i] = 4 * x[i - 1] * (1 - x[i - 1])
        scaled_x = lb + (ub - lb) * x
        return scaled_x

    def self_adaptive_parameters(self):
        self.F = np.random.uniform(0.4, 0.9)
        self.CR = np.random.uniform(0.1, 0.9)

    def hybrid_mutation(self, idx, population, lb, ub):
        idxs = [i for i in range(self.pop_size) if i != idx]
        a, b, c = population[np.random.choice(idxs, 3, replace=False)]
        mutant_1 = np.clip(a + self.F * (b - c), lb, ub)
        
        if self.global_best is not None:
            # Additional mutation towards global best
            d = population[np.random.choice(idxs)]
            mutant_2 = np.clip(a + self.F * (d - self.global_best), lb, ub)
            mutant = (mutant_1 + mutant_2) / 2
        else:
            mutant = mutant_1
        
        return mutant

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.chaotic_initialization(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.pop_size

        while num_evaluations < self.budget:
            new_population = np.copy(population)
            for i in range(self.pop_size):
                self.self_adaptive_parameters()
                mutant = self.hybrid_mutation(i, population, lb, ub)
                
                # Adaptive crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                trial_fitness = func(trial)
                num_evaluations += 1
                
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                
                if trial_fitness < self.global_best_fitness:
                    self.global_best = trial
                    self.global_best_fitness = trial_fitness
                
                if num_evaluations >= self.budget:
                    break
            population = new_population

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]