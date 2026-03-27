import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.F_memory = [0.5] * self.pop_size
        self.CR_memory = [0.9] * self.pop_size
        self.memory_size = 5
        self.memory_idx = 0
        self.F_mean = 0.5
        self.CR_mean = 0.9

    def chaotic_initialization(self, lb, ub, size):
        x = np.zeros(size)
        x[0] = np.random.rand()
        for i in range(1, size[0]):
            x[i] = 4 * x[i - 1] * (1 - x[i - 1])
        scaled_x = lb + (ub - lb) * x
        return scaled_x

    def update_memory(self, F, CR):
        self.F_memory[self.memory_idx] = F
        self.CR_memory[self.memory_idx] = CR
        self.memory_idx = (self.memory_idx + 1) % self.memory_size
        self.F_mean = np.mean(self.F_memory)
        self.CR_mean = np.mean(self.CR_memory)

    def self_adaptive_parameters(self):
        F = np.random.normal(self.F_mean, 0.1)
        F = np.clip(F, 0.4, 0.9)
        CR = np.random.normal(self.CR_mean, 0.1)
        CR = np.clip(CR, 0.1, 0.9)
        return F, CR

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.chaotic_initialization(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.pop_size

        while num_evaluations < self.budget:
            new_population = np.copy(population)
            for i in range(self.pop_size):
                F, CR = self.self_adaptive_parameters()
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                num_evaluations += 1
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    self.update_memory(F, CR)
                if num_evaluations >= self.budget:
                    break

            population = new_population

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]