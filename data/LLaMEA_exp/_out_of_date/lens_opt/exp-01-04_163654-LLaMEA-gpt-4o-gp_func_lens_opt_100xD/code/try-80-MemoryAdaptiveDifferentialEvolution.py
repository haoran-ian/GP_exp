import numpy as np

class MemoryAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.memory_size = 5
        self.F = 0.5
        self.CR = 0.9
        self.mem_F = np.random.uniform(0.4, 0.9, self.memory_size)
        self.mem_CR = np.random.uniform(0.1, 0.9, self.memory_size)
    
    def chaotic_initialization(self, lb, ub, size):
        x = np.zeros(size)
        x[0] = np.random.rand()
        for i in range(1, size[0]):
            x[i] = 4 * x[i - 1] * (1 - x[i - 1])
        scaled_x = lb + (ub - lb) * x
        return scaled_x

    def self_adaptive_parameters(self, i):
        self.F = np.random.choice(self.mem_F)
        self.CR = np.random.choice(self.mem_CR)
    
    def update_memory(self, F, CR, success):
        if success:
            replace_idx = np.random.randint(0, self.memory_size)
            self.mem_F[replace_idx] = F
            self.mem_CR[replace_idx] = CR
    
    def tournament_selection(self, fitness, k=3):
        selected_indices = np.random.choice(self.pop_size, k, replace=False)
        best_idx = min(selected_indices, key=lambda idx: fitness[idx])
        return best_idx

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.chaotic_initialization(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.pop_size

        while num_evaluations < self.budget:
            new_population = np.copy(population)
            for i in range(self.pop_size):
                idx = self.tournament_selection(fitness)
                self.self_adaptive_parameters(idx)
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                num_evaluations += 1
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    self.update_memory(self.F, self.CR, success=True)
                else:
                    self.update_memory(self.F, self.CR, success=False)
                if num_evaluations >= self.budget:
                    break

            population = new_population

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]