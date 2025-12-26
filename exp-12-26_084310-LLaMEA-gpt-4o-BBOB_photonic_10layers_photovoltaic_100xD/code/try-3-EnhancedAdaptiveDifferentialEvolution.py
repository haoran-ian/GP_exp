import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.memory_size = 5
        self.F_memory = np.full(self.memory_size, self.F)
        self.CR_memory = np.full(self.memory_size, self.CR)
        self.success_mem_index = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best_individual = pop[best_idx]
        best_fitness = fitness[best_idx]
        eval_count = self.pop_size
        
        while eval_count < self.budget:
            for i in range(self.pop_size):
                if eval_count >= self.budget:
                    break
                
                # Update F and CR from memory
                self.F = self.F_memory[np.random.randint(self.memory_size)]
                self.CR = self.CR_memory[np.random.randint(self.memory_size)]

                # Mutation
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                
                # Selection
                trial_fitness = func(trial)
                eval_count += 1
                
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial
                    
                    # Store successful F and CR
                    self.F_memory[self.success_mem_index] = self.F
                    self.CR_memory[self.success_mem_index] = self.CR
                    self.success_mem_index = (self.success_mem_index + 1) % self.memory_size

                # Adaptive F and CR based on success
                if trial_fitness < fitness[i]:
                    self.F = min(1.0, self.F + 0.1 * (np.random.rand() - 0.5))
                    self.CR = min(1.0, self.CR + 0.1 * (np.random.rand() - 0.5))
                else:
                    self.F = max(0.1, self.F - 0.1 * (np.random.rand() - 0.5))
                    self.CR = max(0.1, self.CR - 0.1 * (np.random.rand() - 0.5))

        return best_individual, best_fitness