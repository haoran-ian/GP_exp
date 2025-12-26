import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 10 * dim
        self.pop_size = self.initial_pop_size
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.eval_count = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        self.eval_count += self.pop_size
        
        best_idx = np.argmin(fitness)
        best_individual = pop[best_idx]
        best_fitness = fitness[best_idx]
        
        while self.eval_count < self.budget:
            new_pop = []
            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break
                # Mutation with Gaussian component
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c) + np.random.normal(0, 0.1, self.dim), lb, ub)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                
                # Selection
                trial_fitness = func(trial)
                self.eval_count += 1
                
                if trial_fitness < fitness[i]:
                    new_pop.append(trial)
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial
                else:
                    new_pop.append(pop[i])

                # Adaptive F and CR
                if trial_fitness < fitness[i]:
                    self.F = min(1.0, self.F + 0.05 * (np.random.rand() - 0.5))
                    self.CR = min(1.0, self.CR + 0.05 * (np.random.rand() - 0.5))
                else:
                    self.F = max(0.1, self.F - 0.05 * (np.random.rand() - 0.5))
                    self.CR = max(0.1, self.CR - 0.05 * (np.random.rand() - 0.5))
            
            # Dynamic population resizing
            if self.eval_count % (self.budget // 10) == 0:
                self.pop_size = max(5, int(self.pop_size * 0.9))
                pop = pop[:self.pop_size]
                fitness = fitness[:self.pop_size]

            pop = np.array(new_pop)

        return best_individual, best_fitness