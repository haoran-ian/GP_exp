import numpy as np

class EADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.memory = np.zeros((self.initial_population_size, dim))
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size
        self.memory[:self.population_size] = pop
        
        while eval_count < self.budget:
            if eval_count % (self.population_size * 5) == 0 and eval_count > self.population_size:
                # Dynamic population resizing
                self.population_size = min(self.initial_population_size, int(self.population_size * 1.1))
                pop = np.vstack((pop, np.random.uniform(lb, ub, (self.population_size - pop.shape[0], self.dim))))
                fitness = np.append(fitness, [func(ind) for ind in pop[self.memory.shape[0]:]])
                eval_count += self.population_size - self.memory.shape[0]
                self.memory = np.vstack((self.memory, pop[self.memory.shape[0]:]))
            
            for i in range(self.population_size):
                indices = list(range(0, i)) + list(range(i+1, self.population_size))
                a, b, c = self.memory[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)
                
                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover, mutant, pop[i])
                
                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                # Dynamic adaptation
                if eval_count % (self.population_size * 2) == 0:
                    diversity = np.mean([np.linalg.norm(p1 - p2) for p1 in pop for p2 in pop]) / self.dim
                    self.mutation_factor = 0.3 + 0.7 * diversity
                    self.crossover_rate = 0.1 + 0.8 * (1 - diversity)

                if eval_count >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]