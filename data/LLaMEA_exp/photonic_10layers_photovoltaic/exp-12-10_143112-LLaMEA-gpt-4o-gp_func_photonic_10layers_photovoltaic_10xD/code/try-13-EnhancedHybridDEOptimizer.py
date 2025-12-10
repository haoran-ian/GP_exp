import numpy as np

class EnhancedHybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.f = 0.5  # Adaptive differential mutation factor
        self.cr = 0.9  # Crossover probability
        self.local_search_iters = 3
        self.lr = 0.1  # Learning rate for stochastic local search

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                
                # Adaptive differential mutation
                f_local = np.random.uniform(0.4, 0.9)
                mutant = np.clip(a + f_local * (b - c), lb, ub)
                
                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                trial_fit = func(trial)
                evaluations += 1

                if trial_fit < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = trial_fit
                
                # Stochastic local search
                for _ in range(self.local_search_iters):
                    if evaluations >= self.budget:
                        break
                    
                    direction = np.random.uniform(-1, 1, self.dim)
                    neighbor = trial + self.lr * direction * (ub - lb)
                    neighbor = np.clip(neighbor, lb, ub)
                    neighbor_fit = func(neighbor)
                    evaluations += 1

                    if neighbor_fit < trial_fit:
                        trial = neighbor
                        trial_fit = neighbor_fit

                # Update the best solution found in local search
                new_population[i] = trial
                new_fitness[i] = trial_fit
            
            population = new_population
            fitness = new_fitness

        best_idx = np.argmin(fitness)
        return population[best_idx]