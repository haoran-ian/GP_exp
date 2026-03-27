import numpy as np

class EnhancedHybridOptimizerV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        
    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        initial_pop_size = max(10, 5 * self.dim) 
        de_cr_init = 0.9  
        simplex_size = self.dim + 1  

        population = np.random.uniform(bounds[0], bounds[1], (initial_pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += initial_pop_size

        def adaptive_de():
            nonlocal population, fitness
            pop_size = len(population)
            de_cr = de_cr_init * (1 - self.evaluations / self.budget)  # Dynamic DE crossover probability
            
            for _ in range(self.budget - self.evaluations):
                if self.evaluations >= self.budget:
                    break
                for i in range(pop_size):
                    indices = np.random.choice(pop_size, 3, replace=False)
                    x0, x1, x2 = population[indices]
                    de_f_dynamic = 0.5 + 0.5 * np.random.rand()  
                    mutant = np.clip(x0 + de_f_dynamic * (x1 - x2), bounds[0], bounds[1])
                    cross_points = np.random.rand(self.dim) < de_cr
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    trial = np.where(cross_points, mutant, population[i])
                    f_trial = func(trial)
                    self.evaluations += 1
                    if f_trial < fitness[i]:
                        population[i] = trial
                        fitness[i] = f_trial

        def probabilistic_local_search():
            nonlocal population, fitness
            for _ in range(self.budget - self.evaluations):
                if self.evaluations >= self.budget:
                    break
                best_index = np.argmin(fitness)
                best_individual = population[best_index]
                random_walk = best_individual + np.random.normal(0, 0.1, self.dim)
                random_walk = np.clip(random_walk, bounds[0], bounds[1])
                f_random_walk = func(random_walk)
                self.evaluations += 1
                if f_random_walk < fitness[best_index]:
                    population[best_index] = random_walk
                    fitness[best_index] = f_random_walk

        iteration = 0
        while self.evaluations < self.budget:
            if iteration % 2 == 0:
                adaptive_de()
            else:
                probabilistic_local_search()
            iteration += 1

        best_index = np.argmin(fitness)
        return population[best_index]