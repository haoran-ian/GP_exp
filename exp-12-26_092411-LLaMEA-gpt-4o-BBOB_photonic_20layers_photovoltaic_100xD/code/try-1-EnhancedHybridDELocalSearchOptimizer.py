import numpy as np

class EnhancedHybridDELocalSearchOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.8  # Initial differential weight
        self.CR = 0.9  # Initial crossover probability
        self.num_evaluations = 0
        self.elite_rate = 0.1  # Rate of elite preservation
        
    def adapt_parameters(self):
        # Adaptive parameter adjustment based on current evaluations
        self.F = 0.5 + 0.5 * np.random.rand()
        self.CR = 0.8 + 0.2 * np.random.rand()
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        fitness = np.array([func(ind) for ind in population])
        self.num_evaluations += self.population_size
        
        while self.num_evaluations < self.budget:
            self.adapt_parameters()
            # Sort population by fitness for elitism
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            elite_size = max(1, int(self.elite_rate * self.population_size))
            
            for i in range(elite_size, self.population_size):
                # Mutation
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Selection
                trial_fitness = func(trial)
                self.num_evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i], fitness[i] = trial, trial_fitness
                
                # Local Search (Exploitation phase)
                if self.num_evaluations < self.budget:
                    local_candidates = np.random.normal(trial, 0.1 * (ub - lb), size=(5, self.dim))
                    local_fitnesses = np.array([func(cand) for cand in local_candidates])
                    self.num_evaluations += 5
                    best_local_idx = np.argmin(local_fitnesses)
                    if local_fitnesses[best_local_idx] < trial_fitness:
                        population[i], fitness[i] = local_candidates[best_local_idx], local_fitnesses[best_local_idx]

            # Sort and save best solution found
            best_idx = np.argmin(fitness)
            best_solution, best_fitness = population[best_idx], fitness[best_idx]

        return best_solution, best_fitness