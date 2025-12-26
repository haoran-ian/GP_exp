import numpy as np

class EnhancedDiversityAdaptiveDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F_base = 0.5  # Base differential weight
        self.CR = 0.9  # Initial crossover probability
        self.num_evaluations = 0
        self.elite_rate = 0.1  # Rate of elite preservation

    def adapt_parameters(self):
        # Self-adaptive mutation scaling
        self.F = self.F_base + 0.3 * np.random.randn()

    def calculate_diversity(self, population):
        # Calculate population diversity
        return np.mean(np.linalg.norm(population - population.mean(axis=0), axis=1))

    def hybrid_local_search(self, candidate, fitness, func, lb, ub):
        # Hybrid local search with simulated annealing
        if np.random.rand() > 0.5:
            temperature = 1.0
            for _ in range(5):  # Attempt 5 local moves
                candidate_new = candidate + np.random.randn(self.dim) * temperature
                candidate_new = np.clip(candidate_new, lb, ub)
                fitness_new = func(candidate_new)
                self.num_evaluations += 1
                if fitness_new < fitness or np.random.rand() < np.exp((fitness - fitness_new) / temperature):
                    candidate, fitness = candidate_new, fitness_new
                temperature *= 0.9
        return candidate, fitness

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        fitness = np.array([func(ind) for ind in population])
        self.num_evaluations += self.population_size
        
        while self.num_evaluations < self.budget:
            self.adapt_parameters()
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            elite_size = max(1, int(self.elite_rate * self.population_size))
            
            for i in range(elite_size, self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                trial_fitness = func(trial)
                self.num_evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i], fitness[i] = trial, trial_fitness
                
                # Hybrid local search
                if self.num_evaluations < self.budget:
                    candidate, candidate_fitness = self.hybrid_local_search(trial, trial_fitness, func, lb, ub)
                    if candidate_fitness < fitness[i]:
                        population[i], fitness[i] = candidate, candidate_fitness

            best_idx = np.argmin(fitness)
            best_solution, best_fitness = population[best_idx], fitness[best_idx]

        return best_solution, best_fitness