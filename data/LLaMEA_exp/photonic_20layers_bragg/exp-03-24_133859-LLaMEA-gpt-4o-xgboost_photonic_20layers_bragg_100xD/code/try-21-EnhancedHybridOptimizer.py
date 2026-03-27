import numpy as np

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population_size = 10 * self.dim
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = 0
        
        def adaptive_step_size(base_step, iteration):
            return base_step * (0.5 ** (iteration / (self.budget // population_size)))

        def stochastic_local_search(ind, step_size=0.1, steps=10):
            candidate = ind.copy()
            for _ in range(steps):
                perturbation = np.random.normal(0, step_size, size=self.dim)
                candidate = np.clip(candidate + perturbation, lb, ub)
                candidate_fitness = func(candidate)
                if candidate_fitness < func(ind):
                    ind = candidate
                    break
            return ind

        def adaptive_differential_evolution():
            nonlocal evaluations
            for iteration in range(self.budget // population_size):
                success_count = 0
                diversity = np.std(population, axis=0).mean()
                for i in range(population_size):
                    if evaluations >= self.budget:
                        break
                    idxs = np.random.choice(np.delete(np.arange(population_size), i), 3, replace=False)
                    x0, x1, x2 = population[idxs]
                    mut_factor = adaptive_step_size(0.9 + (0.4 * diversity), iteration)  # Adjusted mutation factor
                    mutant = np.clip(x0 + mut_factor * (x1 - x2), lb, ub)
                    fitness_improvement_ratio = success_count / (i+1)
                    cross_prob = 0.85 * (1 + 0.5 * fitness_improvement_ratio)  # Adjusted crossover probability
                    cross_points = np.random.rand(self.dim) < cross_prob
                    trial = np.where(cross_points, mutant, population[i])
                    trial_fitness = func(trial)
                    evaluations += 1
                    if trial_fitness < fitness[i]:
                        fitness[i] = trial_fitness
                        population[i] = trial
                        success_count += 1
                if success_count < population_size / 5:
                    break

        for _ in range(self.budget // (population_size * 2)):
            adaptive_differential_evolution()
            for i in range(population_size):
                if evaluations >= self.budget:
                    break
                improved = stochastic_local_search(population[i], step_size=0.15)  # Adjusted step size for local search
                improved_fitness = func(improved)
                evaluations += 1
                if improved_fitness < fitness[i]:
                    fitness[i] = improved_fitness
                    population[i] = improved
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]