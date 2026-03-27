import numpy as np

class EnhancedHybridOptimizerPlus:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population_size = max(10, 5 * self.dim)
        initial_population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in initial_population])
        evaluations = population_size

        def adaptive_step_size(base_step, iteration):
            return base_step * np.exp(-0.001 * iteration)

        def neighborhood_search(ind, step_size=0.1):
            perturbation = np.random.normal(0, step_size, size=self.dim)
            candidate = np.clip(ind + perturbation, lb, ub)
            return candidate
        
        def gradient_based_search(ind, step_size=0.01):
            gradient = np.random.uniform(-1, 1, size=self.dim)
            gradient /= np.linalg.norm(gradient)
            candidate = np.clip(ind - step_size * gradient, lb, ub)
            return candidate

        def hybrid_search(ind, step_size=0.1):
            candidate_n = neighborhood_search(ind, step_size)
            candidate_g = gradient_based_search(ind, step_size)
            return candidate_n if func(candidate_n) < func(candidate_g) else candidate_g

        def dynamic_differential_evolution():
            nonlocal evaluations
            for iteration in range(self.budget // (2 * population_size)):
                if evaluations >= self.budget:
                    return
                dynamic_size = int(population_size * (1 - evaluations / self.budget))
                selected_indices = np.random.choice(range(population_size), dynamic_size, replace=False)
                for i in selected_indices:
                    idxs = np.random.choice(np.delete(np.arange(population_size), i), 3, replace=False)
                    x0, x1, x2 = initial_population[idxs]
                    mut_factor = adaptive_step_size(0.8, iteration)
                    mutant = np.clip(x0 + mut_factor * (x1 - x2), lb, ub)
                    cross_points = np.random.rand(self.dim) < 0.7
                    trial = np.where(cross_points, mutant, initial_population[i])
                    trial_fitness = func(trial)
                    evaluations += 1
                    if trial_fitness < fitness[i]:
                        fitness[i] = trial_fitness
                        initial_population[i] = trial

        for _ in range(self.budget // population_size):
            dynamic_differential_evolution()
            for i in range(population_size):
                if evaluations >= self.budget:
                    break
                improved = hybrid_search(initial_population[i], step_size=0.05)
                improved_fitness = func(improved)
                evaluations += 1
                if improved_fitness < fitness[i]:
                    fitness[i] = improved_fitness
                    initial_population[i] = improved

        best_idx = np.argmin(fitness)
        return initial_population[best_idx], fitness[best_idx]