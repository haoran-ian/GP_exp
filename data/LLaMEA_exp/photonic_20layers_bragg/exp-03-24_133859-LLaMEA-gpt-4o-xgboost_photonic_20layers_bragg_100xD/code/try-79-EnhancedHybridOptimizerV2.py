import numpy as np

class EnhancedHybridOptimizerV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        initial_population_size = 10 * self.dim
        population = np.random.uniform(lb, ub, (initial_population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = initial_population_size

        def adaptive_step_size(base_step, iteration):
            return base_step * (0.5 ** (iteration / (self.budget // (2 * initial_population_size))))

        def dynamic_population_adjustment(iteration):
            return max(5, initial_population_size - iteration // 10)

        def learning_based_mutation(ind, best_ind, step_size=0.1):
            candidate = ind.copy()
            perturbation = np.random.uniform(-step_size, step_size, size=self.dim) * \
                           (best_ind - ind)
            candidate = np.clip(candidate + perturbation, lb, ub)
            return candidate

        def stochastic_local_search(ind, step_size=0.1, steps=5):
            best_candidate = ind.copy()
            best_candidate_fitness = func(ind)
            for _ in range(steps):
                candidate = learning_based_mutation(ind, best_candidate, step_size)
                candidate_fitness = func(candidate)
                if candidate_fitness < best_candidate_fitness:
                    best_candidate, best_candidate_fitness = candidate, candidate_fitness
            return best_candidate

        def adaptive_differential_evolution():
            nonlocal evaluations
            for iteration in range(self.budget // (2 * initial_population_size)):
                population_size = dynamic_population_adjustment(iteration)
                current_population = population[:population_size]
                current_fitness = fitness[:population_size]
                success_count = 0
                diversity = np.std(current_population, axis=0).mean()
                diversity_factor = 1 / (1 + np.exp(-10 * (diversity - 0.5)))
                best_ind = current_population[np.argmin(current_fitness)]
                for i in range(population_size):
                    if evaluations >= self.budget:
                        break
                    idxs = np.random.choice(np.delete(np.arange(population_size), i), 3, replace=False)
                    x0, x1, x2 = current_population[idxs]
                    mut_factor = adaptive_step_size(0.8 + (0.5 * diversity_factor), iteration)
                    mutant = np.clip(x0 + mut_factor * (x1 - x2), lb, ub)
                    fitness_improvement_ratio = success_count / (i+1) if i > 0 else 0
                    cross_prob = 0.9 * (1 - fitness_improvement_ratio)
                    cross_points = np.random.rand(self.dim) < cross_prob
                    trial = np.where(cross_points, mutant, current_population[i])
                    trial_fitness = func(trial)
                    evaluations += 1
                    if trial_fitness < current_fitness[i]:
                        current_fitness[i] = trial_fitness
                        current_population[i] = trial
                        success_count += 1
                population[:population_size] = current_population
                fitness[:population_size] = current_fitness

        exploration_weight = 0.5
        for _ in range(self.budget // (4 * initial_population_size)):
            adaptive_differential_evolution()
            for i in range(initial_population_size):
                if evaluations >= self.budget:
                    break
                improved = stochastic_local_search(population[i], step_size=0.1 * exploration_weight)
                improved_fitness = func(improved)
                evaluations += 1
                if improved_fitness < fitness[i]:
                    fitness[i] = improved_fitness
                    population[i] = improved
            exploration_weight *= 0.85

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]