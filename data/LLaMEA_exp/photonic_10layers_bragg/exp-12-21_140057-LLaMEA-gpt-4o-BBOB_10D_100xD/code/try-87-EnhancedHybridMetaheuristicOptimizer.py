import numpy as np

class EnhancedHybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 10 * self.dim
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        evaluations = population_size

        F_base = 0.8
        Cr_base = 0.9
        temp = 1.0
        cooling_rate = 0.98
        neighborhood_radius = 0.1
        diversity_threshold = 0.1

        def differential_evolution(pop, fit):
            nonlocal F_base, Cr_base
            for i in range(population_size):
                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = pop[a] + F_base * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                Cr_dynamic = Cr_base * (1 - evaluations / self.budget)
                cross_points = np.random.rand(self.dim) < Cr_dynamic
                trial = np.where(cross_points, mutant, pop[i])
                trial_fitness = func(trial)

                if trial_fitness < fit[i]:
                    pop[i] = trial
                    fit[i] = trial_fitness

        def adaptive_tabu_search(ind, fit):
            candidates = [ind + np.random.normal(0, neighborhood_radius, self.dim) for _ in range(3)]
            candidates = [np.clip(c, lb, ub) for c in candidates]
            candidates_fitness = [func(c) for c in candidates]
            best_candidate_idx = np.argmin(candidates_fitness)
            if candidates_fitness[best_candidate_idx] < fit:
                return candidates[best_candidate_idx], candidates_fitness[best_candidate_idx]
            return ind, fit

        def calculate_diversity(pop):
            return np.mean(np.std(pop, axis=0))

        while evaluations < self.budget:
            diversity = calculate_diversity(population)
            if diversity < diversity_threshold:
                F_base = np.random.uniform(0.5, 1.0)
                Cr_base = np.random.uniform(0.7, 1.0)

            differential_evolution(population, fitness)
            evaluations += population_size

            for i in range(population_size):
                if evaluations >= self.budget:
                    break
                
                population[i], fitness[i] = adaptive_tabu_search(population[i], fitness[i])
                if fitness[i] < best_fitness:
                    best_solution = population[i]
                    best_fitness = fitness[i]

                new_solution = population[i] + np.random.normal(0, 0.1, self.dim)
                new_solution = np.clip(new_solution, lb, ub)
                new_fitness = func(new_solution)
                delta = new_fitness - fitness[i]

                if delta < 0 or np.exp(-delta / temp) > np.random.rand():
                    population[i] = new_solution
                    fitness[i] = new_fitness
                    if new_fitness < best_fitness:
                        best_solution = new_solution
                        best_fitness = new_fitness

                evaluations += 1

            temp *= cooling_rate
            neighborhood_radius *= 0.9

        return best_solution