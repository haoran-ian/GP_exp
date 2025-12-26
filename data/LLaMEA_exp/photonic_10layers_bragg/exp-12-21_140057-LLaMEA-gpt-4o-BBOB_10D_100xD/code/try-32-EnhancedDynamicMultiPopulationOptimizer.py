import numpy as np

class EnhancedDynamicMultiPopulationOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialization
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 10 * self.dim
        num_populations = 3
        sub_population_size = population_size // num_populations
        populations = [np.random.uniform(lb, ub, (sub_population_size, self.dim)) for _ in range(num_populations)]
        fitness = [np.array([func(ind) for ind in pop]) for pop in populations]
        memory = []
        best_solution, best_fitness = None, np.inf
        evaluations = population_size

        # Dynamic Parameters
        F_base = 0.8
        Cr_base = 0.9
        temp = 1.0
        cooling_rate = 0.95
        memory_size = 5

        def differential_evolution(pop, fit):
            nonlocal F_base, Cr_base
            for i in range(sub_population_size):
                indices = list(range(sub_population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = pop[a] + F_base * (pop[b] - pop[c]) * (1 + 0.5 * np.random.rand())
                mutant = np.clip(mutant, lb, ub)
                Cr_dynamic = Cr_base * (1 - evaluations / self.budget)
                cross_points = np.random.rand(self.dim) < Cr_dynamic
                trial = np.where(cross_points, mutant, pop[i])
                trial_fitness = func(trial)

                if trial_fitness < fit[i]:
                    pop[i] = trial
                    fit[i] = trial_fitness

        def adaptive_neighborhood_search(ind, fit):
            new_solution = ind + np.random.normal(0, 0.1, self.dim)
            new_solution = np.clip(new_solution, lb, ub)
            new_fitness = func(new_solution)
            if new_fitness < fit:
                return new_solution, new_fitness
            return ind, fit

        def calculate_diversity(pop):
            return np.mean(np.std(pop, axis=0))

        # Search loop
        while evaluations < self.budget:
            for j in range(num_populations):
                if evaluations >= self.budget:
                    break

                diversity = calculate_diversity(populations[j])
                if diversity < 0.1:
                    F_base = max(0.5, F_base + 0.2 * np.random.rand())
                    Cr_base = min(1.0, Cr_base - 0.1 * np.random.rand())

                differential_evolution(populations[j], fitness[j])
                evaluations += sub_population_size

                for i in range(sub_population_size):
                    if evaluations >= self.budget:
                        break

                    populations[j][i], fitness[j][i] = adaptive_neighborhood_search(populations[j][i], fitness[j][i])
                    if fitness[j][i] < best_fitness:
                        best_solution = populations[j][i]
                        best_fitness = fitness[j][i]

                    new_solution = populations[j][i] + np.random.normal(0, 0.1, self.dim)
                    new_solution = np.clip(new_solution, lb, ub)
                    new_fitness = func(new_solution)
                    delta = new_fitness - fitness[j][i]

                    if delta < 0 or np.exp(-delta / temp) > np.random.rand():
                        populations[j][i] = new_solution
                        fitness[j][i] = new_fitness
                        if new_fitness < best_fitness:
                            best_solution = new_solution
                            best_fitness = new_fitness

                    evaluations += 1

                temp *= cooling_rate

            # Manage memory and adapt
            memory.append(best_fitness)
            if len(memory) > memory_size:
                memory.pop(0)
            if len(memory) == memory_size and memory[-1] >= memory[0]:
                F_base = max(0.5, F_base + 0.1)
                Cr_base = min(1.0, Cr_base - 0.05)

        return best_solution