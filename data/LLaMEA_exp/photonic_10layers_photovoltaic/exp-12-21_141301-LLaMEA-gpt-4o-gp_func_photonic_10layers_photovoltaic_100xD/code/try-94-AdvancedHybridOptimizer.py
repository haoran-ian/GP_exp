import numpy as np

class AdvancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize parameters
        initial_pop_size = 20
        F_init = 0.8
        CR_init = 0.9
        T0 = 1.0
        alpha = 0.95
        memory_size = 10
        memory = np.full(memory_size, F_init)
        beta = 0.55

        # Initialize population
        population = np.random.rand(initial_pop_size, self.dim)
        for i in range(initial_pop_size):
            population[i] = func.bounds.lb + population[i] * (func.bounds.ub - func.bounds.lb)

        # Evaluate initial population
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = initial_pop_size

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        temperature = T0
        memory_idx = 0

        exploration_scales = [0.8, 0.5, 0.2]

        while num_evaluations < self.budget:
            reduction_factor = 0.95 - 0.005 * (num_evaluations / self.budget)
            pop_size = max(4, int(initial_pop_size * reduction_factor))

            if np.random.rand() < 0.3:
                F = memory[memory_idx]
            else:
                diversity = np.std(population, axis=0).mean() / (func.bounds.ub - func.bounds.lb).mean()
                F = F_init * (1 - (num_evaluations / (self.budget * 2.0))) + 0.15 * diversity

            CR = CR_init * (1 - np.sqrt(num_evaluations / self.budget)) * beta

            memory[memory_idx] = F
            memory_idx = (memory_idx + 1) % memory_size

            elite_count = int((0.2 + 0.2 * np.sin(0.01 * num_evaluations)) * pop_size)
            elite_indices = fitness.argsort()[:elite_count]
            elites = population[elite_indices]

            for i in range(pop_size):
                if i in elite_indices:
                    continue

                idxs = np.random.choice(pop_size, 3, replace=False)
                x0, x1, x2 = population[idxs]
                mutant = x0 + F * (x1 - x2)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                crossover_mask = np.random.rand(self.dim) < CR
                trial = np.where(crossover_mask, mutant, population[i])

                scale = exploration_scales[np.random.randint(len(exploration_scales))]
                trial = scale * trial + (1 - scale) * population[i]

                trial_fitness = func(trial)
                num_evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness
                else:
                    acceptance_prob = np.exp((fitness[i] - trial_fitness) / (temperature + 0.1))
                    if np.random.rand() < acceptance_prob:
                        population[i] = trial
                        fitness[i] = trial_fitness

                if num_evaluations >= self.budget - 1:
                    break

            population[:elite_count] = elites

            temperature *= np.log1p(alpha)

        return best_solution