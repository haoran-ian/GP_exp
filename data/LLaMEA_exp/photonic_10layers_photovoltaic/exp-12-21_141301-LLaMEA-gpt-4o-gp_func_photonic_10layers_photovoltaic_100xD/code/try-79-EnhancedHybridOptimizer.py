import numpy as np

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        initial_pop_size = 30
        F_init = 0.9  # Initial differential weight
        CR_init = 0.85  # Initial crossover probability
        T0 = 1.0  # Initial temperature for simulated annealing
        alpha = 0.93  # Cooling rate
        elitism_rate = 0.1
        memory_size = 15  # Expanded memory size
        memory = np.full(memory_size, F_init)
        beta = 0.6

        population = np.random.rand(initial_pop_size, self.dim)
        for i in range(initial_pop_size):
            population[i] = func.bounds.lb + population[i] * (func.bounds.ub - func.bounds.lb)
        
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = initial_pop_size

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        temperature = T0
        memory_idx = 0

        exploration_scales = [0.8, 0.4, 0.1]

        while num_evaluations < self.budget:
            pop_size = max(4, int(initial_pop_size * (0.95 - 0.005 * (num_evaluations / self.budget))))

            if np.random.rand() < 0.35:
                F = memory[memory_idx]
            else:
                F = F_init * (1 - (num_evaluations / (self.budget * 2.0))) + 0.2

            CR = CR_init * (1 - np.power(num_evaluations / self.budget, beta))
            
            memory[memory_idx] = F
            memory_idx = (memory_idx + 1) % memory_size

            elite_count = int(elitism_rate * pop_size)
            elite_indices = fitness.argsort()[:elite_count]
            elites = population[elite_indices]

            for i in range(pop_size):
                if i in elite_indices:
                    continue

                idxs = np.random.choice(pop_size, 3, replace=False)
                x0, x1, x2 = population[idxs]
                gradient_mutant = x0 + F * (x1 - x2)
                gradient_mutant = np.clip(gradient_mutant, func.bounds.lb, func.bounds.ub)

                idxs_random = np.random.choice(pop_size, 3, replace=False)
                x_rand_0, x_rand_1, x_rand_2 = population[idxs_random]
                random_mutant = x_rand_0 + F * (x_rand_1 - x_rand_2)
                random_mutant = np.clip(random_mutant, func.bounds.lb, func.bounds.ub)

                crossover_mask = np.random.rand(self.dim) < CR
                trial_gradient = np.where(crossover_mask, gradient_mutant, population[i])
                trial_random = np.where(crossover_mask, random_mutant, population[i])
                
                trial = (trial_gradient + trial_random) / 2.0

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
            temperature *= alpha

        return best_solution