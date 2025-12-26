import numpy as np

class EnhancedAdaptiveSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize parameters
        initial_pop_size = 20
        F_init = 0.8  # Initial differential weight
        CR_init = 0.9  # Initial crossover probability
        T0 = 1.0  # Initial temperature for simulated annealing
        alpha = 0.95  # Cooling rate
        elitism_rate = 0.2  # Increased elitism rate
        memory_size = 10  # Memory size for adaptive learning
        memory = np.full(memory_size, F_init)  # Memory for differential weight

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

        # Multi-scale exploration
        exploration_scales = [0.8, 0.5, 0.2]

        while num_evaluations < self.budget:
            # Adaptive population reduction
            reduction_factor = 0.95 - 0.005 * (num_evaluations / self.budget)
            pop_size = max(4, int(initial_pop_size * reduction_factor))

            # Adaptive learning rate using memory
            F = memory[memory_idx] if np.random.rand() < 0.3 else F_init * (1 - (num_evaluations / (self.budget * 2.0))) + 0.15

            # Dynamic crossover strategy
            CR = CR_init * (1 - np.sqrt(num_evaluations / self.budget))
            
            memory[memory_idx] = F
            memory_idx = (memory_idx + 1) % memory_size

            # Elitism: protect the top solutions
            elite_count = int(elitism_rate * pop_size)
            elite_indices = fitness.argsort()[:elite_count]
            elites = population[elite_indices]

            for i in range(pop_size):
                if i in elite_indices:
                    continue

                # Differential Evolution mutation and crossover
                idxs = np.random.choice(pop_size, 3, replace=False)
                x0, x1, x2 = population[idxs]
                mutant = x0 + F * (x1 - x2)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                crossover_mask = np.random.rand(self.dim) < CR
                trial = np.where(crossover_mask, mutant, population[i])

                # Multi-scale exploration
                scale = exploration_scales[np.random.randint(len(exploration_scales))]
                trial = scale * trial + (1 - scale) * population[i]

                # Swarm-inspired local search
                local_scale = 0.1 * np.random.randn(self.dim)
                local_search = trial + local_scale * (best_solution - trial)
                local_search = np.clip(local_search, func.bounds.lb, func.bounds.ub)

                # Evaluate trial vector
                trial_fitness = func(trial)
                local_fitness = func(local_search)
                num_evaluations += 2

                # Simulated Annealing acceptance
                if trial_fitness < fitness[i] or local_fitness < trial_fitness:
                    better_trial = trial if trial_fitness < local_fitness else local_search
                    better_fitness = min(trial_fitness, local_fitness)
                    population[i] = better_trial
                    fitness[i] = better_fitness
                    if better_fitness < best_fitness:
                        best_solution = better_trial
                        best_fitness = better_fitness
                else:
                    acceptance_prob = np.exp((fitness[i] - trial_fitness) / (temperature + 0.1))
                    if np.random.rand() < acceptance_prob:
                        population[i] = trial
                        fitness[i] = trial_fitness

                if num_evaluations >= self.budget - 1:
                    break

            # Reintroduce elites to maintain diversity
            population[:elite_count] = elites

            temperature *= alpha

        return best_solution