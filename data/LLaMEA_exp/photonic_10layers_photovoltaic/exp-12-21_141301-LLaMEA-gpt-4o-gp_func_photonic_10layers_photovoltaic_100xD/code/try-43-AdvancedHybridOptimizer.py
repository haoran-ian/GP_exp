import numpy as np

class AdvancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize parameters
        initial_pop_size = 30
        F1_init = 0.7  # Differential weight for first strategy
        F2_init = 0.5  # Differential weight for second strategy
        CR_init = 0.8  # Initial crossover probability
        T0 = 1.0  # Initial temperature for simulated annealing
        alpha = 0.92  # Cooling rate
        elitism_rate = 0.25  # Elitism rate
        memory_size = 15  # Memory size for adaptive learning
        memory_F = np.full(memory_size, F1_init)  # Memory for differential weights
        
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

        # Multi-scale exploration with additional strategy
        exploration_scales = [0.7, 0.4, 0.1]

        while num_evaluations < self.budget:
            # Adaptive population reduction
            reduction_factor = 0.9 - 0.004 * (num_evaluations / self.budget)
            pop_size = max(5, int(initial_pop_size * reduction_factor))

            # Adaptive learning rate using memory
            if np.random.rand() < 0.4:
                F1 = memory_F[memory_idx]
                F2 = F2_init * (1 - (num_evaluations / (self.budget * 1.5))) + 0.1
            else:
                F1 = F1_init * (1 - (num_evaluations / (self.budget * 1.8))) + 0.2
                F2 = memory_F[memory_idx]

            # Dynamic crossover strategy
            CR = CR_init * (1 - (num_evaluations / self.budget) ** 0.5)
            
            memory_F[memory_idx] = (F1 + F2) / 2
            memory_idx = (memory_idx + 1) % memory_size

            # Elitism: protect the top solutions
            elite_count = int(elitism_rate * pop_size)
            elite_indices = fitness.argsort()[:elite_count]
            elites = population[elite_indices]

            for i in range(pop_size):
                if i in elite_indices:
                    continue

                # Differential Evolution mutation and crossover with two strategies
                idxs = np.random.choice(pop_size, 3, replace=False)
                x0, x1, x2 = population[idxs]
                mutant1 = x0 + F1 * (x1 - x2)
                mutant1 = np.clip(mutant1, func.bounds.lb, func.bounds.ub)
                
                idxs = np.random.choice(pop_size, 3, replace=False)
                x0, x1, x2 = population[idxs]
                mutant2 = x0 + F2 * (x1 - x2)
                mutant2 = np.clip(mutant2, func.bounds.lb, func.bounds.ub)

                crossover_mask = np.random.rand(self.dim) < CR
                trial1 = np.where(crossover_mask, mutant1, population[i])
                trial2 = np.where(crossover_mask, mutant2, population[i])

                # Multi-scale exploration
                scale = exploration_scales[np.random.randint(len(exploration_scales))]
                trial1 = scale * trial1 + (1 - scale) * population[i]
                trial2 = scale * trial2 + (1 - scale) * population[i]

                # Evaluate trial vectors
                trial1_fitness = func(trial1)
                trial2_fitness = func(trial2)
                num_evaluations += 2

                # Simulated Annealing acceptance
                if trial1_fitness < fitness[i] or trial2_fitness < fitness[i]:
                    if trial1_fitness < trial2_fitness:
                        selected_trial, selected_fitness = trial1, trial1_fitness
                    else:
                        selected_trial, selected_fitness = trial2, trial2_fitness

                    population[i] = selected_trial
                    fitness[i] = selected_fitness
                    if selected_fitness < best_fitness:
                        best_solution = selected_trial
                        best_fitness = selected_fitness
                else:
                    acceptance_prob1 = np.exp((fitness[i] - trial1_fitness) / (temperature + 0.1))
                    acceptance_prob2 = np.exp((fitness[i] - trial2_fitness) / (temperature + 0.1))
                    if np.random.rand() < acceptance_prob1:
                        population[i] = trial1
                        fitness[i] = trial1_fitness
                    elif np.random.rand() < acceptance_prob2:
                        population[i] = trial2
                        fitness[i] = trial2_fitness

                if num_evaluations >= self.budget - 1:
                    break

            # Reintroduce elites to maintain diversity
            population[:elite_count] = elites

            temperature *= alpha

        return best_solution