import numpy as np

class DynamicAdaptiveMultiAgentHybrid:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize parameters
        population_size = 10 + 2 * self.dim
        F_base = 0.5  # Base differential weight
        CR_base = 0.9  # Base crossover probability
        local_search_prob = 0.2
        
        # Initialize population
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population = np.random.rand(population_size, self.dim) * (ub - lb) + lb
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size
        elite = population[np.argmin(fitness)]

        while evaluations < self.budget:
            # Calculate population diversity
            population_std = np.std(population, axis=0)
            diversity = np.mean(population_std / (ub - lb))

            # Adapt strategies based on phases of the search
            phase_ratio = evaluations / self.budget
            if phase_ratio < 0.5:
                F = F_base + 0.25 * np.tanh(5 * (0.5 - diversity))
                CR = CR_base - 0.15 * np.tanh(5 * (0.5 - diversity))
            else:
                F = F_base + 0.2 * np.cos(phase_ratio * np.pi)
                CR = CR_base - 0.05 * np.cos(phase_ratio * np.pi)
            
            # Adaptive inertia weight mechanism
            inertia_weight = 0.5 + 0.4 * np.cos(2 * np.pi * evaluations / self.budget)
            
            # Adjust population size adaptively
            population_size = int(population_size * (1.0 + 0.3 * np.sin(2 * np.pi * evaluations / self.budget)))
            population_size = min(max(5, population_size), 50)

            for i in range(population_size):
                # Differential mutation with adaptive F and inertia
                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = inertia_weight * population[i] + (1 - inertia_weight) * (population[a] + F * (population[b] - population[c]))
                mutant = np.clip(mutant, lb, ub)

                # Crossover with adaptive CR
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, population[i])

                # Evaluate trial vector
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < func(elite):
                        elite = trial

                # Neighborhood-based local search
                if np.random.rand() < local_search_prob and evaluations < self.budget:
                    neighbor_indices = np.random.choice(indices, 2, replace=False)
                    local_direction = population[neighbor_indices[0]] - population[neighbor_indices[1]]
                    adaptive_step_size = (ub - lb) * 0.05 * (0.5 + 0.5 * (fitness.mean() - fitness[i]) / (fitness.std() + 1e-12))
                    local_trial = population[i] + np.random.uniform(-adaptive_step_size, adaptive_step_size) * local_direction
                    local_trial = np.clip(local_trial, lb, ub)
                    local_fitness = func(local_trial)
                    evaluations += 1
                    if local_fitness < fitness[i]:
                        population[i] = local_trial
                        fitness[i] = local_fitness
                        if local_fitness < func(elite):
                            elite = local_trial

                if evaluations >= self.budget:
                    break

        # Return the best found solution
        return elite