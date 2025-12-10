import numpy as np

class EnhancedDynamicAdaptiveMultiAgentHybrid:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize parameters
        population_size = 10 + 2 * self.dim
        F_base = 0.5  # Base differential weight
        CR_base = 0.9  # Base crossover probability
        local_search_prob = 0.25  # Probability of performing local search
        multi_modality_factor = 0.1

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

            # Dynamic feedback-driven strategy adaptation
            phase_ratio = evaluations / self.budget
            feedback = np.tanh(5 * (0.5 - diversity))
            F = F_base + 0.3 * feedback * (1 - phase_ratio)
            CR = CR_base - 0.2 * feedback * phase_ratio

            # Adaptive swarm factor considering multi-modal exploration
            swarm_factor = 0.5 + 0.5 * np.sin(2 * np.pi * evaluations / self.budget)
            multi_mode_explore = (np.random.rand() < multi_modality_factor)

            # Adjust population size adaptively
            population_size = min(max(5, int(population_size * (1.1 - diversity))), 50)

            for i in range(population_size):
                # Differential mutation with adaptive F
                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = swarm_factor * (population[a] + F * (population[b] - population[c]))
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

                # Enhanced Adaptive local search
                if np.random.rand() < local_search_prob and evaluations < self.budget:
                    adaptive_step_size = (ub - lb) * 0.1 * (0.5 + 0.5 * (fitness.mean() - fitness[i]) / (fitness.std() + 1e-12))
                    local_trial = population[i] + np.random.uniform(-adaptive_step_size, adaptive_step_size)
                    local_trial = np.clip(local_trial, lb, ub)
                    local_fitness = func(local_trial)
                    evaluations += 1
                    if local_fitness < fitness[i]:
                        population[i] = local_trial
                        fitness[i] = local_fitness
                        if local_fitness < func(elite):
                            elite = local_trial

                # Multi-modal exploration
                if multi_mode_explore and evaluations < self.budget:
                    random_shift = np.random.uniform(-0.1, 0.1, self.dim) * (ub - lb)
                    multi_trial = population[i] + random_shift
                    multi_trial = np.clip(multi_trial, lb, ub)
                    multi_fitness = func(multi_trial)
                    evaluations += 1
                    if multi_fitness < fitness[i]:
                        population[i] = multi_trial
                        fitness[i] = multi_fitness
                        if multi_fitness < func(elite):
                            elite = multi_trial

                if evaluations >= self.budget:
                    break

        # Return the best found solution
        return elite