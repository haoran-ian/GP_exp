import numpy as np

class AdvancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize parameters
        population_size = 10 + 2 * self.dim
        F_base = 0.5  # Base differential weight
        CR_base = 0.9  # Base crossover probability

        # Initialize adaptive parameters
        local_search_base_prob = 0.2
        exploit_prob = 0.1
        
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

            # Adapt strategies based on diversity feedback
            if diversity < 0.3:
                F = F_base * 1.2
                CR = CR_base * 0.8
                local_search_prob = local_search_base_prob * 1.5
            else:
                F = F_base
                CR = CR_base
                local_search_prob = local_search_base_prob

            # Adaptive swarm factor for exploration
            swarm_factor = 0.5 + 0.5 * np.sin(2 * np.pi * evaluations / self.budget)
            
            # Adjust population size adaptively based on diversity
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

                # Enhanced Adaptive local search with exploitation feedback
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

                # Additional exploitation phase inspired by diversity
                if np.random.rand() < exploit_prob and evaluations < self.budget:
                    direction = (elite - population[i])
                    exploit_step = np.random.rand() * direction
                    exploit_trial = population[i] + exploit_step
                    exploit_trial = np.clip(exploit_trial, lb, ub)
                    exploit_fitness = func(exploit_trial)
                    evaluations += 1
                    if exploit_fitness < fitness[i]:
                        population[i] = exploit_trial
                        fitness[i] = exploit_fitness
                        if exploit_fitness < func(elite):
                            elite = exploit_trial

                if evaluations >= self.budget:
                    break

        # Return the best found solution
        return elite