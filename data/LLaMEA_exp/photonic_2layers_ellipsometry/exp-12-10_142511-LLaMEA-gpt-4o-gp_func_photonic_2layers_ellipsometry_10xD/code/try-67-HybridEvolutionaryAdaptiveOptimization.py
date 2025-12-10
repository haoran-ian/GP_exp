import numpy as np

class HybridEvolutionaryAdaptiveOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize parameters
        init_population_size = 10 + 2 * self.dim
        F_base = 0.5  # Base differential weight
        CR_base = 0.9  # Base crossover probability
        local_search_prob = 0.2  # Probability of performing local search
        elite_preservation_ratio = 0.05  # Ratio of elite individuals preserved

        # Initialize population
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population = np.random.rand(init_population_size, self.dim) * (ub - lb) + lb
        fitness = np.array([func(ind) for ind in population])
        evaluations = init_population_size
        elite = population[np.argmin(fitness)]

        while evaluations < self.budget:
            # Calculate population diversity
            population_std = np.std(population, axis=0)
            diversity = np.mean(population_std / (ub - lb))

            # Adapt strategies based on phases of the search
            phase_ratio = evaluations / self.budget
            if phase_ratio < 0.5:
                F = F_base + 0.3 * np.tanh(5 * (0.5 - diversity))
                CR = CR_base - 0.2 * np.tanh(5 * (0.5 - diversity))
            else:
                F = F_base + 0.2 * (1 - diversity) * np.cos(phase_ratio * np.pi)
                CR = CR_base - 0.2 * (1 - diversity) * np.cos(phase_ratio * np.pi)

            # Adaptive swarm factor for exploration
            swarm_factor = 0.5 + 0.5 * np.sin(2 * np.pi * evaluations / self.budget)

            # Adjust population size adaptively
            preserved_elites = int(elite_preservation_ratio * init_population_size)
            population_size = min(max(5, int(init_population_size * (1.1 - diversity))), 50)
            population_size = max(population_size, preserved_elites + 2)

            # Preserve top elite individuals
            elite_indices = np.argsort(fitness)[:preserved_elites]
            new_population = population[elite_indices].copy()
            new_fitness = fitness[elite_indices].copy()

            while len(new_population) < population_size:
                for i in range(len(population) - preserved_elites):
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
                        new_population = np.vstack((new_population, trial))
                        new_fitness = np.append(new_fitness, trial_fitness)
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
                            new_population = np.vstack((new_population, local_trial))
                            new_fitness = np.append(new_fitness, local_fitness)
                            if local_fitness < func(elite):
                                elite = local_trial

                    if evaluations >= self.budget or len(new_population) >= population_size:
                        break

            population = new_population
            fitness = new_fitness

        # Return the best found solution
        return elite