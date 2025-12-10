import numpy as np

class DynamicAdaptiveMultiAgentHybrid:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population_size = 10 + 2 * self.dim
        F_base = 0.5
        CR_base = 0.9
        local_search_prob = 0.2
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population = np.random.rand(population_size, self.dim) * (ub - lb) + lb
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size
        elite = population[np.argmin(fitness)]

        while evaluations < self.budget:
            population_std = np.std(population, axis=0)
            diversity = np.mean(population_std / (ub - lb))
            phase_ratio = evaluations / self.budget
            if phase_ratio < 0.5:
                F = F_base + 0.3 * np.tanh(5 * (0.5 - diversity))
                CR = CR_base - 0.2 * np.tanh(5 * (0.5 - diversity))
            else:
                F = F_base + 0.1 * np.cos(phase_ratio * np.pi)
                CR = CR_base - 0.1 * np.cos(phase_ratio * np.pi)

            swarm_factor = 0.5 + 0.5 * np.sin(2 * np.pi * evaluations / self.budget)
            population_size = min(max(5, int(population_size * (1.1 - diversity))), 50)

            for i in range(population_size):
                indices = list(range(population_size))
                indices.remove(i)
                np.random.shuffle(indices)
                a, b, c = indices[:3]
                mutant = swarm_factor * (population[a] + F * (population[b] - population[c]))
                mutant = np.clip(mutant, lb, ub)

                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, population[i])

                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < func(elite):
                        elite = trial

                if np.random.rand() < local_search_prob and evaluations < self.budget:
                    neighbors = np.random.choice(range(population_size), 3, replace=False)
                    local_search_step = (population[neighbors[0]] - population[neighbors[1]]) * np.random.rand()
                    local_trial = population[i] + local_search_step
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

        return elite