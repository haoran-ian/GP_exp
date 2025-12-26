import numpy as np

class ImprovedDynamicPhaseBasedSynergisticEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize parameters
        population_size = 12 + 2 * self.dim
        F_base = 0.5
        CR_base = 0.9
        local_search_prob = 0.3
        archive_size = population_size // 2
        
        # Initialize population and archive
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population = np.random.rand(population_size, self.dim) * (ub - lb) + lb
        archive = np.empty((0, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size
        elite = population[np.argmin(fitness)]

        while evaluations < self.budget:
            # Calculate diversity
            population_std = np.std(population, axis=0)
            diversity = np.mean(population_std / (ub - lb))

            # Phase-specific adaptation
            phase_ratio = evaluations / self.budget
            F = F_base + 0.4 * np.tanh(5 * (0.5 - diversity))
            CR = CR_base - 0.3 * np.tanh(5 * (0.5 - diversity))
            
            if phase_ratio >= 0.7:
                F = F_base + 0.2 * np.sin(phase_ratio * np.pi)
                CR = CR_base - 0.1 * np.sin(phase_ratio * np.pi)
            
            # Adaptive swarm factor
            swarm_factor = 0.5 + 0.5 * np.cos(2 * np.pi * evaluations / self.budget)

            # Adjust population size adaptively
            population_size = min(max(5, int(population_size * (1.1 - diversity))), 60)

            for i in range(population_size):
                # Differential mutation with archive
                indices = list(range(population_size))
                indices.remove(i)
                a, b = np.random.choice(indices, 2, replace=False)
                c = np.random.choice(archive_size) if archive.size > 0 else np.random.choice(indices)
                
                mutant = swarm_factor * (population[a] + F * (population[b] - (archive[c] if archive.size > 0 else population[c])))
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, population[i])

                # Evaluate trial vector
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    archive = np.vstack((archive, population[i]))
                    if archive.shape[0] > archive_size:
                        archive = archive[np.random.choice(archive.shape[0], archive_size, replace=False)]
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

                if evaluations >= self.budget:
                    break

        # Return the best found solution
        return elite