import numpy as np

class HybridMetaheuristicV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.temperature = 1.0
        self.cooling_rate = 0.95  # Adjusted for smoother cooling
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.diversity_threshold = 0.1  # Initial diversity threshold
        self.phase_switch_threshold = budget // 3  # Switch between phases

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size
        phase = 1

        while budget_used < self.budget:
            # Adaptive mutation and crossover based on phase
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                adaptive_mutation_factor = self.mutation_factor / (1 + 0.01 * budget_used)
                if phase == 2:  # Enhance exploration
                    adaptive_mutation_factor *= 1.5
                mutant = np.clip(a + adaptive_mutation_factor * (b - c), lb, ub)
                dynamic_crossover_rate = self.crossover_rate + 0.05 * np.sin(budget_used / self.budget * np.pi)
                if phase == 3:  # Enhance exploitation
                    dynamic_crossover_rate += 0.1
                crossover = np.random.rand(self.dim) < dynamic_crossover_rate
                trial = np.where(crossover, mutant, population[i])

                # Simulated Annealing: accept based on Metropolis criterion
                trial_fitness = func(trial)
                budget_used += 1
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

                if budget_used >= self.budget:
                    break
            
            # Cool down temperature
            self.temperature *= self.cooling_rate

            # Phase transition logic
            if budget_used > self.phase_switch_threshold and phase == 1:
                phase = 2  # Explore more
            elif budget_used > 2 * self.phase_switch_threshold and phase == 2:
                phase = 3  # Exploit more
            
            # Diversity monitoring for maintaining exploration potential
            cluster_centers = np.mean(population, axis=0)
            diversity = np.linalg.norm(population - cluster_centers, axis=1).mean()
            if diversity < self.diversity_threshold * (ub - lb).mean():
                self.mutation_factor *= 1.1  # Encourage exploration
                self.diversity_threshold *= 0.9  # Gradually tighten threshold

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]