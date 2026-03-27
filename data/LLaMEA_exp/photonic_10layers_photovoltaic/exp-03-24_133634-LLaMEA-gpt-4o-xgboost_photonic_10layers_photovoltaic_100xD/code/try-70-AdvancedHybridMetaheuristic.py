import numpy as np

class AdvancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.temperature = 1.0
        self.cooling_rate = 0.95  # Slightly less aggressive cooling
        self.mutation_factor = 0.9
        self.crossover_rate = 0.7
        self.agents = 5  # Number of parallel agents

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        populations = [np.random.uniform(lb, ub, (self.population_size, self.dim)) for _ in range(self.agents)]
        fitnesses = [np.array([func(ind) for ind in pop]) for pop in populations]
        budget_used = self.population_size * self.agents

        while budget_used < self.budget:
            for agent_idx in range(self.agents):
                population = populations[agent_idx]
                fitness = fitnesses[agent_idx]

                for i in range(self.population_size):
                    idxs = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                    adaptive_mutation_factor = self.mutation_factor / (1 + 0.01 * budget_used)
                    mutant = np.clip(a + adaptive_mutation_factor * (b - c), lb, ub)
                    dynamic_crossover_rate = self.crossover_rate + 0.05 * np.sin(budget_used / self.budget * np.pi)
                    crossover = np.random.rand(self.dim) < dynamic_crossover_rate
                    trial = np.where(crossover, mutant, population[i])

                    trial_fitness = func(trial)
                    budget_used += 1
                    if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temperature):
                        population[i] = trial
                        fitness[i] = trial_fitness

                    if budget_used >= self.budget:
                        break

                # Cool down temperature
                self.temperature *= self.cooling_rate

                # Inter-agent communication and competition
                if agent_idx > 0 and np.min(fitnesses[agent_idx]) < np.min(fitnesses[agent_idx - 1]):
                    populations[agent_idx - 1] = np.copy(population)
                    fitnesses[agent_idx - 1] = np.copy(fitness)

            # Adaptive mutation factor adjustment based on diversity
            all_populations = np.vstack(populations)
            global_diversity = np.std(all_populations, axis=0).mean()
            if global_diversity < 0.15 * (ub - lb).mean():
                self.mutation_factor *= 1.1

        best_agent_idx = np.argmin([np.min(fit) for fit in fitnesses])
        best_idx = np.argmin(fitnesses[best_agent_idx])
        return populations[best_agent_idx][best_idx], fitnesses[best_agent_idx][best_idx]