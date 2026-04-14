import numpy as np

class Adaptive_Resizing_Chaotic_Search:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.base_population_size = min(50, 5 * dim)
        self.populations = [np.random.uniform(size=(self.base_population_size, self.dim)) for _ in range(3)]
        self.fitness = [np.full(self.base_population_size, np.inf) for _ in range(3)]
        self.CR = [0.9, 0.7, 0.5]
        self.F = [0.8, 0.9, 1.0]
        self.evaluations = 0
        self.memory = []
        self.memory_decay_rate = 0.98
        self.chaotic_factor = 0.5
        self.dynamic_chaotic_factor = 0.5

    def init_populations(self, bounds):
        lower_bound = bounds.lb
        upper_bound = bounds.ub
        for pop in self.populations:
            pop[:] = np.random.uniform(lower_bound, upper_bound, (self.base_population_size, self.dim))
        for fit in self.fitness:
            fit.fill(np.inf)

    def adapt_parameters(self):
        self.dynamic_chaotic_factor = 4 * self.dynamic_chaotic_factor * (1 - self.dynamic_chaotic_factor)
        self.chaotic_factor += 0.0002  # Adjusted chaotic factor incrementally

    def differential_evolution(self, population, target_idx, bounds):
        idxs = [idx for idx in range(len(population)) if idx != target_idx]
        a, b, c = population[np.random.choice(idxs, 3, replace=False)]
        mutant = np.clip(a + self.F[0] * (b - c), bounds.lb, bounds.ub)
        cross_points = np.random.rand(self.dim) < self.CR[0]
        trial = np.where(cross_points, mutant, population[target_idx])
        return trial

    def adaptive_local_search(self, candidate, bounds, success_rate=0.5):
        intensity = 0.15 * (1 - self.evaluations / self.budget) * success_rate * self.dynamic_chaotic_factor  # Adjusting intensity
        perturbation = np.clip(candidate + np.random.normal(0, intensity, self.dim), bounds.lb, bounds.ub)
        return perturbation

    def intelligent_memory(self, candidate, fitness, bounds):
        if self.memory:
            best_mem = min(self.memory, key=lambda x: x[1])[0]
            direction = best_mem - candidate
            perturbation = np.clip(candidate + self.dynamic_chaotic_factor * direction, bounds.lb, bounds.ub)
            return perturbation
        return candidate

    def decay_memory(self):
        self.memory = [(solution, fit * self.memory_decay_rate) for solution, fit in self.memory]

    def resize_population(self, pop_idx, success_rate):
        if success_rate < 0.1:
            new_size = max(5, int(self.base_population_size * 0.9))
            self.populations[pop_idx] = self.populations[pop_idx][:new_size]
            self.fitness[pop_idx] = self.fitness[pop_idx][:new_size]
        elif success_rate > 0.7:
            new_size = min(100, int(self.base_population_size * 1.1))
            new_pop = np.random.uniform(size=(new_size - len(self.populations[pop_idx]), self.dim))
            self.populations[pop_idx] = np.vstack((self.populations[pop_idx], new_pop))
            self.fitness[pop_idx] = np.append(self.fitness[pop_idx], np.full(new_size - len(self.fitness[pop_idx]), np.inf))
        self.base_population_size = len(self.populations[pop_idx])

    def __call__(self, func):
        self.init_populations(func.bounds)
        for pop_idx, pop in enumerate(self.populations):
            for i in range(self.base_population_size):
                self.fitness[pop_idx][i] = func(pop[i])
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    return pop[np.argmin(self.fitness[pop_idx])]

        while self.evaluations < self.budget:
            self.adapt_parameters()
            self.decay_memory()

            for pop_idx, pop in enumerate(self.populations):
                success_count = 0
                for i in range(len(pop)):
                    trial = self.differential_evolution(pop, i, func.bounds)
                    trial_fitness = func(trial)
                    self.evaluations += 1

                    if trial_fitness < self.fitness[pop_idx][i]:
                        pop[i] = trial
                        self.fitness[pop_idx][i] = trial_fitness
                        self.memory.append((trial, trial_fitness))
                        success_count += 1

                    if np.random.rand() < 0.1:
                        local_candidate = self.adaptive_local_search(pop[i], func.bounds, success_rate=success_count/len(pop))
                        local_fitness = func(local_candidate)
                        self.evaluations += 1
                        if local_fitness < self.fitness[pop_idx][i]:
                            pop[i] = local_candidate
                            self.fitness[pop_idx][i] = local_fitness
                            self.memory.append((local_candidate, local_fitness))

                    if self.evaluations >= self.budget:
                        break

                self.resize_population(pop_idx, success_count / len(pop))

        best_population_idx = np.argmin([np.min(fit) for fit in self.fitness])
        return self.populations[best_population_idx][np.argmin(self.fitness[best_population_idx])]