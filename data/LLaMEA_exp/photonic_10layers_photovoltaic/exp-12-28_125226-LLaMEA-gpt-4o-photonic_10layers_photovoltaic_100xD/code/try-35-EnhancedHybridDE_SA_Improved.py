import numpy as np

class EnhancedHybridDE_SA_Improved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.cr = 0.9
        self.f = 0.8
        self.init_temperature = 100
        self.cooling_rate = 0.99
        self.temperature = self.init_temperature
        self.eval_count = 0

    def opposition_based_learning(self, pop, lb, ub):
        opposite_pop = lb + ub - pop
        return np.clip(opposite_pop, lb, ub)

    def adaptive_cooling_schedule(self):
        self.temperature = self.init_temperature * (1 - self.eval_count / self.budget)

    def chaotic_search(self, candidate, lb, ub):
        beta = 0.7
        chaotic_step = beta * (2 * np.random.rand(self.dim) - 1)
        chaotic_candidate = np.clip(candidate + chaotic_step, lb, ub)
        return chaotic_candidate

    def dynamic_population_resizing(self):
        shrink_factor = 1 - (self.eval_count / self.budget)
        new_size = max(4, int(self.initial_population_size * shrink_factor))
        if new_size < self.population_size:
            self.population_size = new_size

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.rand(self.initial_population_size, self.dim) * (ub - lb) + lb
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += self.initial_population_size

        while self.eval_count < self.budget:
            new_population = []
            self.dynamic_population_resizing()
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                adaptive_f = self.f * (1 - self.eval_count / self.budget) + 0.1
                mutant = np.clip(a + adaptive_f * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < self.cr
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                self.eval_count += 1

                if trial_fitness < fitness[i] or np.random.rand() < np.exp(-(trial_fitness - fitness[i]) / self.temperature):
                    new_population.append(trial)
                    fitness[i] = trial_fitness
                else:
                    new_population.append(population[i])

                if self.eval_count >= self.budget:
                    break

            self.adaptive_cooling_schedule()
            opposite_population = self.opposition_based_learning(new_population, lb, ub)
            opposite_fitness = np.array([func(ind) for ind in opposite_population])
            self.eval_count += self.population_size

            for j in range(self.population_size):
                if opposite_fitness[j] < fitness[j]:
                    new_population[j] = opposite_population[j]
                    fitness[j] = opposite_fitness[j]

            population = np.array(new_population)

            # Apply chaotic search on the best candidate
            best_idx = np.argmin(fitness)
            best_candidate = population[best_idx]
            chaotic_candidate = self.chaotic_search(best_candidate, lb, ub)
            chaotic_fitness = func(chaotic_candidate)
            self.eval_count += 1

            if chaotic_fitness < fitness[best_idx]:
                population[best_idx] = chaotic_candidate
                fitness[best_idx] = chaotic_fitness

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]