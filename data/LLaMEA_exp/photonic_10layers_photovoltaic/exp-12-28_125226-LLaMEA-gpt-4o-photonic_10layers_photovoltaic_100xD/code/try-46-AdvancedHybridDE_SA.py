import numpy as np

class AdvancedHybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.cr = 0.9
        self.f_min = 0.5
        self.f_max = 0.9
        self.init_temperature = 100
        self.cooling_rate = 0.99
        self.temperature = self.init_temperature
        self.eval_count = 0
        self.elite_fraction = 0.1

    def adaptive_scaling_factor(self):
        return self.f_min + (self.f_max - self.f_min) * (1 - self.eval_count / self.budget)**2

    def dual_cooling_schedule(self):
        self.temperature = self.init_temperature * (self.cooling_rate ** (self.eval_count / self.budget))

    def opposition_based_learning(self, pop, lb, ub):
        opposite_pop = lb + ub - pop
        return np.clip(opposite_pop, lb, ub)

    def elite_preservation(self, population, fitness):
        elite_count = int(self.population_size * self.elite_fraction)
        elite_indices = np.argsort(fitness)[:elite_count]
        return population[elite_indices], fitness[elite_indices]
    
    def stochastic_ranking(self, population, fitness):
        sort_idx = np.argsort(fitness)
        return population[sort_idx], fitness[sort_idx]

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += self.population_size

        while self.eval_count < self.budget:
            new_population = []
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                adaptive_f = self.adaptive_scaling_factor()
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

            self.dual_cooling_schedule()
            opposite_population = self.opposition_based_learning(new_population, lb, ub)
            opposite_fitness = np.array([func(ind) for ind in opposite_population])
            self.eval_count += self.population_size

            for j in range(self.population_size):
                if opposite_fitness[j] < fitness[j]:
                    new_population[j] = opposite_population[j]
                    fitness[j] = opposite_fitness[j]

            population, fitness = self.stochastic_ranking(np.array(new_population), fitness)

            elite_pop, elite_fit = self.elite_preservation(population, fitness)
            non_elite_size = self.population_size - len(elite_pop)
            if non_elite_size > 0:
                population = np.vstack((elite_pop, np.random.rand(non_elite_size, self.dim) * (ub - lb) + lb))
                fitness = np.hstack((elite_fit, [func(ind) for ind in population[len(elite_pop):]]))
                self.eval_count += non_elite_size

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]