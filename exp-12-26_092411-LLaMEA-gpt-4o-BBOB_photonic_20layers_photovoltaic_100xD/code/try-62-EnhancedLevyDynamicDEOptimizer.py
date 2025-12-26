import numpy as np

class EnhancedLevyDynamicDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.initial_F = 0.8
        self.initial_CR = 0.9
        self.elite_rate = 0.1
        self.num_evaluations = 0

    def adapt_parameters(self):
        self.F = self.initial_F * (0.5 + 0.5 * np.random.rand())
        self.CR = self.initial_CR * (0.8 + 0.2 * np.random.rand())

    def logistic_map(self, x, r=4.0):
        return r * x * (1 - x)

    def levy_flight(self, L):
        beta = 1.5
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / abs(v) ** (1 / beta)
        return L * step

    def calculate_diversity(self, population):
        return np.mean(np.linalg.norm(population - population.mean(axis=0), axis=1))

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        fitness = np.array([func(ind) for ind in population])
        self.num_evaluations += self.population_size
        chaos_sequence = np.random.rand(self.population_size)

        while self.num_evaluations < self.budget:
            self.adapt_parameters()
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            elite_size = max(1, int(self.elite_rate * self.population_size))
            
            for i in range(elite_size, self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                trial_fitness = func(trial)
                self.num_evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i], fitness[i] = trial, trial_fitness
                
                if self.num_evaluations < self.budget:
                    diversity = self.calculate_diversity(population)
                    levy_step = self.levy_flight(0.01 * (ub - lb) * (1 + diversity))
                    local_trial = trial + levy_step
                    local_trial = np.clip(local_trial, lb, ub)
                    local_fitness = func(local_trial)
                    self.num_evaluations += 1
                    if local_fitness < fitness[i]:
                        population[i], fitness[i] = local_trial, local_fitness

            self.population_size = max(4, int(self.population_size * 0.995))
            population = population[:self.population_size]
            fitness = fitness[:self.population_size]

            chaos_sequence = self.logistic_map(chaos_sequence)
            self.F = self.initial_F * (0.5 + 0.5 * chaos_sequence[np.argmin(fitness)])
            self.CR = self.initial_CR * (0.8 + 0.2 * chaos_sequence[np.argmin(fitness)])

        best_idx = np.argmin(fitness)
        best_solution, best_fitness = population[best_idx], fitness[best_idx]

        return best_solution, best_fitness