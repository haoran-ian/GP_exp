import numpy as np

class HybridAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20 * dim
        self.population = np.random.rand(self.initial_population_size, dim)
        self.F_memory = [0.5] * self.initial_population_size
        self.CR_memory = [0.9] * self.initial_population_size
        self.successful_f = []
        self.successful_cr = []

    def levy_flight(self, L):
        u = np.random.normal(0, 1, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / np.abs(v) ** (1 / L)
        return step

    def adapt_parameters(self):
        if self.successful_f:
            self.F_memory = np.random.choice(self.successful_f, size=self.initial_population_size)
        if self.successful_cr:
            self.CR_memory = np.random.choice(self.successful_cr, size=self.initial_population_size)

    def differential_evolution(self, func, lb, ub):
        bounds = np.array([lb, ub])
        best_solution = None
        best_fitness = np.inf
        evaluations = 0

        self.population = lb + (ub - lb) * self.population
        fitness = np.apply_along_axis(func, 1, self.population)
        
        while evaluations < self.budget:
            self.adapt_parameters()
            for i in range(self.initial_population_size):
                indices = [idx for idx in range(self.initial_population_size) if idx != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]

                F_dynamic = self.F_memory[i]
                CR_dynamic = self.CR_memory[i]
                
                mutant_vector = np.clip(a + F_dynamic * (b - c), lb, ub)
                crossover_mask = np.random.rand(self.dim) < CR_dynamic
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])
                
                if np.random.rand() < 0.5:
                    trial_vector += self.levy_flight(1.5) * (trial_vector - self.population[i])
                
                trial_fitness = func(trial_vector)
                evaluations += 1
                
                if trial_fitness < fitness[i]:
                    self.population[i] = trial_vector
                    fitness[i] = trial_fitness
                    self.successful_f.append(F_dynamic)
                    self.successful_cr.append(CR_dynamic)

                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_solution = trial_vector

                if evaluations >= self.budget:
                    break

        return best_solution, best_fitness

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution, best_fitness = self.differential_evolution(func, lb, ub)
        return best_solution