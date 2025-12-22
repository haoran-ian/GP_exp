import numpy as np

class HybridAdaptivePSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 20
        self.initial_inertia = 0.9
        self.final_inertia = 0.4
        self.c1 = 1.5
        self.c2 = 1.5
        self.initial_mutation_factor = 0.8
        self.final_mutation_factor = 0.2
        self.crossover_rate = 0.9
        self.population = None
        self.velocity = None
        self.personal_best = None
        self.personal_best_value = None
        self.global_best = None
        self.global_best_value = np.inf

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.population = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        self.velocity = np.random.uniform(-abs(ub - lb), abs(ub - lb), (self.num_particles, self.dim))
        self.personal_best = np.copy(self.population)
        self.personal_best_value = np.full(self.num_particles, np.inf)
        self.global_best = np.copy(self.population[0])
        self.global_best_value = np.inf

    def update_particles(self, func, eval_ratio):
        inertia_weight = self.initial_inertia - eval_ratio * (self.initial_inertia - self.final_inertia)
        for i in range(self.num_particles):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            cognitive = self.c1 * r1 * (self.personal_best[i] - self.population[i])
            social = self.c2 * r2 * (self.global_best - self.population[i])
            self.velocity[i] = inertia_weight * self.velocity[i] + cognitive + social
            self.population[i] += self.velocity[i]
            self.population[i] = np.clip(self.population[i], func.bounds.lb, func.bounds.ub)
            fitness_value = func(self.population[i])
            if fitness_value < self.personal_best_value[i]:
                self.personal_best[i] = np.copy(self.population[i])
                self.personal_best_value[i] = fitness_value
            if fitness_value < self.global_best_value:
                self.global_best = np.copy(self.population[i])
                self.global_best_value = fitness_value

    def differential_evolution(self, func, eval_ratio):
        for i in range(self.num_particles):
            indices = [idx for idx in range(self.num_particles) if idx != i]
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutation_factor = self.initial_mutation_factor - eval_ratio * (self.initial_mutation_factor - self.final_mutation_factor)
            mutant_vector = self.population[a] + mutation_factor * (self.population[b] - self.population[c])
            mutant_vector = np.clip(mutant_vector, func.bounds.lb, func.bounds.ub)
            crossover = np.random.rand(self.dim) < self.crossover_rate
            if not np.any(crossover):
                crossover[np.random.randint(0, self.dim)] = True
            trial_vector = np.where(crossover, mutant_vector, self.population[i])
            trial_fitness = func(trial_vector)
            if trial_fitness < self.personal_best_value[i]:
                self.population[i] = np.copy(trial_vector)
                self.personal_best_value[i] = trial_fitness
                self.personal_best[i] = np.copy(trial_vector)
                if trial_fitness < self.global_best_value:
                    self.global_best = np.copy(trial_vector)
                    self.global_best_value = trial_fitness

    def __call__(self, func):
        self.initialize_population(func.bounds)
        eval_count = 0
        while eval_count < self.budget:
            eval_ratio = eval_count / self.budget
            self.update_particles(func, eval_ratio)
            self.differential_evolution(func, eval_ratio)
            eval_count += self.num_particles * 2