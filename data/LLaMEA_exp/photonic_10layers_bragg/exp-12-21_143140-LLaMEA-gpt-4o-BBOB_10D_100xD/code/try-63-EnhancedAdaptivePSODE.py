import numpy as np

class EnhancedAdaptivePSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 20
        self.w_init = 0.9
        self.w_final = 0.4
        self.c1_init = 2.5
        self.c1_final = 0.5
        self.c2_init = 0.5
        self.c2_final = 2.5
        self.mutation_factor_init = 0.8
        self.mutation_factor_final = 0.2
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
        w = self.w_init - (self.w_init - self.w_final) * eval_ratio
        c1 = self.c1_init - (self.c1_init - self.c1_final) * eval_ratio
        c2 = self.c2_init + (self.c2_final - self.c2_init) * eval_ratio
        for i in range(self.num_particles):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            cognitive = c1 * r1 * (self.personal_best[i] - self.population[i])
            social = c2 * r2 * (self.global_best - self.population[i])
            self.velocity[i] = w * self.velocity[i] + cognitive + social
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
        mutation_factor = self.mutation_factor_init - (self.mutation_factor_init - self.mutation_factor_final) * eval_ratio
        for i in range(self.num_particles):
            indices = [idx for idx in range(self.num_particles) if idx != i]
            a, b, c = np.random.choice(indices, 3, replace=False)
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