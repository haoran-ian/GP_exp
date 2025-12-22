import numpy as np

class DynamicPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 20
        self.w_min, self.w_max = 0.3, 0.9
        self.c1_min, self.c1_max = 0.5, 2.5
        self.c2_min, self.c2_max = 0.5, 2.5
        self.mutation_factor = 0.5
        self.crossover_rate = 0.5
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

    def update_parameters(self, eval_ratio):
        self.w = self.w_max - eval_ratio * (self.w_max - self.w_min)
        self.c1 = self.c1_max - eval_ratio * (self.c1_max - self.c1_min)
        self.c2 = self.c2_min + eval_ratio * (self.c2_max - self.c2_min)

        # Adjust mutation factor based on convergence
        if self.global_best_value != np.inf:
            self.mutation_factor = 0.5 + 0.5 * (1 - eval_ratio) * (self.global_best_value / np.mean(self.personal_best_value))

    def update_particles(self, func):
        for i in range(self.num_particles):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            cognitive = self.c1 * r1 * (self.personal_best[i] - self.population[i])
            social = self.c2 * r2 * (self.global_best - self.population[i])
            self.velocity[i] = self.w * self.velocity[i] + cognitive + social
            self.population[i] += self.velocity[i]
            self.population[i] = np.clip(self.population[i], func.bounds.lb, func.bounds.ub)
            fitness_value = func(self.population[i])
            if fitness_value < self.personal_best_value[i]:
                self.personal_best[i] = np.copy(self.population[i])
                self.personal_best_value[i] = fitness_value
            if fitness_value < self.global_best_value:
                self.global_best = np.copy(self.population[i])
                self.global_best_value = fitness_value

    def differential_evolution(self, func):
        for i in range(self.num_particles):
            indices = [idx for idx in range(self.num_particles) if idx != i]
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant_vector = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
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
            self.update_parameters(eval_ratio)
            self.update_particles(func)
            self.differential_evolution(func)
            eval_count += self.num_particles * 2