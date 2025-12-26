import numpy as np

class HybridAdaptivePSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 20
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1_max = 2.5
        self.c1_min = 0.5
        self.c2_max = 2.5
        self.c2_min = 0.5
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
        for i in range(self.num_particles):
            w = self.w_max - eval_ratio * (self.w_max - self.w_min)
            c1 = self.c1_max - eval_ratio * (self.c1_max - self.c1_min)
            c2 = self.c2_min + eval_ratio * (self.c2_max - self.c2_min)
            
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            cognitive = c1 * r1 * (self.personal_best[i] - self.population[i])
            social = c2 * r2 * (self.global_best - self.population[i])
            neighborhood_effect = self.get_neighborhood_effect(i)
            self.velocity[i] = w * self.velocity[i] + cognitive + social + neighborhood_effect
            self.population[i] += self.velocity[i]
            self.population[i] = np.clip(self.population[i], func.bounds.lb, func.bounds.ub)

            fitness_value = func(self.population[i])
            if fitness_value < self.personal_best_value[i]:
                self.personal_best[i] = np.copy(self.population[i])
                self.personal_best_value[i] = fitness_value
            if fitness_value < self.global_best_value:
                self.global_best = np.copy(self.population[i])
                self.global_best_value = fitness_value

    def get_neighborhood_effect(self, index):
        neighbor_indices = [(index + offset) % self.num_particles for offset in range(-1, 2) if offset != 0]
        neighborhood_center = np.mean(self.population[neighbor_indices], axis=0)
        return 0.1 * (neighborhood_center - self.population[index])

    def differential_evolution(self, func, eval_ratio):
        for i in range(self.num_particles):
            indices = [idx for idx in range(self.num_particles) if idx != i]
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutation_factor = 0.5 + eval_ratio * (self.personal_best_value[i] / self.global_best_value) 
            crossover_rate = 0.9 * (1 - eval_ratio)
            mutant_vector = self.population[a] + mutation_factor * (self.population[b] - self.population[c])
            mutant_vector = np.clip(mutant_vector, func.bounds.lb, func.bounds.ub)
            crossover = np.random.rand(self.dim) < crossover_rate
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