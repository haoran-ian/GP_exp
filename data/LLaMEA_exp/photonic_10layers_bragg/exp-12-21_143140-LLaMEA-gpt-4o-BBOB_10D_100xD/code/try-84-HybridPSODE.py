import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 20
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 1.5
        self.c2 = 1.5
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

    def update_particles(self, func, eval_ratio):
        inertia_weight = self.w_max - (self.w_max - self.w_min) * eval_ratio
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
            self.mutation_factor = 0.5 + eval_ratio * (self.personal_best_value[i] / self.global_best_value)
            self.crossover_rate = 0.9 * (1 - eval_ratio)
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

    def tournament_selection(self):
        tournament_size = 3
        selected_indices = np.random.choice(self.num_particles, tournament_size, replace=False)
        best_index = selected_indices[np.argmin(self.personal_best_value[selected_indices])]
        return best_index

    def __call__(self, func):
        self.initialize_population(func.bounds)
        eval_count = 0
        while eval_count < self.budget:
            eval_ratio = eval_count / self.budget
            self.update_particles(func, eval_ratio)
            self.differential_evolution(func, eval_ratio)
            if eval_count % (self.num_particles * 2) == 0 and eval_count > 0:
                selected_index = self.tournament_selection()
                self.global_best = np.copy(self.personal_best[selected_index])
                self.global_best_value = self.personal_best_value[selected_index]
            eval_count += self.num_particles * 2