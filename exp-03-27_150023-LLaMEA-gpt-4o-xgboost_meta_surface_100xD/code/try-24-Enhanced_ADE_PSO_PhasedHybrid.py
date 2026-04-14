import numpy as np

class Enhanced_ADE_PSO_PhasedHybrid:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.base_population_size = min(100, 10 * dim)
        self.population_size = self.base_population_size
        self.population = None
        self.fitness = None
        self.CR = 0.9  # Crossover probability
        self.F = 0.8   # Differential weight
        self.evaluations = 0
        self.memory = []
        self.memory_decay_rate = 0.95
        self.inertia_weight = 0.9
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.velocity = None

    def init_population(self, bounds):
        lower_bound = bounds.lb
        upper_bound = bounds.ub
        self.population = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.velocity = np.zeros((self.population_size, self.dim))

    def compute_diversity(self):
        mean_position = np.mean(self.population, axis=0)
        diversity = np.mean(np.linalg.norm(self.population - mean_position, axis=1))
        return diversity

    def adapt_parameters(self):
        diversity = self.compute_diversity()
        self.CR = 0.5 + 0.4 * (diversity / np.sqrt(self.dim))
        self.F = 0.5 + 0.3 * (diversity / np.sqrt(self.dim))
        self.inertia_weight = 0.7 + 0.2 * (diversity / np.sqrt(self.dim))

    def adjust_population_size_phased(self):
        if self.evaluations < self.budget // 3:
            self.population_size = self.base_population_size
        elif self.evaluations < 2 * self.budget // 3:
            self.population_size = int(self.base_population_size * 0.7)
        else:
            self.population_size = int(self.base_population_size * 0.5)
        self.population = self.population[:self.population_size]
        self.fitness = self.fitness[:self.population_size]
        self.velocity = self.velocity[:self.population_size]

    def differential_evolution(self, target_idx, bounds):
        idxs = [idx for idx in range(len(self.population)) if idx != target_idx]
        a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
        mutant = np.clip(a + self.F * (b - c), bounds.lb, bounds.ub)
        cross_points = np.random.rand(self.dim) < self.CR
        trial = np.where(cross_points, mutant, self.population[target_idx])
        return trial

    def particle_swarm_step(self, i, personal_best, global_best, bounds):
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        self.velocity[i] = (self.inertia_weight * self.velocity[i] +
                            self.cognitive_coeff * r1 * (personal_best[i] - self.population[i]) +
                            self.social_coeff * r2 * (global_best - self.population[i]))
        new_position = np.clip(self.population[i] + self.velocity[i], bounds.lb, bounds.ub)
        return new_position

    def decay_memory(self):
        self.memory = [(solution, fitness * self.memory_decay_rate) for solution, fitness in self.memory]

    def __call__(self, func):
        self.init_population(func.bounds)
        personal_best = self.population.copy()
        personal_best_fitness = np.full(self.population_size, np.inf)
        global_best = None
        global_best_fitness = np.inf

        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            personal_best[i] = self.population[i]
            personal_best_fitness[i] = self.fitness[i]
            if self.fitness[i] < global_best_fitness:
                global_best = self.population[i]
                global_best_fitness = self.fitness[i]
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return global_best

        while self.evaluations < self.budget:
            self.adapt_parameters()
            self.adjust_population_size_phased()
            self.decay_memory()

            for i in range(self.population_size):
                if np.random.rand() < 0.5:
                    trial = self.differential_evolution(i, func.bounds)
                else:
                    trial = self.particle_swarm_step(i, personal_best, global_best, func.bounds)

                trial_fitness = func(trial)
                self.evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    personal_best[i] = trial
                    personal_best_fitness[i] = trial_fitness
                    self.memory.append((trial, trial_fitness))

                    if trial_fitness < global_best_fitness:
                        global_best = trial
                        global_best_fitness = trial_fitness

                if self.evaluations >= self.budget:
                    break

        return global_best