import numpy as np

class Enhanced_Adaptive_Hybrid_PSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(20, int(budget / 100))
        self.particles = np.random.rand(self.population_size, self.dim)
        self.velocities = np.zeros((self.population_size, self.dim))
        self.personal_best = np.copy(self.particles)
        self.global_best = None
        self.fitness = np.full(self.population_size, np.inf)
        self.best_fitness = np.inf
        self.eval_count = 0
        self.w = 0.9
        self.c1 = 1.5
        self.c2 = 1.5
        self.F = 0.8
        self.CR = 0.7
        self.prev_best_fitness = np.inf
        self.div_threshold = 0.1
    
    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        while self.eval_count < self.budget:
            self._evaluate_fitness(func)
            self._dynamic_update_parameters()

            if self._calculate_diversity() > self.div_threshold:
                self._pso_update(bounds)
            else:
                self._de_update(bounds)
            
            self._opposition_based_learning(func, bounds)

        return self.global_best

    def _evaluate_fitness(self, func):
        for i in range(self.population_size):
            if self.eval_count >= self.budget:
                break
            fitness_value = func(self.particles[i])
            self.eval_count += 1

            if fitness_value < self.fitness[i]:
                self.fitness[i] = fitness_value
                self.personal_best[i] = self.particles[i]

            if fitness_value < self.best_fitness:
                self.best_fitness = fitness_value
                self.global_best = self.particles[i]

    def _dynamic_update_parameters(self):
        diversity = self._calculate_diversity()
        if self.best_fitness < self.prev_best_fitness:
            self.w = max(0.4, self.w * (0.95 + 0.1 * np.random.rand()))
            self.F = min(0.9, self.F * (1.02 + 0.02 * np.random.rand()))
        else:
            self.w = min(0.9, self.w * (1.03 - 0.03 * diversity))
            self.F = max(0.4, self.F * (0.97 + 0.03 * (1 - diversity)))
        self.prev_best_fitness = self.best_fitness

    def _calculate_diversity(self):
        return np.mean(np.std(self.particles, axis=0))

    def _pso_update(self, bounds):
        for i in range(self.population_size):
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            self.velocities[i] = (self.w * self.velocities[i] +
                                  self.c1 * r1 * (self.personal_best[i] - self.particles[i]) +
                                  self.c2 * r2 * (self.global_best - self.particles[i]))
            self.particles[i] += self.velocities[i]
            self.particles[i] = np.clip(self.particles[i], *bounds)

    def _de_update(self, bounds):
        for i in range(self.population_size):
            indices = np.random.choice(self.population_size, 3, replace=False)
            x0, x1, x2 = self.particles[indices]

            mutant_vector = x0 + self.F * (x1 - x2)
            crossover = np.random.rand(self.dim) < self.CR
            trial_vector = np.where(crossover, mutant_vector, self.particles[i])
            trial_vector = np.clip(trial_vector, *bounds)

            if self.eval_count < self.budget:
                trial_fitness = func(trial_vector)
                self.eval_count += 1

                if trial_fitness < self.fitness[i]:
                    self.particles[i] = trial_vector
                    self.fitness[i] = trial_fitness

                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.global_best = trial_vector
    
    def _opposition_based_learning(self, func, bounds):
        opp_particles = bounds[1] - self.particles + bounds[0]
        for i in range(self.population_size):
            if self.eval_count >= self.budget:
                break
            fitness_value = func(opp_particles[i])
            self.eval_count += 1

            if fitness_value < self.fitness[i]:
                self.particles[i] = opp_particles[i]
                self.fitness[i] = fitness_value

                if fitness_value < self.best_fitness:
                    self.best_fitness = fitness_value
                    self.global_best = opp_particles[i]