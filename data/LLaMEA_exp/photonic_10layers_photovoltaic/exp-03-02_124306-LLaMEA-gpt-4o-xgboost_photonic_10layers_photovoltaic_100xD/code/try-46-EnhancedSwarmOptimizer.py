import numpy as np
from scipy.stats import levy_stable

class EnhancedSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.initial_population_size = self.population_size
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.5
        self.F = 0.8
        self.CR = 0.9
        self.memory_size = 10
        self.memory = []
        self.particles = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = np.inf
        self.function_evaluations = 0

    def _initialize(self, bounds):
        n = self.population_size * self.dim
        x = np.linspace(0, 1, n)
        logistic_map = 4 * x * (1 - x)
        chaotic_sequence = logistic_map.reshape(self.population_size, self.dim)
        self.particles = bounds.lb + (bounds.ub - bounds.lb) * chaotic_sequence
        self.velocities = np.random.uniform(-abs(bounds.ub - bounds.lb), abs(bounds.ub - bounds.lb), (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)

    def _evaluate(self, func):
        scores = np.array([func(p) for p in self.particles])
        self.function_evaluations += self.population_size
        for i in range(self.population_size):
            if scores[i] < self.personal_best_scores[i]:
                self.personal_best_scores[i] = scores[i]
                self.personal_best_positions[i] = self.particles[i]
            if scores[i] < self.global_best_score:
                self.global_best_score = scores[i]
                self.global_best_position = self.particles[i]
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        self.memory.append((self.global_best_position, self.global_best_score))

    def _update_particles(self, bounds):
        for i in range(self.population_size):
            r1, r2 = np.random.rand(2)
            self.w = 0.6 + 0.2 * np.random.rand()
            inertia = self.w * self.velocities[i]
            cognitive = self.c1 * r1 * (self.personal_best_positions[i] - self.particles[i])
            social = self.c2 * r2 * (self.global_best_position - self.particles[i])
            self.velocities[i] = inertia + cognitive + social
            self.particles[i] += self.velocities[i]
            self.particles[i] = np.clip(self.particles[i], bounds.lb, bounds.ub)

    def _adaptive_mutation(self, func, bounds):
        dynamic_CR = self.CR * (1 + 0.05 * np.sin(self.function_evaluations / 30))
        for i in range(self.population_size):
            if np.random.rand() < dynamic_CR:
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = self.particles[indices]
                mutant = x1 + self.F * (x2 - x3)
                trial = np.where(np.random.rand(self.dim) < dynamic_CR, mutant, self.particles[i])
                trial = np.clip(trial, bounds.lb, bounds.ub)
                trial_score = func(trial)
                self.function_evaluations += 1
                if trial_score < self.personal_best_scores[i]:
                    self.particles[i] = trial
                    self.personal_best_scores[i] = trial_score
                    if trial_score < self.global_best_score:
                        self.global_best_score = trial_score
                        self.global_best_position = trial

    def _levy_flight(self, bounds):
        alpha = 1.5  # Adjusted for more aggressive exploration
        for i in range(self.population_size):
            step = levy_stable.rvs(alpha, 0, size=self.dim)
            step_size = 0.02 * step * (self.particles[i] - self.global_best_position)  # Adjusted step size
            noise = 0.05 * np.random.randn(self.dim)
            self.particles[i] += step_size + noise
            self.particles[i] = np.clip(self.particles[i], bounds.lb, bounds.ub)

    def _maintain_diversity(self, bounds):
        diversity = np.std(self.particles, axis=0).mean()
        if diversity < 0.15:
            self.particles = bounds.lb + np.random.rand(self.population_size, self.dim) * (bounds.ub - bounds.lb)

    def _randomized_phase_transition(self):
        if np.random.rand() < 0.5:
            improvements = np.sum(self.personal_best_scores < self.global_best_score)
            if improvements / self.population_size > 0.2:
                self.w *= 0.8
                self.F *= 1.2
            else:
                self.w *= 1.2
                self.F *= 0.8
        else:
            if self.memory:
                historical_best = min(self.memory, key=lambda x: x[1])
                if historical_best[1] < self.global_best_score:
                    self.global_best_position, self.global_best_score = historical_best
        self.population_size = min(self.initial_population_size, max(10, int(self.population_size * 1.1)))
        if self.function_evaluations % 100 == 0:
            self.F = 0.7

    def __call__(self, func):
        bounds = func.bounds
        self._initialize(bounds)
        while self.function_evaluations < self.budget:
            self._evaluate(func)
            self._update_particles(bounds)
            if np.random.rand() < 0.6:
                self._adaptive_mutation(func, bounds)
            self._levy_flight(bounds)
            self._maintain_diversity(bounds)
            self._randomized_phase_transition()
        return self.global_best_position, self.global_best_score