import numpy as np

class EnhancedHybridSwarmOptimizer:
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
        self.particles = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = np.inf
        self.function_evaluations = 0
        self.chaotic_sequence = self._generate_chaotic_sequence()

    def _generate_chaotic_sequence(self):
        sequence = np.zeros(self.budget)
        x = 0.7
        for i in range(self.budget):
            x = 4 * x * (1 - x)
            sequence[i] = x
        return sequence

    def _initialize(self, bounds):
        self.particles = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-abs(bounds.ub-bounds.lb), abs(bounds.ub-bounds.lb), (self.population_size, self.dim))
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

    def _update_particles(self, bounds):
        for i in range(self.population_size):
            r1, r2 = np.random.rand(2)
            inertia = self.w * self.velocities[i]
            cognitive = self.c1 * r1 * (self.personal_best_positions[i] - self.particles[i])
            social = self.c2 * r2 * (self.global_best_position - self.particles[i])
            self.velocities[i] = inertia + cognitive + social
            self.particles[i] += self.velocities[i]
            self.particles[i] = np.clip(self.particles[i], bounds.lb, bounds.ub)

    def _adaptive_mutation(self, func, bounds):
        for i in range(self.population_size):
            if np.random.rand() < self.CR:
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = self.particles[indices]
                mutant = x1 + self.F * (x2 - x3)
                trial = np.where(self.chaotic_sequence[self.function_evaluations] < self.CR, mutant, self.particles[i])
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
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        for i in range(self.population_size):
            u = np.random.randn(self.dim) * sigma
            v = np.random.randn(self.dim)
            step = u / abs(v) ** (1 / beta)
            step_size = 0.01 * step * (self.particles[i] - self.global_best_position)
            self.particles[i] += step_size
            self.particles[i] = np.clip(self.particles[i], bounds.lb, bounds.ub)

    def _detect_phase_transition(self):
        improvements = np.sum(self.personal_best_scores < self.global_best_score)
        if improvements / self.population_size > 0.15:
            self.w *= 0.85
            self.F *= 1.15
        else:
            self.w *= 1.15
            self.F *= 0.85
        self.population_size = min(self.initial_population_size, max(10, int(self.population_size * 1.05)))

    def __call__(self, func):
        bounds = func.bounds
        self._initialize(bounds)
        while self.function_evaluations < self.budget:
            self._evaluate(func)
            self._update_particles(bounds)
            if np.random.rand() < 0.5:
                self._adaptive_mutation(func, bounds)
            self._levy_flight(bounds)
            self._detect_phase_transition()
        return self.global_best_position, self.global_best_score