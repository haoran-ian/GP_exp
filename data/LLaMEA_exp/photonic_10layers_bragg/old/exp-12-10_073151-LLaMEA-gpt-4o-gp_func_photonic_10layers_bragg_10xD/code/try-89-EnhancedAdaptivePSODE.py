import numpy as np

class EnhancedAdaptivePSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.min_population_size = 5
        self.c1_min = 1.2
        self.c1_max = 2.0
        self.c2_min = 1.2
        self.c2_max = 2.0
        self.w_max = 0.9
        self.w_min = 0.4
        self.F = 0.8
        self.CR = 0.9
        self.population = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')

    def chaotic_initialization(self, lb, ub):
        pop = np.zeros((self.initial_population_size, self.dim))
        x = 0.7
        for i in range(self.initial_population_size):
            for j in range(self.dim):
                x = 4 * x * (1 - x)
                pop[i, j] = lb[j] + (ub[j] - lb[j]) * x
        return pop

    def levy_flight(self, size):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size=size)
        v = np.random.normal(0, 1, size=size)
        step = u / np.abs(v) ** (1 / beta)
        return step

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.current_population_size = self.initial_population_size
        self.population = self.chaotic_initialization(lb, ub)
        self.velocities = np.random.uniform(-1, 1, (self.current_population_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.current_population_size, float('inf'))

    def update_personal_best(self, func):
        scores = np.apply_along_axis(func, 1, self.population)
        for i, score in enumerate(scores):
            if score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = score
                self.personal_best_positions[i] = self.population[i]
        return scores

    def update_global_best(self, scores):
        min_index = np.argmin(scores)
        if scores[min_index] < self.global_best_score:
            self.global_best_score = scores[min_index]
            self.global_best_position = self.population[min_index]

    def calculate_inertia_weight(self, evaluations):
        chaotic_factor = 4 * (evaluations / self.budget) * (1 - (evaluations / self.budget))
        return self.w_max - (self.w_max - self.w_min) * chaotic_factor

    def adaptive_learning_factors(self, evaluations):
        progress = evaluations / self.budget
        self.c1 = self.c1_min + (self.c1_max - self.c1_min) * (1 - progress)
        self.c2 = self.c2_min + (self.c2_max - self.c2_min) * progress

    def pareto_dominance_adjustment(self, evaluations):
        dominance_factor = (1 - evaluations / self.budget) ** 2
        self.F = self.F * dominance_factor
        self.CR = self.CR * (1 - dominance_factor)

    def pso_step(self, func, evaluations):
        scores = self.update_personal_best(func)
        self.update_global_best(scores)
        
        w = self.calculate_inertia_weight(evaluations)
        self.adaptive_learning_factors(evaluations)
        for i in range(self.current_population_size):
            r1, r2 = np.random.rand(2)
            cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.population[i])
            social_component = self.c2 * r2 * (self.global_best_position - self.population[i])
            velocity_scaling_factor = 1 + 1.5 * (evaluations / self.budget)
            decaying_factor = 0.99 ** (evaluations / self.budget)
            self.velocities[i] = (w * self.velocities[i] + cognitive_component + social_component) * velocity_scaling_factor * decaying_factor
            self.population[i] += self.velocities[i] + self.levy_flight(self.dim)

    def chaos_enhanced_mutation(self, lb, ub):
        x = np.random.rand()
        return lb + (ub - lb) * (x * (1 - x) * 4)

    def de_step(self, bounds, func):
        lb, ub = bounds.lb, bounds.ub
        new_population = np.copy(self.population)
        for i in range(self.current_population_size):
            idxs = [idx for idx in range(self.current_population_size) if idx != i]
            a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + self.F * (b - c), lb, ub)
            cross_points = np.random.rand(self.dim) < self.CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, self.population[i])
            trial = np.clip(trial + self.chaos_enhanced_mutation(lb, ub), lb, ub)
            if func(trial) < func(self.population[i]):
                new_population[i] = trial
        self.population = new_population

    def adapt_population_size(self, evaluations):
        if evaluations > self.budget // 2:
            self.current_population_size = max(
                self.min_population_size,
                int(self.initial_population_size * (1 - (evaluations / self.budget)))
            )
            self.population = self.population[:self.current_population_size]
            self.velocities = self.velocities[:self.current_population_size]
            self.personal_best_positions = self.personal_best_positions[:self.current_population_size]
            self.personal_best_scores = self.personal_best_scores[:self.current_population_size]

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)
        evaluations = 0

        while evaluations < self.budget:
            self.pareto_dominance_adjustment(evaluations)
            if evaluations % 2 == 0:
                self.pso_step(func, evaluations)
            else:
                self.de_step(bounds, func)
            
            evaluations += self.current_population_size
            self.adapt_population_size(evaluations)

        return self.global_best_position, self.global_best_score