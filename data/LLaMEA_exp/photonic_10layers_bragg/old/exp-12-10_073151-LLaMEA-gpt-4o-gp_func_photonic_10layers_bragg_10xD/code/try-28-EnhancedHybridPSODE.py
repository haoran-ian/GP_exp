import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.min_population_size = 5
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.9
        self.w_decay = 0.95
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

    def pso_step(self, func):
        scores = self.update_personal_best(func)
        self.update_global_best(scores)
        
        for i in range(self.current_population_size):
            r1, r2 = np.random.rand(2)
            cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.population[i])
            social_component = self.c2 * r2 * (self.global_best_position - self.population[i])
            neighborhood_attraction = np.mean(self.population, axis=0) - self.population[i]
            self.velocities[i] = self.w * self.velocities[i] + cognitive_component + social_component + 0.1 * neighborhood_attraction
            self.population[i] += self.velocities[i]
        
        self.w *= self.w_decay

    def chaos_enhanced_mutation(self, lb, ub):
        x = np.random.rand()
        return lb + (ub - lb) * (x * (1 - x) * 4 * self.w_decay)  # Added decay to chaos mutation

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
            trial = trial + self.chaos_enhanced_mutation(lb, ub)
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
            if evaluations % 2 == 0:
                self.pso_step(func)
            else:
                self.de_step(bounds, func)
            
            evaluations += self.current_population_size
            self.adapt_population_size(evaluations)

        return self.global_best_position, self.global_best_score