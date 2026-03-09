import numpy as np
from sklearn.cluster import KMeans

class DynamicMultiPopHybridAlgorithmRefined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.particles = np.random.rand(self.population_size, self.dim)
        self.velocities = np.random.rand(self.population_size, self.dim) * 0.1
        self.personal_best_positions = self.particles.copy()
        self.global_best_position = self.particles[0].copy()
        self.personal_best_values = np.full(self.population_size, np.inf)
        self.global_best_value = np.inf
        self.c1, self.c2 = 1.5, 2.5
        self.w = 0.7  # Increased inertia for broader exploration
        self.de_cross_rate = 0.9  # Higher crossover rate for diversity
        self.de_f = 0.5
        self.evaluations = 0
        self.diversity_threshold = 0.1  # Lower threshold for diversity enhancement
        self.phase_length = self.budget // 5  # Longer phases for stability
        self.feedback_step = self.budget // 8
        self.num_subpopulations = 2  # Reduced initial subpopulations for focused search
        self.cluster_adjustment_rate = 0.05  # Rate to adjust clustering strategy

    def update_velocity(self, phase):
        r1 = np.random.rand(self.population_size, self.dim)
        r2 = np.random.rand(self.population_size, self.dim)
        cognitive_component = self.c1 * r1 * (self.personal_best_positions - self.particles)
        social_component = self.c2 * r2 * (self.global_best_position - self.particles)
        self.velocities = self.w * self.velocities + cognitive_component + social_component
        self.velocities = np.clip(self.velocities, -0.1, 0.1)

    def update_position(self, bounds):
        self.particles += self.velocities
        self.particles = np.clip(self.particles, bounds.lb, bounds.ub)

    def measure_diversity(self):
        centroid = np.mean(self.particles, axis=0)
        diversity = np.mean(np.linalg.norm(self.particles - centroid, axis=1))
        return diversity

    def differential_evolution(self, func, bounds):
        for i in range(self.population_size):
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.particles[np.random.choice(indices, 3, replace=False)]
            de_factor = np.random.uniform(0.4, 0.9)
            mutant_vector = np.clip(a + de_factor * (b - c), bounds.lb, bounds.ub)
            crossover = np.random.rand(self.dim) < self.de_cross_rate
            trial_vector = np.where(crossover, mutant_vector, self.particles[i])
            trial_value = func(trial_vector)
            self.evaluations += 1
            if trial_value < self.personal_best_values[i]:
                self.personal_best_positions[i] = trial_vector
                self.personal_best_values[i] = trial_value
                if trial_value < self.global_best_value:
                    self.global_best_position = trial_vector
                    self.global_best_value = trial_value
            if self.evaluations >= self.budget:
                return

    def local_search(self, func, bounds):
        perturbation = np.random.normal(0, 0.01, size=self.dim)  # Slightly larger perturbation
        for i in range(self.population_size):
            candidate = self.particles[i] + perturbation
            candidate = np.clip(candidate, bounds.lb, bounds.ub)
            candidate_value = func(candidate)
            self.evaluations += 1
            if candidate_value < self.personal_best_values[i]:
                self.personal_best_positions[i] = candidate
                self.personal_best_values[i] = candidate_value
                if candidate_value < self.global_best_value:
                    self.global_best_position = candidate
                    self.global_best_value = candidate_value
            if self.evaluations >= self.budget:
                return

    def enhance_diversity(self, func, bounds):
        kmeans = KMeans(n_clusters=self.num_subpopulations)
        kmeans.fit(self.particles)
        for i in range(self.population_size):
            if np.random.rand() < 0.2:  # Increased chance for random repositioning
                self.particles[i] = np.random.uniform(bounds.lb, bounds.ub, self.dim)
            else:
                cluster_center = kmeans.cluster_centers_[kmeans.labels_[i]]
                self.particles[i] = 0.6 * self.particles[i] + 0.4 * cluster_center  # Adjusted recombination
                self.particles[i] = np.clip(self.particles[i], bounds.lb, bounds.ub)

    def update_dynamic_parameters(self):
        self.w = max(0.2, self.w * 0.95)
        self.c1 = max(1.0, self.c1 * 0.98)
        self.c2 = min(2.5, self.c2 * 1.02)

    def feedback_adjustment(self):
        if self.evaluations % self.feedback_step == 0:
            if self.global_best_value < np.median(self.personal_best_values):
                self.phase_length = max(self.phase_length // 2, 1)
                self.diversity_threshold *= 1.1
                self.num_subpopulations = min(5, int(self.num_subpopulations + self.cluster_adjustment_rate * self.evaluations / self.budget))

    def __call__(self, func):
        bounds = func.bounds
        self.evaluations = 0
        while self.evaluations < self.budget:
            self.phase = (self.evaluations // self.phase_length) % 2
            self.update_velocity(self.phase)
            self.update_position(bounds)
            self.differential_evolution(func, bounds)
            self.local_search(func, bounds)
            if self.measure_diversity() < self.diversity_threshold:
                self.enhance_diversity(func, bounds)
            self.update_dynamic_parameters()
            self.feedback_adjustment()
        return self.global_best_position