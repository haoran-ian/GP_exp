import numpy as np

class ImprovedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.crossover_rate = 0.9
        self.scaling_factor = 0.8
        self.temperature = 100.0
        self.cooling_rate = 0.99
        self.inertia_weight = 0.5
        self.cognitive_constant = 1.5
        self.social_constant = 1.5

    def chaotic_initialization(self, bounds):
        x0 = np.random.rand(self.dim)
        chaotic_sequence = np.zeros((self.population_size, self.dim))
        beta = 3.7
        for i in range(self.population_size):
            x0 = beta * x0 * (1 - x0)
            chaotic_sequence[i] = x0
        return chaotic_sequence * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

    def levy_flight(self, L):
        u = np.random.normal(0, 1, self.dim) * L
        v = np.random.normal(0, 1, self.dim)
        step = u / np.power(np.abs(v), 1.0 / 3.0)
        return step

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = self.chaotic_initialization(bounds)
        velocities = np.zeros_like(population)
        personal_best = np.copy(population)
        personal_best_fitness = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_fitness)]
        global_best_fitness = np.min(personal_best_fitness)
        eval_count = self.population_size

        while eval_count < self.budget:
            mutation_prob = max(0.1, min(0.3, 1.0 - (eval_count / self.budget)))  # Adaptive mutation probability
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                # Update velocities using PSO dynamics
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocities[i] = (
                    self.inertia_weight * velocities[i]
                    + self.cognitive_constant * r1 * (personal_best[i] - population[i])
                    + self.social_constant * r2 * (global_best - population[i])
                )

                # Update position using differential evolution strategy
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant_vector = np.clip(a + self.scaling_factor * (b - c), bounds[:, 0], bounds[:, 1])
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                trial_vector = np.where(cross_points, mutant_vector, population[i])

                # Levy flight perturbation
                levy_step = self.levy_flight(0.1 + 0.1 * (eval_count / self.budget))  # Dynamic search space adjustment
                if np.random.rand() < mutation_prob:
                    trial_vector += levy_step

                # Boundary constraints
                trial_vector = np.clip(trial_vector, bounds[:, 0], bounds[:, 1])

                # Evaluate fitness
                fitness = func(trial_vector)
                eval_count += 1

                # Update personal and global bests
                if fitness < personal_best_fitness[i]:
                    personal_best[i] = trial_vector
                    personal_best_fitness[i] = fitness
                    if fitness < global_best_fitness:
                        global_best = trial_vector
                        global_best_fitness = fitness
                        self.cooling_rate *= 0.995  # Adaptive cooling rate

        return global_best, global_best_fitness