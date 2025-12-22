import numpy as np

class RefinedMultiSwarmHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.population_size = self.initial_population_size
        self.w = 0.5  # inertia weight
        self.c1 = 1.0  # cognitive parameter
        self.c2 = 2.0  # social parameter
        self.F = 0.5  # DE scaling factor
        self.CR = 0.9  # DE crossover probability
        self.alpha = 0.01  # Scaling factor for Levy flight
        self.num_swarms = 3

    def adapt_parameters(self, iteration, max_iterations):
        progress = iteration / max_iterations
        angle = progress * 2 * np.pi
        self.w = 0.4 + 0.1 * (np.cos(angle) + 1) / 2
        self.F = 0.5 + 0.2 * np.sin(angle) * np.cos(progress * np.pi)
        self.CR = 0.8 + 0.1 * np.cos(angle)
        self.alpha = 0.01 * (1 + 0.5 * np.sin(angle))

    def levy_flight(self, scale=0.01):
        u = np.random.normal(0, 1, self.dim) * scale
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v) ** (1 / 3))
        return step

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        swarms = [np.random.rand(self.population_size, self.dim) * (ub - lb) + lb for _ in range(self.num_swarms)]
        velocities = [np.random.rand(self.population_size, self.dim) * (ub - lb) / 5 for _ in range(self.num_swarms)]
        personal_best_positions = [np.copy(swarm) for swarm in swarms]
        personal_best_scores = [np.array([func(p) for p in swarm]) for swarm in swarms]
        global_best_position = min((p for sublist in personal_best_positions for p in sublist), key=func)
        global_best_score = func(global_best_position)

        evaluations = self.num_swarms * self.population_size
        max_iterations = self.budget // (self.num_swarms * self.population_size)
        iteration = 0

        while evaluations < self.budget:
            for s in range(self.num_swarms):
                for i in range(self.population_size):
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    velocities[s][i] = (self.w * velocities[s][i] +
                                        self.c1 * r1 * (personal_best_positions[s][i] - swarms[s][i]) +
                                        self.c2 * r2 * (global_best_position - swarms[s][i]))

                    indices = np.random.choice(self.population_size, 3, replace=False)
                    x1, x2, x3 = swarms[s][indices]
                    F_adj = self.F * (1 - iteration / max_iterations)
                    mutant_vector = x1 + F_adj * (x2 - x3)
                    crossover = np.random.rand(self.dim) < self.CR
                    trial_vector = np.where(crossover, mutant_vector, swarms[s][i])

                    swarms[s][i] = np.clip(swarms[s][i] + velocities[s][i], lb, ub)
                    trial_vector = np.clip(trial_vector, lb, ub)

                    if np.random.rand() < 0.3:
                        trial_vector += self.levy_flight(self.alpha * (1 - iteration / max_iterations))

                    trial_score = func(trial_vector)
                    evaluations += 1

                    if trial_score < personal_best_scores[s][i]:
                        personal_best_positions[s][i] = trial_vector
                        personal_best_scores[s][i] = trial_score

                        if trial_score < global_best_score:
                            global_best_position = trial_vector
                            global_best_score = trial_score

                    if evaluations >= self.budget:
                        break

            if iteration % (max_iterations // 4) == 0:
                if evaluations < (3 * self.budget) / 4:
                    self.population_size = max(10, self.initial_population_size // 2)
                else:
                    self.population_size = max(5, self.initial_population_size // 4)

            self.adapt_parameters(iteration, max_iterations)
            iteration += 1

        return global_best_position, global_best_score