import numpy as np

class AdaptiveHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.min_inertia = 0.4
        self.max_inertia = 0.9
        self.min_cognitive = 1.0
        self.max_cognitive = 2.5
        self.min_social = 1.0
        self.max_social = 2.5
        self.mutation_factor = 0.5
        self.crossover_prob = 0.7
        self.iteration = 0
        self.max_iterations = budget // self.population_size

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Initialize particles
        particles = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([func(p) for p in particles])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = particles[global_best_index]

        evaluations = self.population_size

        while evaluations < self.budget:
            inertia_weight = ((self.max_inertia - self.min_inertia) *
                              (self.max_iterations - self.iteration) / self.max_iterations +
                              self.min_inertia)
            cognitive_param = ((self.max_cognitive - self.min_cognitive) *
                               (self.iteration / self.max_iterations) +
                               self.min_cognitive)
            social_param = ((self.max_social - self.min_social) *
                            (self.iteration / self.max_iterations) +
                            self.min_social)

            for i in range(self.population_size):
                # Update velocity and position
                r1, r2 = np.random.rand(2)
                velocities[i] = (inertia_weight * velocities[i] +
                                 cognitive_param * r1 * (personal_best_positions[i] - particles[i]) +
                                 social_param * r2 * (global_best_position - particles[i]))
                particles[i] = particles[i] + velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)

                # Evaluate particles
                score = func(particles[i])
                evaluations += 1

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]

            # Update global best
            global_best_index = np.argmin(personal_best_scores)
            global_best_position = personal_best_positions[global_best_index]

            # Apply DE mutation and crossover to enhance diversity
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = personal_best_positions[np.random.choice(idxs, 3, replace=False)]
                mutant = a + self.mutation_factor * (b - c)
                mutant = np.clip(mutant, lb, ub)

                crossover_mask = np.random.rand(self.dim) < self.crossover_prob
                trial = np.where(crossover_mask, mutant, particles[i])
                trial_score = func(trial)
                evaluations += 1

                # Selection
                if trial_score < personal_best_scores[i]:
                    particles[i] = trial
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial

            self.iteration += 1

        return global_best_position