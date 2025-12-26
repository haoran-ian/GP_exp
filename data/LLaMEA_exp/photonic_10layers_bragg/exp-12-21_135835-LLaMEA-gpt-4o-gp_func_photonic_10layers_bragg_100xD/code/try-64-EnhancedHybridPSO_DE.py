import numpy as np

class EnhancedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight = 0.9
        self.cognitive_param = 2.0
        self.social_param = 2.0
        self.mutation_factor = 0.5
        self.crossover_prob = 0.8
        self.iteration = 0
        self.prev_best_score = float('inf')

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
        global_best_score = personal_best_scores[global_best_index]

        evaluations = self.population_size

        while evaluations < self.budget:
            # Fuzzy logic for adaptive inertia weight
            fitness_improvement = (self.prev_best_score - global_best_score) / self.prev_best_score if self.prev_best_score != 0 else 1
            self.inertia_weight = 0.9 - 0.5 * (1 - fitness_improvement)
            self.prev_best_score = global_best_score

            for i in range(self.population_size):
                # Update velocity and position
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_param * r1 * (personal_best_positions[i] - particles[i]) +
                                 self.social_param * r2 * (global_best_position - particles[i]))
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
            global_best_score = personal_best_scores[global_best_index]

            # Apply DE mutation and crossover to adapt diversity
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = particles[np.random.choice(idxs, 3, replace=False)]
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