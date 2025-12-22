import numpy as np

class AdaptiveDiffGuidedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight = 0.9
        self.cognitive_param = 1.4
        self.social_param = 1.6
        self.mutation_factor = 0.6
        self.crossover_prob = 0.9
        self.iteration = 0
        self.performance_window = 5

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
        past_global_best_scores = [personal_best_scores[global_best_index]]

        while evaluations < self.budget:
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
            current_global_best_index = np.argmin(personal_best_scores)
            if personal_best_scores[current_global_best_index] < personal_best_scores[global_best_index]:
                global_best_index = current_global_best_index
                global_best_position = personal_best_positions[global_best_index]

            # Adaptive parameter adjustment based on performance
            past_global_best_scores.append(personal_best_scores[global_best_index])
            if len(past_global_best_scores) > self.performance_window:
                past_global_best_scores.pop(0)
                improvement = past_global_best_scores[-1] - past_global_best_scores[0]
                if improvement < 0:
                    self.inertia_weight *= 0.99
                    self.cognitive_param *= 1.01
                    self.social_param *= 0.99
                else:
                    self.inertia_weight *= 1.01
                    self.cognitive_param *= 0.99
                    self.social_param *= 1.01

            # Apply DE mutation and crossover to enhance diversity
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