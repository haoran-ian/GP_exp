import numpy as np

class RefinedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight = 0.9
        self.inertia_damp = 0.99
        self.cognitive_param = 1.7
        self.social_param = 1.7
        self.mutation_factor = 0.7
        self.crossover_prob = 0.9
        self.iteration = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub

        particles = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([func(p) for p in particles])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = particles[global_best_index]

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_param * r1 * (personal_best_positions[i] - particles[i]) +
                                 self.social_param * r2 * (global_best_position - particles[i]))
                particles[i] = particles[i] + velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)

                score = func(particles[i])
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]

            self.inertia_weight *= self.inertia_damp

            global_best_index = np.argmin(personal_best_scores)
            global_best_position = personal_best_positions[global_best_index]

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

                if trial_score < personal_best_scores[i]:
                    particles[i] = trial
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial

            self.iteration += 1
        
        return global_best_position