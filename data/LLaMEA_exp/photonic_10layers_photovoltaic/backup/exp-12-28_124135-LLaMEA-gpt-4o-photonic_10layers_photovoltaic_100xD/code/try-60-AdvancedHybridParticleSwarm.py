import numpy as np

class AdvancedHybridParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = min(50, budget)
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.4
        self.cognitive_factor = 1.5
        self.social_factor = 1.5
        self.vel_range_factor = 0.1
        self.de_mutation_factor = 0.8
        self.de_crossover_prob = 0.7

    def __call__(self, func):
        lb = np.array(func[0].bounds.lb)
        ub = np.array(func[0].bounds.ub)
        search_range = ub - lb
        velocity_range = self.vel_range_factor * search_range

        positions = lb + np.random.rand(self.swarm_size, self.dim) * search_range
        velocities = np.random.uniform(-velocity_range, velocity_range, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.swarm_size, np.inf)

        global_best_score = np.inf
        global_best_position = None

        evaluations = 0
        perturbation_decay = 0.9
        perturbation_factor = 0.05

        while evaluations < self.budget:
            scores = np.array([func[0](pos) for pos in positions])
            evaluations += self.swarm_size

            better_mask = scores < personal_best_scores
            personal_best_scores = np.where(better_mask, scores, personal_best_scores)
            personal_best_positions = np.where(better_mask[:, np.newaxis], positions, personal_best_positions)

            min_score = np.min(scores)
            if min_score < global_best_score:
                global_best_score = min_score
                global_best_position = positions[np.argmin(scores)]

            inertia_weight = self.inertia_weight_final + \
                             (self.inertia_weight_initial - self.inertia_weight_final) * \
                             ((self.budget - evaluations) / self.budget)

            r1 = np.random.rand(self.swarm_size, self.dim)
            r2 = np.random.rand(self.swarm_size, self.dim)
            velocities = (
                inertia_weight * velocities +
                self.cognitive_factor * r1 * (personal_best_positions - positions) +
                self.social_factor * r2 * (global_best_position - positions)
            )

            velocities = np.clip(velocities, -velocity_range, velocity_range)

            if evaluations % (self.budget // 5) == 0:
                velocities += perturbation_factor * np.random.uniform(-velocity_range, velocity_range, (self.swarm_size, self.dim))
                perturbation_factor *= perturbation_decay

            positions = positions + velocities
            positions = np.clip(positions, lb, ub)
            
            # Hybrid DE strategy
            if evaluations % (self.budget // 10) == 0:
                for i in range(self.swarm_size):
                    idxs = [idx for idx in range(self.swarm_size) if idx != i]
                    a, b, c = positions[np.random.choice(idxs, 3, replace=False)]
                    mutant_vector = np.clip(a + self.de_mutation_factor * (b - c), lb, ub)
                    crossover = np.random.rand(self.dim) < self.de_crossover_prob
                    trial_vector = np.where(crossover, mutant_vector, positions[i])
                    trial_score = func[0](trial_vector)
                    evaluations += 1
                    if trial_score < personal_best_scores[i]:
                        personal_best_scores[i] = trial_score
                        personal_best_positions[i] = trial_vector
                        if trial_score < global_best_score:
                            global_best_score = trial_score
                            global_best_position = trial_vector

        return global_best_position, global_best_score