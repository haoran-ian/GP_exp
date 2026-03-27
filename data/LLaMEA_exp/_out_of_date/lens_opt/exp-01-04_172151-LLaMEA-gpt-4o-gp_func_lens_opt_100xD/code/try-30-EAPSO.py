import numpy as np

class EAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub

        num_particles = 20
        num_swarms = 3
        particles_per_swarm = num_particles // num_swarms
        w_max = 0.9
        w_min = 0.4
        c1_max = 2.5
        c1_min = 0.5
        c2_max = 2.5
        c2_min = 0.5
        
        swarms_positions = [np.random.uniform(lb, ub, (particles_per_swarm, self.dim)) for _ in range(num_swarms)]
        swarms_velocities = [np.random.uniform(-abs(ub - lb), abs(ub - lb), (particles_per_swarm, self.dim)) for _ in range(num_swarms)]
        personal_best_positions = [np.copy(pos) for pos in swarms_positions]
        personal_best_scores = [np.full(particles_per_swarm, np.inf) for _ in range(num_swarms)]
        global_best_position = None
        global_best_score = np.inf

        num_evaluations = 0

        def update_parameters(diversity, iter_fraction):
            w = w_max - (w_max - w_min) * iter_fraction
            c1 = c1_max - (c1_max - c1_min) * diversity
            c2 = c2_min + (c2_max - c2_min) * diversity
            return w, c1, c2

        while num_evaluations < self.budget:
            iter_fraction = num_evaluations / self.budget

            for swarm_idx in range(num_swarms):
                for i in range(particles_per_swarm):
                    score = func(swarms_positions[swarm_idx][i])
                    num_evaluations += 1

                    if score < personal_best_scores[swarm_idx][i]:
                        personal_best_scores[swarm_idx][i] = score
                        personal_best_positions[swarm_idx][i] = swarms_positions[swarm_idx][i]

                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = swarms_positions[swarm_idx][i]

            for swarm_idx in range(num_swarms):
                diversity = np.mean(np.std(swarms_positions[swarm_idx], axis=0))
                w, c1, c2 = update_parameters(diversity, iter_fraction)

                for i in range(particles_per_swarm):
                    r1 = np.random.rand(self.dim)
                    r2 = np.random.rand(self.dim)
                    cognitive = c1 * r1 * (personal_best_positions[swarm_idx][i] - swarms_positions[swarm_idx][i])
                    social = c2 * r2 * (global_best_position - swarms_positions[swarm_idx][i])
                    swarms_velocities[swarm_idx][i] = w * swarms_velocities[swarm_idx][i] + cognitive + social
                    swarms_positions[swarm_idx][i] += swarms_velocities[swarm_idx][i]

                    swarms_positions[swarm_idx][i] = np.clip(swarms_positions[swarm_idx][i], lb, ub)

                    if np.linalg.norm(swarms_velocities[swarm_idx][i]) < 1e-5:
                        swarms_velocities[swarm_idx][i] = np.random.uniform(-abs(ub - lb), abs(ub - lb), self.dim)

        return global_best_position, global_best_score