import numpy as np

class EnhancedAdaptiveDynamicMultiSwarmEvoPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.swarm_count = 3
        self.population_size = [self.initial_population_size // self.swarm_count] * self.swarm_count
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        self.mutation_rate = 0.1
        self.elite_count = 5
        self.diversity_threshold = 0.1
        self.neighborhood_size = 5

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarms = [np.random.uniform(lb, ub, (psize, self.dim))
                  for psize in self.population_size]
        velocities = [np.random.uniform(-1, 1, (psize, self.dim))
                      for psize in self.population_size]
        personal_best_positions = [np.copy(swarm) for swarm in swarms]
        personal_best_scores = [np.array([func(ind) for ind in swarm])
                                for swarm in swarms]
        global_best_position = min(
            (p[np.argmin(s)], ps)
            for p, s in zip(personal_best_positions, personal_best_scores)
            for ps in np.min(s)
        )[0]
        global_best_score = np.min(
            [np.min(scores) for scores in personal_best_scores]
        )

        eval_count = sum(self.population_size)

        while eval_count < self.budget:
            for swarm_idx, (swarm, vel, pbest_pos, pbest_scores) in enumerate(
                zip(swarms, velocities, personal_best_positions, personal_best_scores)
            ):
                for i, position in enumerate(swarm):
                    phase = eval_count / self.budget
                    exploration_factor = 1 - phase ** 2
                    dynamic_c1 = self.c1 * exploration_factor
                    dynamic_c2 = self.c2 * (1 - exploration_factor)
                    dynamic_w = self.w * (0.5 + 0.5 * np.random.rand()) * (1 - 0.5 * phase)

                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    vel[i] = (
                        dynamic_w * vel[i]
                        + dynamic_c1 * r1 * (pbest_pos[i] - position)
                        + dynamic_c2 * r2 * (global_best_position - position)
                    )

                    proposed_position = position + vel[i]
                    proposed_position = np.clip(proposed_position, lb, ub)

                    diversity = np.std(swarm, axis=0).mean()
                    adaptive_mutation_rate = self.mutation_rate * (1 + diversity)
                    if np.random.rand() < adaptive_mutation_rate:
                        elite_indices = np.argsort(pbest_scores)[:self.elite_count]
                        elite_pos = pbest_pos[elite_indices]
                        elite_vector = np.mean(elite_pos, axis=0)
                        mutation_vector = np.random.uniform(lb, ub, self.dim)
                        if np.linalg.norm(mutation_vector - position) > self.diversity_threshold * (ub - lb).mean():
                            proposed_position = self._adaptive_combine_diff(proposed_position, mutation_vector, elite_vector)

                    neighbors = np.random.uniform(lb, ub, (self.neighborhood_size, self.dim))
                    neighbor_scores = np.array([func(neighbor) for neighbor in neighbors])
                    best_neighbor_position = neighbors[np.argmin(neighbor_scores)]
                    best_neighbor_score = np.min(neighbor_scores)
                    
                    if best_neighbor_score < func(proposed_position):
                        proposed_position = best_neighbor_position
                        score = best_neighbor_score
                    else:
                        score = func(proposed_position)

                    eval_count += 1

                    if score < pbest_scores[i]:
                        pbest_scores[i] = score
                        pbest_pos[i] = proposed_position

                    if score < global_best_score:
                        global_best_position = proposed_position
                        global_best_score = score

                    if eval_count >= self.budget:
                        break

                self.population_size[swarm_idx] = max(10, int(self.initial_population_size * (1 - phase) // self.swarm_count))
                if self.population_size[swarm_idx] < len(swarm):
                    swarms[swarm_idx] = swarm[:self.population_size[swarm_idx]]
                    velocities[swarm_idx] = vel[:self.population_size[swarm_idx]]
                    personal_best_positions[swarm_idx] = pbest_pos[:self.population_size[swarm_idx]]
                    personal_best_scores[swarm_idx] = pbest_scores[:self.population_size[swarm_idx]]

        return global_best_position

    def _adaptive_combine_diff(self, pos, mut, elite):
        F = 0.8
        return pos + F * (elite - pos) + (mut - pos) * np.random.rand() ** 2