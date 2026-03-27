import numpy as np

class MultiSwarmDynamicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.particle_count = 30
        self.sub_swarms = 3
        self.particles_per_swarm = self.particle_count // self.sub_swarms
        self.positions = [None] * self.sub_swarms
        self.velocities = [None] * self.sub_swarms
        self.personal_best_positions = [None] * self.sub_swarms
        self.personal_best_scores = [None] * self.sub_swarms
        self.global_best_positions = [None] * self.sub_swarms
        self.global_best_scores = [float('inf')] * self.sub_swarms
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.4
        self.c1_initial = 2.0
        self.c2_initial = 2.0
        self.c1_final = 1.5
        self.c2_final = 1.5
        self.eval_count = 0

    def initialize_particles(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        for s in range(self.sub_swarms):
            self.positions[s] = np.random.uniform(lb, ub, (self.particles_per_swarm, self.dim))
            self.velocities[s] = np.random.uniform(-1, 1, (self.particles_per_swarm, self.dim))
            self.personal_best_positions[s] = np.copy(self.positions[s])
            self.personal_best_scores[s] = np.full(self.particles_per_swarm, float('inf'))

    def update_velocity_and_position(self, swarm_id):
        r1, r2 = np.random.rand(self.particles_per_swarm, self.dim), np.random.rand(self.particles_per_swarm, self.dim)
        t = self.eval_count / self.budget
        inertia_weight = (1 - t) * self.inertia_weight_initial + t * self.inertia_weight_final
        c1 = (1 - t) * self.c1_initial + t * self.c1_final
        c2 = (1 - t) * self.c2_initial + t * self.c2_final
        
        velocities = (
            inertia_weight * self.velocities[swarm_id] +
            c1 * r1 * (self.personal_best_positions[swarm_id] - self.positions[swarm_id]) +
            c2 * r2 * (self.global_best_positions[swarm_id] - self.positions[swarm_id])
        )
        
        self.velocities[swarm_id] = velocities
        self.positions[swarm_id] += velocities

    def evaluate_particles(self, func, swarm_id):
        diversity_threshold = 0.1
        perturbation_scale = 1 - (self.eval_count / self.budget)
        for i in range(self.particles_per_swarm):
            if self.eval_count < self.budget:
                score = func(self.positions[swarm_id][i])
                self.eval_count += 1
                if score < self.personal_best_scores[swarm_id][i]:
                    self.personal_best_scores[swarm_id][i] = score
                    self.personal_best_positions[swarm_id][i] = self.positions[swarm_id][i]
                if score < self.global_best_scores[swarm_id]:
                    self.global_best_scores[swarm_id] = score
                    self.global_best_positions[swarm_id] = self.positions[swarm_id][i]

        if np.std(self.positions[swarm_id]) < diversity_threshold:
            self.positions[swarm_id] += np.random.uniform(-0.1, 0.1, self.positions[swarm_id].shape)

    def adaptive_neighborhood_search(self, func, swarm_id):
        for i in range(self.particles_per_swarm):
            neighborhood_radius = 0.05 * (func.bounds.ub - func.bounds.lb)
            local_best_score = float('inf')
            local_best_position = self.positions[swarm_id][i]
            for _ in range(5):
                neighbor_position = np.clip(self.positions[swarm_id][i] + np.random.uniform(-neighborhood_radius, neighborhood_radius, self.dim), func.bounds.lb, func.bounds.ub)
                score = func(neighbor_position)
                self.eval_count += 1
                if score < local_best_score:
                    local_best_score = score
                    local_best_position = neighbor_position
            if local_best_score < self.personal_best_scores[swarm_id][i]:
                self.personal_best_scores[swarm_id][i] = local_best_score
                self.personal_best_positions[swarm_id][i] = local_best_position
            if local_best_score < self.global_best_scores[swarm_id]:
                self.global_best_scores[swarm_id] = local_best_score
                self.global_best_positions[swarm_id] = local_best_position

    def dynamic_sub_swarm_interaction(self):
        # Exchange information among sub-swarms
        interaction_probability = 0.1
        for s1 in range(self.sub_swarms):
            for s2 in range(s1 + 1, self.sub_swarms):
                if np.random.rand() < interaction_probability:
                    if self.global_best_scores[s1] < self.global_best_scores[s2]:
                        self.global_best_positions[s2] = self.global_best_positions[s1]
                        self.global_best_scores[s2] = self.global_best_scores[s1]
                    else:
                        self.global_best_positions[s1] = self.global_best_positions[s2]
                        self.global_best_scores[s1] = self.global_best_scores[s2]

    def __call__(self, func):
        self.initialize_particles(func.bounds)
        while self.eval_count < self.budget:
            for s in range(self.sub_swarms):
                self.evaluate_particles(func, s)
                self.update_velocity_and_position(s)
                self.adaptive_neighborhood_search(func, s)
            self.dynamic_sub_swarm_interaction()
        best_swarm = np.argmin(self.global_best_scores)
        return self.global_best_positions[best_swarm], self.global_best_scores[best_swarm]