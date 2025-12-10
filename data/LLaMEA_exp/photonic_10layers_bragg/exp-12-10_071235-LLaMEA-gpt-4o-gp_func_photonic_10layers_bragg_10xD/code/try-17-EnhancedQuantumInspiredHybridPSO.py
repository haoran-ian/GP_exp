import numpy as np

class EnhancedQuantumInspiredHybridPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.inertia_weight = 0.9
        self.cognitive_coefficient = 2.0
        self.social_coefficient = 2.0
        self.global_learning_rate = 0.1
        self.mutation_coefficient = 0.1
        self.quantum_coefficient = 0.1
        self.lower_bound = None
        self.upper_bound = None
        self.num_swarms = 3

    def __call__(self, func):
        self.lower_bound = np.array(func.bounds.lb)
        self.upper_bound = np.array(func.bounds.ub)
        swarms = [np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim)) for _ in range(self.num_swarms)]
        velocities = [np.random.uniform(-1, 1, (self.population_size, self.dim)) for _ in range(self.num_swarms)]
        personal_best_positions = [np.copy(swarm) for swarm in swarms]
        personal_best_scores = [np.full(self.population_size, np.inf) for _ in range(self.num_swarms)]
        global_best_positions = [None] * self.num_swarms
        global_best_scores = [np.inf] * self.num_swarms
        evaluations = 0

        while evaluations < self.budget:
            for s in range(self.num_swarms):
                for i in range(self.population_size):
                    fitness = func(swarms[s][i])
                    evaluations += 1

                    if fitness < personal_best_scores[s][i]:
                        personal_best_scores[s][i] = fitness
                        personal_best_positions[s][i] = swarms[s][i]

                    if fitness < global_best_scores[s]:
                        global_best_scores[s] = fitness
                        global_best_positions[s] = swarms[s][i]

                for i in range(self.population_size):
                    r1 = np.random.rand(self.dim)
                    r2 = np.random.rand(self.dim)

                    # Dynamic adjustment of parameters
                    self.inertia_weight = max(0.4, self.inertia_weight * 0.98)
                    self.cognitive_coefficient = max(1.5, self.cognitive_coefficient * 0.98)
                    self.social_coefficient = max(1.5, self.social_coefficient * 0.98)

                    velocities[s][i] = (
                        self.inertia_weight * velocities[s][i]
                        + self.cognitive_coefficient * r1 * (personal_best_positions[s][i] - swarms[s][i])
                        + self.social_coefficient * r2 * (global_best_positions[s] - swarms[s][i])
                    )

                    swarms[s][i] += velocities[s][i] * self.global_learning_rate

                    # Adaptive mutation based on distance to global best
                    distance_to_best = np.linalg.norm(swarms[s][i] - global_best_positions[s])
                    mutation_probability = self.mutation_coefficient * (distance_to_best / np.linalg.norm(self.upper_bound - self.lower_bound))
                    
                    if np.random.rand() < mutation_probability:
                        mutation_vector = np.random.normal(0, 1, self.dim)
                        swarms[s][i] += mutation_vector * mutation_probability

                    # Dynamic Quantum tunneling with swarm interaction
                    if np.random.rand() < self.quantum_coefficient:
                        other_swarm = (s + np.random.randint(1, self.num_swarms)) % self.num_swarms
                        quantum_tunnel = np.random.normal(0, 1, self.dim)
                        swarms[s][i] = global_best_positions[other_swarm] + quantum_tunnel * self.quantum_coefficient

                    swarms[s][i] = np.clip(swarms[s][i], self.lower_bound, self.upper_bound)

        best_swarm_idx = np.argmin(global_best_scores)
        return global_best_positions[best_swarm_idx]