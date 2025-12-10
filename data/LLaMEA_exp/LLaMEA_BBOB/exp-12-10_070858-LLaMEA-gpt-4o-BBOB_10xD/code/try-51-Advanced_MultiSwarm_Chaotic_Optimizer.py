import numpy as np

class Advanced_MultiSwarm_Chaotic_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.num_swarms = 3
        self.inertia_weight = 0.9
        self.cognitive_coeff = 1.6
        self.social_coeff = 1.5
        self.temperature = 100.0
        self.cooling_rate = 0.98
        self.mutation_rate = 0.15
        self.current_evals = 0
        self.swarms = self.initialize_swarms()

    def initialize_swarms(self):
        swarms = []
        for _ in range(self.num_swarms):
            particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
            velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
            personal_best_positions = np.copy(particles)
            personal_best_scores = np.full(self.population_size, float('inf'))
            global_best_position = np.zeros(self.dim)
            global_best_score = float('inf')
            swarms.append({
                'particles': particles,
                'velocities': velocities,
                'personal_best_positions': personal_best_positions,
                'personal_best_scores': personal_best_scores,
                'global_best_position': global_best_position,
                'global_best_score': global_best_score
            })
        return swarms

    def chaotic_map(self, x):
        return 4 * x * (1 - x)

    def __call__(self, func):
        while self.current_evals < self.budget:
            x = np.random.rand()
            x = self.chaotic_map(x)
            for swarm in self.swarms:
                dynamic_inertia_weight = 0.4 + 0.5 * (1 - self.current_evals / self.budget)
                adaptive_cooling_rate = self.cooling_rate + 0.02 * np.sin(3 * np.pi * self.current_evals / self.budget)
                for i in range(self.population_size):
                    score = func(swarm['particles'][i])
                    self.current_evals += 1
                    if score < swarm['personal_best_scores'][i]:
                        swarm['personal_best_scores'][i] = score
                        swarm['personal_best_positions'][i] = swarm['particles'][i].copy()
                    if score < swarm['global_best_score']:
                        swarm['global_best_score'] = score
                        swarm['global_best_position'] = swarm['particles'][i].copy()

                    r1, r2, r3 = np.random.rand(), np.random.rand(), np.random.rand()
                    cognitive_velocity = self.cognitive_coeff * r1 * (swarm['personal_best_positions'][i] - swarm['particles'][i])
                    social_velocity = self.social_coeff * r2 * (swarm['global_best_position'] - swarm['particles'][i])
                    diversity_control = 0.5 * r3 * (swarm['particles'][np.random.randint(0, self.population_size)] - swarm['particles'][i])
                    swarm['velocities'][i] = (dynamic_inertia_weight * swarm['velocities'][i] +
                                              cognitive_velocity + social_velocity + diversity_control)
                    swarm['particles'][i] += swarm['velocities'][i]
                    swarm['particles'][i] = np.clip(swarm['particles'][i], self.lower_bound, self.upper_bound)

                    dynamic_mutation_rate = self.mutation_rate * (1 - self.current_evals / self.budget)
                    if np.random.rand() < dynamic_mutation_rate:
                        mutation_vector = np.random.normal(0, 0.1, self.dim)
                        swarm['particles'][i] += mutation_vector
                        swarm['particles'][i] = np.clip(swarm['particles'][i], self.lower_bound, self.upper_bound)

                if self.current_evals % (self.population_size * 2) == 0:
                    sorted_indices = np.argsort(swarm['personal_best_scores'])
                    top_solutions = swarm['personal_best_positions'][sorted_indices[:3]]
                    for other_swarm in self.swarms:
                        if other_swarm != swarm:
                            other_swarm['particles'][:3] = top_solutions

                self.temperature *= adaptive_cooling_rate * (1 + 0.1 * np.cos(np.pi * self.current_evals / self.budget))
                self.cognitive_coeff = 1.5 + 0.1 * np.sin(np.pi * self.current_evals / self.budget)
                self.social_coeff = 1.4 + 0.1 * np.sin(np.pi * self.current_evals / self.budget)

        global_best_position = None
        global_best_score = float('inf')
        for swarm in self.swarms:
            if swarm['global_best_score'] < global_best_score:
                global_best_score = swarm['global_best_score']
                global_best_position = swarm['global_best_position']
        return global_best_position, global_best_score