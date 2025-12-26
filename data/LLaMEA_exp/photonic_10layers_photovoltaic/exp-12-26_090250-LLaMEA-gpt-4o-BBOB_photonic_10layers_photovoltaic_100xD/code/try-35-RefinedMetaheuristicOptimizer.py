import numpy as np

class RefinedMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = min(50, self.budget // 5)
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.initial_learning_factor = 0.9
        self.final_learning_factor = 0.4
        self.initial_temperature = 100.0
        self.cooling_rate = 0.995
        self.min_population_size = 10
        self.num_swarms = 3

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        search_space = bounds[1] - bounds[0]
        temperature = self.initial_temperature
        population_size = self.initial_population_size
        
        swarms = [
            {
                'positions': np.random.rand(population_size, self.dim) * search_space + bounds[0],
                'velocities': np.random.randn(population_size, self.dim) * 0.1 * search_space,
                'personal_best_positions': np.empty((population_size, self.dim)),
                'personal_best_scores': np.full(population_size, np.inf),
                'global_best_position': None,
                'global_best_score': np.inf
            } for _ in range(self.num_swarms)
        ]

        for swarm in swarms:
            swarm['personal_best_positions'] = np.copy(swarm['positions'])
            swarm['personal_best_scores'] = np.array([func(pos) for pos in swarm['positions']])
            best_idx = np.argmin(swarm['personal_best_scores'])
            swarm['global_best_position'] = swarm['personal_best_positions'][best_idx]
            swarm['global_best_score'] = swarm['personal_best_scores'][best_idx]

        evaluations = population_size * self.num_swarms

        while evaluations < self.budget:
            learning_factor = (self.initial_learning_factor - self.final_learning_factor) * (1 - evaluations / self.budget) + self.final_learning_factor
            
            for swarm in swarms:
                for i in range(population_size):
                    inertia = self.inertia_weight * swarm['velocities'][i] * (1 - evaluations / self.budget)
                    cognitive_component = self.cognitive_coeff * np.random.rand(self.dim) * (swarm['personal_best_positions'][i] - swarm['positions'][i])
                    social_component = self.social_coeff * np.random.rand(self.dim) * (swarm['global_best_position'] - swarm['positions'][i])
                    swarm['velocities'][i] = inertia + cognitive_component + social_component
                    swarm['velocities'][i] *= learning_factor

                    swarm['positions'][i] += swarm['velocities'][i]
                    swarm['positions'][i] = np.clip(swarm['positions'][i], bounds[0], bounds[1])

                    score = func(swarm['positions'][i])
                    evaluations += 1

                    if score < swarm['personal_best_scores'][i]:
                        swarm['personal_best_positions'][i] = swarm['positions'][i]
                        swarm['personal_best_scores'][i] = score

                    if score < swarm['global_best_score']:
                        swarm['global_best_position'] = swarm['positions'][i]
                        swarm['global_best_score'] = score

                    if np.random.rand() < np.exp(-abs(score - swarm['global_best_score']) / temperature):
                        swarm['velocities'][i] *= np.random.rand() * 2

                if evaluations / self.budget > 0.5:
                    population_size = max(self.min_population_size, population_size - 1)
                    swarm['positions'] = swarm['positions'][:population_size]
                    swarm['velocities'] = swarm['velocities'][:population_size]
                    swarm['personal_best_positions'] = swarm['personal_best_positions'][:population_size]
                    swarm['personal_best_scores'] = swarm['personal_best_scores'][:population_size]

            temperature *= self.cooling_rate

        best_swarm = min(swarms, key=lambda s: s['global_best_score'])
        return best_swarm['global_best_position'], best_swarm['global_best_score']