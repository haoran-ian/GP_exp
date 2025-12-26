import numpy as np

class DynamicMultiSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_count = 3
        self.population_size = 50
        self.inertia_weight = 0.9
        self.inertia_min = 0.4
        self.cognitive_coefficient = 2.0
        self.social_coefficient = 2.0
        self.vel_clamp = None
        self.swarms = [self.initialize_swarm() for _ in range(self.swarm_count)]
        self.best_global_position = None
        self.best_global_score = np.inf

    def initialize_swarm(self):
        return {
            'population': None,
            'velocities': None,
            'personal_best_positions': None,
            'personal_best_scores': None,
            'global_best_position': None,
            'global_best_score': np.inf
        }
    
    def initialize_population(self, lb, ub, swarm):
        swarm['population'] = np.random.uniform(lb, ub, (self.population_size, self.dim))
        swarm['velocities'] = np.random.uniform(-0.1, 0.1, (self.population_size, self.dim))
        swarm['personal_best_positions'] = np.copy(swarm['population'])
        swarm['personal_best_scores'] = np.full(self.population_size, np.inf)
        swarm['global_best_position'] = np.random.uniform(lb, ub, self.dim)
        self.vel_clamp = 0.1 * (ub - lb)

    def update_particles(self, swarm, lb, ub):
        r1 = np.random.rand(self.population_size, self.dim)
        r2 = np.random.rand(self.population_size, self.dim)

        cognitive_velocity = self.cognitive_coefficient * r1 * (swarm['personal_best_positions'] - swarm['population'])
        social_velocity = self.social_coefficient * r2 * (swarm['global_best_position'] - swarm['population'])

        swarm['velocities'] = self.inertia_weight * swarm['velocities'] + cognitive_velocity + social_velocity
        swarm['velocities'] = np.clip(swarm['velocities'], -self.vel_clamp, self.vel_clamp)

        swarm['population'] += swarm['velocities']
        
        if np.random.rand() < 0.05:
            random_indices = np.random.choice(self.population_size, size=int(0.1 * self.population_size), replace=False)
            swarm['population'][random_indices] = np.random.uniform(lb, ub, (len(random_indices), self.dim))

    def evaluate_population(self, func, swarm):
        for i in range(self.population_size):
            score = func(swarm['population'][i])
            if score < swarm['personal_best_scores'][i]:
                swarm['personal_best_scores'][i] = score
                swarm['personal_best_positions'][i] = swarm['population'][i]
            if score < swarm['global_best_score']:
                swarm['global_best_score'] = score
                swarm['global_best_position'] = swarm['population'][i]

    def adapt_parameters(self, progress):
        self.inertia_weight = self.inertia_min + (0.9 - self.inertia_min) * (1 - progress)
        self.cognitive_coefficient = 2.0 - 1.5 * progress
        self.social_coefficient = 1.0 + 1.5 * progress
        self.vel_clamp = (0.1 + 0.1 * progress) * (ub - lb)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub

        for swarm in self.swarms:
            self.initialize_population(lb, ub, swarm)

        evaluations = 0
        while evaluations < self.budget:
            progress = evaluations / self.budget
            self.adapt_parameters(progress)

            for swarm in self.swarms:
                self.update_particles(swarm, lb, ub)
                self.evaluate_population(func, swarm)
                
                if swarm['global_best_score'] < self.best_global_score:
                    self.best_global_score = swarm['global_best_score']
                    self.best_global_position = swarm['global_best_position']
            
            evaluations += self.swarm_count * self.population_size

        return self.best_global_position, self.best_global_score