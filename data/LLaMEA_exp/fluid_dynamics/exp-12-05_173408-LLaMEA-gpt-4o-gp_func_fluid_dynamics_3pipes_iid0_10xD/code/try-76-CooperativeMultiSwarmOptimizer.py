import numpy as np

class CooperativeMultiSwarmOptimizer:
    def __init__(self, budget, dim, num_swarms=3, num_particles=10):
        self.budget = budget
        self.dim = dim
        self.num_swarms = num_swarms
        self.num_particles = num_particles
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        
        self.swarms = [self._initialize_swarm() for _ in range(num_swarms)]
        self.global_best_position = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)  # Initialize to avoid NoneType
        self.global_best_value = float('inf')
        self.evaluations = 0

    def _initialize_swarm(self):
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        best_positions = positions.copy()
        best_values = np.full(self.num_particles, float('inf'))
        return {'positions': positions, 'velocities': velocities, 'best_positions': best_positions, 'best_values': best_values}

    def _update_particles(self, swarm, global_best_position):
        w = 0.5 + 0.4 * (1 - self.evaluations / self.budget)  # Dynamic inertia weight
        c1 = 1.8  # Increased cognitive coefficient
        c2 = 1.2  # Decreased social coefficient
        r1, r2 = np.random.rand(2, self.num_particles, self.dim)
        
        swarm['velocities'] = (
            w * swarm['velocities'] 
            + c1 * r1 * (swarm['best_positions'] - swarm['positions']) 
            + c2 * r2 * (global_best_position - swarm['positions'])
        )
        swarm['positions'] += swarm['velocities']
        np.clip(swarm['positions'], self.lower_bound, self.upper_bound, out=swarm['positions'])

    def _evaluate_particles(self, swarm, func):
        for i in range(self.num_particles):
            if self.evaluations >= self.budget:
                break
            position = swarm['positions'][i]
            value = func(position)
            self.evaluations += 1

            if value < swarm['best_values'][i]:
                swarm['best_values'][i] = value
                swarm['best_positions'][i] = position

            if value < self.global_best_value:
                self.global_best_value = value
                self.global_best_position = position + 0.05 * np.random.uniform(-1, 1, self.dim)

    def __call__(self, func):
        while self.evaluations < self.budget:
            for swarm in self.swarms:
                self._update_particles(swarm, self.global_best_position)
                self._evaluate_particles(swarm, func)

                # Encourage cooperation between swarms
                for other_swarm in self.swarms:
                    if other_swarm is not swarm:
                        other_swarm_best = np.min(other_swarm['best_values'])
                        if other_swarm_best < swarm['best_values'].mean():
                            # Attract swarm towards other swarm's best known positions
                            self._update_particles(swarm, other_swarm['best_positions'][np.argmin(other_swarm['best_values'])])

        return self.global_best_position, self.global_best_value