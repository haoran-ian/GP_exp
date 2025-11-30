import numpy as np

class MultiSwarmPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.swarm_count = 3  # Added multiple swarms
        self.inertia_weight = 0.9  
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.F = 0.5  
        self.CR = 0.9  
        
        # Initialize populations for multiple swarms
        self.populations = [np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim)) for _ in range(self.swarm_count)]
        self.velocities = [np.random.uniform(-1, 1, (self.population_size, self.dim)) for _ in range(self.swarm_count)]
        
        self.personal_best_position = [np.copy(pop) for pop in self.populations]
        self.personal_best_value = [np.full(self.population_size, float('inf')) for _ in range(self.swarm_count)]
        self.global_best_position = np.zeros(self.dim)
        self.global_best_value = float('inf')

    def __call__(self, func):
        evaluations = 0

        while evaluations < self.budget:
            # Evaluate the populations for each swarm
            for swarm in range(self.swarm_count):
                for i in range(self.population_size):
                    value = func(self.populations[swarm][i])
                    evaluations += 1
                    if value < self.personal_best_value[swarm][i]:
                        self.personal_best_value[swarm][i] = value
                        self.personal_best_position[swarm][i] = self.populations[swarm][i]
                    if value < self.global_best_value:
                        self.global_best_value = value
                        self.global_best_position = self.populations[swarm][i]

                # Update velocity and position based on PSO for each swarm
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                for i in range(self.population_size):
                    self.velocities[swarm][i] = (self.inertia_weight * self.velocities[swarm][i] +
                                                 self.cognitive_coeff * r1 * (self.personal_best_position[swarm][i] - self.populations[swarm][i]) +
                                                 self.social_coeff * r2 * (self.global_best_position - self.populations[swarm][i]))
                    self.populations[swarm][i] = np.clip(self.populations[swarm][i] + self.velocities[swarm][i], self.lower_bound, self.upper_bound)

                # Apply DE mutation strategy with adaptive F
                self.F = 0.8 - 0.4 * evaluations / self.budget
                for i in range(self.population_size):
                    if evaluations >= self.budget:
                        break
                    idxs = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = self.populations[swarm][np.random.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                    trial = np.where(np.random.rand(self.dim) < self.CR, mutant, self.populations[swarm][i])
                    trial_value = func(trial)
                    evaluations += 1
                    if trial_value < self.personal_best_value[swarm][i]:
                        self.populations[swarm][i] = trial
                        self.personal_best_value[swarm][i] = trial_value
                        self.personal_best_position[swarm][i] = trial
                        if trial_value < self.global_best_value:
                            self.global_best_value = trial_value
                            self.global_best_position = trial

            # Adaptively reduce inertia weight
            self.inertia_weight = max(0.4, self.inertia_weight * 0.99)  

        return self.global_best_position, self.global_best_value