import numpy as np

class EnhancedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight_max = 0.9
        self.inertia_weight_min = 0.4
        self.cognitive_param = 1.5
        self.social_param = 1.5
        self.mutation_factor = 0.5
        self.crossover_prob = 0.7
        self.iteration = 0
        self.num_swarms = 2

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.num_swarms, self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_swarms, self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([[func(p) for p in swarm] for swarm in particles])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = particles.reshape(-1, self.dim)[global_best_index]
        evaluations = self.num_swarms * self.population_size

        while evaluations < self.budget:
            inertia_weight = self.inertia_weight_max - ((self.inertia_weight_max - self.inertia_weight_min) * (evaluations / self.budget))
            for swarm_idx in range(self.num_swarms):
                for i in range(self.population_size):
                    r1, r2 = np.random.rand(2)
                    velocities[swarm_idx, i] = (inertia_weight * velocities[swarm_idx, i] +
                                                self.cognitive_param * r1 * (personal_best_positions[swarm_idx, i] - particles[swarm_idx, i]) +
                                                self.social_param * r2 * (global_best_position - particles[swarm_idx, i]))
                    particles[swarm_idx, i] = particles[swarm_idx, i] + velocities[swarm_idx, i]
                    particles[swarm_idx, i] = np.clip(particles[swarm_idx, i], lb, ub)
                    
                    score = func(particles[swarm_idx, i])
                    evaluations += 1

                    if score < personal_best_scores[swarm_idx, i]:
                        personal_best_scores[swarm_idx, i] = score
                        personal_best_positions[swarm_idx, i] = particles[swarm_idx, i]

                swarm_best_index = np.argmin(personal_best_scores[swarm_idx])
                swarm_best_position = personal_best_positions[swarm_idx, swarm_best_index]
                
                if personal_best_scores[swarm_idx, swarm_best_index] < np.min(personal_best_scores):
                    global_best_position = swarm_best_position
            
            for swarm_idx in range(self.num_swarms):
                for i in range(self.population_size):
                    if evaluations >= self.budget:
                        break
                    idxs = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = particles[swarm_idx, np.random.choice(idxs, 3, replace=False)]
                    mutant = a + self.mutation_factor * (b - c)
                    mutant = np.clip(mutant, lb, ub)

                    crossover_mask = np.random.rand(self.dim) < self.crossover_prob
                    trial = np.where(crossover_mask, mutant, particles[swarm_idx, i])
                    trial_score = func(trial)
                    evaluations += 1

                    if trial_score < personal_best_scores[swarm_idx, i]:
                        particles[swarm_idx, i] = trial
                        personal_best_scores[swarm_idx, i] = trial_score
                        personal_best_positions[swarm_idx, i] = trial

            self.iteration += 1
        
        return global_best_position