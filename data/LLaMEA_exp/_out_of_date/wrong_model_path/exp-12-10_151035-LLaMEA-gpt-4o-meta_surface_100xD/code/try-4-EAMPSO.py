import numpy as np

class EAMPSO:
    def __init__(self, budget, dim, population_size=30, w=0.9, c1=2.0, c2=2.0, inertia_decay=0.99, local_search_radius=0.1):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.w = w  # initial inertia weight
        self.c1 = c1  # cognitive (personal) coefficient
        self.c2 = c2  # social (global) coefficient
        self.inertia_decay = inertia_decay  # decay factor for inertia weight
        self.local_search_radius = local_search_radius  # radius for local search
        self.global_best_position = None
        self.global_best_value = np.inf

    def local_search(self, position, func, lower_bounds, upper_bounds):
        # Perform a simple random local search around a position
        perturbation = np.random.uniform(-self.local_search_radius, self.local_search_radius, self.dim)
        candidate_position = np.clip(position + perturbation, lower_bounds, upper_bounds)
        candidate_value = func(candidate_position)
        return candidate_position, candidate_value

    def __call__(self, func):
        lower_bounds = np.array(func.bounds.lb)
        upper_bounds = np.array(func.bounds.ub)

        # Initialize particles
        particles = np.random.uniform(lower_bounds, upper_bounds, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_values = np.full(self.population_size, np.inf)

        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Evaluate current position
                value = func(particles[i])
                evaluations += 1

                # Update personal best
                if value < personal_best_values[i]:
                    personal_best_values[i] = value
                    personal_best_positions[i] = particles[i].copy()

                # Update global best
                if value < self.global_best_value:
                    self.global_best_value = value
                    self.global_best_position = particles[i].copy()
                    
                    # Local search around the new global best
                    candidate_position, candidate_value = self.local_search(self.global_best_position, func, lower_bounds, upper_bounds)
                    evaluations += 1
                    if candidate_value < self.global_best_value:
                        self.global_best_value = candidate_value
                        self.global_best_position = candidate_position.copy()

            # Update particles' velocities and positions
            for i in range(self.population_size):
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_component = self.c1 * r1 * (personal_best_positions[i] - particles[i])
                social_component = self.c2 * r2 * (self.global_best_position - particles[i])
                velocities[i] = self.w * velocities[i] + cognitive_component + social_component

                # Update particle position
                particles[i] += velocities[i]

                # Ensure particles are within bounds
                particles[i] = np.clip(particles[i], lower_bounds, upper_bounds)

            # Decay inertia weight
            self.w *= self.inertia_decay

        return self.global_best_position, self.global_best_value