class EnhancedSwarmOptimizerV2:
    # ... (rest of the code remains unchanged)
    
    def update_velocities_and_positions(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.inertia_weight = 0.5 + (0.4 * np.cos(np.pi * self.fitness_evals / self.budget))
        elite = self.select_elite_members()
        for i in range(self.population_size):
            inertia = self.inertia_weight * self.velocities[i]
            cognitive = self.c1 * np.random.rand(self.dim) * (self.personal_best[i] - self.population[i])
            social = self.c2 * np.random.rand(self.dim) * (self.global_best - self.population[i])
            elite_velocity = np.mean(elite, axis=0) - self.population[i]
            elite_influence = 0.25 * (elite_velocity + 0.8 * self.velocities[i])  # Modified line
            neighborhood_influence = self.adaptive_neighborhood_influence(i)
            self.velocities[i] = inertia + cognitive + social + elite_influence + 0.15 * neighborhood_influence
            self.velocity_clamp = 0.1 * (1 - (self.fitness_evals / self.budget))
            self.velocities[i] = np.clip(self.velocities[i], -self.velocity_clamp, self.velocity_clamp)
            self.population[i] += self.velocities[i] * self.local_search_adaptive_factor(i)
            self.population[i] = self.mutate(self.population[i], lb, ub)
            self.population[i] = np.clip(self.population[i], lb, ub)
    
    # ... (rest of the code remains unchanged)