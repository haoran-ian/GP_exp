import numpy as np

class AdvancedSynergisticMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialization
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 15 * self.dim
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        evaluations = population_size
        
        # Parameters
        F = 0.7  # DE mutation factor
        Cr = 0.8  # DE crossover probability
        temp = 1.0  # Initial temperature for SA
        cooling_rate = 0.98
        neighborhood_radius = 0.2  # Initial radius for neighborhood search
        phase_change_threshold = 0.2 * self.budget  # Define a phase change threshold
        
        def differential_evolution(pop, fit):
            for i in range(population_size):
                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                cross_points = np.random.rand(self.dim) < Cr + 0.05
                trial = np.where(cross_points, mutant, pop[i])
                trial_fitness = func(trial)

                if trial_fitness < fit[i]:
                    pop[i] = trial
                    fit[i] = trial_fitness

        def adaptive_neighborhood_search(ind, fit):
            new_solution = ind + np.random.normal(0, neighborhood_radius, self.dim)
            new_solution = np.clip(new_solution, lb, ub)
            new_fitness = func(new_solution)
            if new_fitness < fit:
                return new_solution, new_fitness
            return ind, fit

        # Search loop
        while evaluations < self.budget:
            if evaluations < phase_change_threshold:
                # Phase 1: Exploration
                Cr = 0.9  # Increase crossover probability for exploration
                neighborhood_radius = 0.2
                
            else:
                # Phase 2: Exploitation
                Cr = 0.6  # Decrease crossover probability for exploitation
                neighborhood_radius = 0.05

            differential_evolution(population, fitness)
            evaluations += population_size

            # Adaptive Neighborhood Search and Diversity Preservation
            for i in range(population_size):
                if evaluations >= self.budget:
                    break
                
                population[i], fitness[i] = adaptive_neighborhood_search(population[i], fitness[i])
                if fitness[i] < best_fitness:
                    best_solution = population[i]
                    best_fitness = fitness[i]
                
                # Simulated Annealing Step
                new_solution = population[i] + np.random.normal(0, 0.1, self.dim)
                new_solution = np.clip(new_solution, lb, ub)
                new_fitness = func(new_solution)
                delta = new_fitness - fitness[i]
                
                if delta < 0 or np.exp(-delta / temp) > np.random.rand():
                    population[i] = new_solution
                    fitness[i] = new_fitness
                    if new_fitness < best_fitness:
                        best_solution = new_solution
                        best_fitness = new_fitness

                evaluations += 1

            # Update cooling schedule and neighborhood radius
            temp *= cooling_rate

        return best_solution