import numpy as np

class RefinedHybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_prob = 0.9
        self.evaluations = 0
    
    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        pop_size = self.initial_pop_size
        population = self._initialize_population(bounds, pop_size)
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations += len(fitness)

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        while self.evaluations < self.budget:
            if self.evaluations > self.budget * 0.5:
                # Adaptively resize the population
                pop_size = max(4, int(pop_size * 0.9))
                population = population[:pop_size]
                fitness = fitness[:pop_size]

            population = self._differential_evolution(population, fitness, bounds, func, best_solution)
            if self.evaluations < self.budget:
                population, fitness = self._enhanced_local_search(population, fitness, bounds, func)

            best_idx = np.argmin(fitness)
            current_best_solution = population[best_idx]
            current_best_fitness = fitness[best_idx]

            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_solution = current_best_solution

        return best_solution
    
    def _initialize_population(self, bounds, size):
        return bounds[0] + (bounds[1] - bounds[0]) * np.random.rand(size, self.dim)

    def _differential_evolution(self, pop, fitness, bounds, func, best_solution):
        new_pop = np.copy(pop)
        for i in range(len(pop)):
            idxs = [idx for idx in range(len(pop)) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = a + self.mutation_factor * (b - c) + 0.1 * (best_solution - pop[i])
            mutant = np.clip(mutant, bounds[0], bounds[1])
            
            cross_points = np.random.rand(self.dim) < self.crossover_prob
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            
            trial = np.where(cross_points, mutant, pop[i])
            trial_fitness = func(trial)
            self.evaluations += 1
            
            if trial_fitness < fitness[i]:
                new_pop[i] = trial
                fitness[i] = trial_fitness
        
        return new_pop

    def _enhanced_local_search(self, pop, fitness, bounds, func):
        for i in range(len(pop)):
            perturbation = np.random.normal(0, 0.1, self.dim) * (bounds[1] - bounds[0])
            perturbed = np.clip(pop[i] + perturbation, bounds[0], bounds[1])
            perturbed_fitness = func(perturbed)
            self.evaluations += 1
            
            if perturbed_fitness < fitness[i]:
                pop[i] = perturbed
                fitness[i] = perturbed_fitness
            else:
                # Additional local search step
                further_perturbed = np.clip(perturbed + perturbation * 0.5, bounds[0], bounds[1])
                further_fitness = func(further_perturbed)
                self.evaluations += 1

                if further_fitness < fitness[i]:
                    pop[i] = further_perturbed
                    fitness[i] = further_fitness

        return pop, fitness