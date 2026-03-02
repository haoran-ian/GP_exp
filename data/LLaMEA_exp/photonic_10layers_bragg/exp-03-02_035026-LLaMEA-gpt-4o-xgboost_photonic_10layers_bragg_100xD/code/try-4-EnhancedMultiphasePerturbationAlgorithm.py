import numpy as np
from sklearn.cluster import KMeans

class EnhancedMultiphasePerturbationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = np.random.uniform(lb, ub, size=self.dim)
        best_fitness = func(best_solution)
        evaluations = 1
        history = [best_solution]

        while evaluations < self.budget:
            phase = evaluations / self.budget
            
            if phase < 0.3:  # Exploration Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=0.5 * self._fitness_variance(best_fitness))
            elif phase < 0.7:  # Balanced Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=0.2 * self._fitness_variance(best_fitness))
            else:  # Exploitation Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=0.05 * self._fitness_variance(best_fitness))
                
            candidate_fitness = np.array([func(sol) for sol in candidate_solutions])
            evaluations += len(candidate_solutions)

            # Update history and cluster solutions
            history.extend(candidate_solutions)
            if len(history) > 50:
                history = history[-50:]
            clusters, cluster_fitness = self._cluster_solutions(np.array(history), func)
            
            # Select the best candidate from clusters
            min_idx = np.argmin(cluster_fitness)
            if cluster_fitness[min_idx] < best_fitness:
                best_solution = clusters[min_idx]
                best_fitness = cluster_fitness[min_idx]
        
        return best_solution
    
    def _generate_solutions(self, center, lb, ub, scale):
        perturbations = np.random.uniform(-scale, scale, size=(10, self.dim))
        solutions = center + perturbations * (ub - lb)
        return np.clip(solutions, lb, ub)

    def _fitness_variance(self, best_fitness):
        return max(0.1, np.abs(best_fitness) / 10)

    def _cluster_solutions(self, solutions, func):
        kmeans = KMeans(n_clusters=min(len(solutions), 5), random_state=0).fit(solutions)
        cluster_centers = kmeans.cluster_centers_
        cluster_fitness = [func(center) for center in cluster_centers]
        return cluster_centers, cluster_fitness