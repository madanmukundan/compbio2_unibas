import random
from typing import List, Dict, Tuple, Optional
from individual import Individual
import concurrent.futures
import numpy as np
import itertools

class Population:
    def __init__(self,
                 initial_size: int = 10,
                 genome_length: int = 500,
                 mutation_rate: float = 0.01):

        self.individuals: List[Individual] = []
        self.generations: List[List[Individual]] = []
        self.genome_length = genome_length
        self.mutation_rate = mutation_rate
        
        # Generate initial population
        self.initialize_population(initial_size)
    
    def initialize_population(self, population_size: int):
        """Generate initial population with random genomes"""
        initial_generation = []
        for _ in range(population_size):
            genome = ''.join(random.choice(Individual.gene_letters) for _ in range(self.genome_length))
            individual = Individual(genome=genome, mutation_rate=self.mutation_rate)
            initial_generation.append(individual)
        
        self.individuals = initial_generation
        self.generations = [initial_generation]
    
    def get_next_generation(self, new_size: Optional[int] = None):
        """Generate the next generation through reproduction and mutation"""
        if new_size is None:
            new_size = len(self.individuals)
        
        # Select parents (with replacement) and create offspring
        parents = random.choices(self.individuals, k=new_size)  # with replacement
        new_generation = [parent.mutate() for parent in parents]
        
        self.individuals = new_generation
        self.generations.append(new_generation)
        
        return new_generation
    
    def find_most_recent_common_ancestor(self, individuals: List[Individual]) -> Tuple[Optional[str], int, Optional[Individual]]:
        """Find the most recent common ancestor of given individuals"""
        if not individuals or len(individuals) < 2:
            return None, -1, None
        
        # Get sets of ancestor id's for each individual
        ancestor_sets = []
        for individual in individuals:
            ancestors = set(individual.ancestors)  # convert to set for quick comparison
            ancestors.add(individual.id)  # Include the individual itself
            ancestor_sets.append(ancestors)
        
        # Find common ancestors, set intersection
        common_ancestors_ids = set.intersection(*ancestor_sets)
        
        if not common_ancestors_ids:
            return None, -1, None
        
        # Find the most recent common ancestor
        most_recent_ancestor_id = None
        most_recent_gen = -1
        most_recent_ancestor = None
        
        # Since full ancestor Individual not saved, parse generations lists to find the MRCA
        for gen_idx, generation in reversed(list(enumerate(self.generations))):
            for individual in generation:
                # Technically is BFS dummy search but compares to set so O(n) for every individual
                if individual.id in common_ancestors_ids and gen_idx > most_recent_gen:
                    most_recent_gen = gen_idx  # correctly goes back to front
                    most_recent_ancestor_id = individual.id
                    most_recent_ancestor = individual
                    # Short circuit after searching from most recent, only hits once :(
                    return most_recent_ancestor_id, most_recent_gen, most_recent_ancestor

        return None, -1, None
    
    def get_individual_by_id(self, individual_id: str) -> Optional[Individual]:
        """Get individual by ID from all generations"""
        for generation in self.generations:
            for individual in generation:
                if individual.id == individual_id:
                    return individual
        return None
    
    def time_to_most_recent_common_ancestor(self) -> int:
        """Calculate how many generations ago all current individuals had a common ancestor"""
        mrca_id, mrca_gen, mrca_ind = self.find_most_recent_common_ancestor(self.individuals)
        if mrca_id is None:
            return -1, None
        
        # Time from current generation to MRCA
        current_gen = len(self.generations) - 1
        return current_gen - mrca_gen, mrca_ind
        # return mrca_gen, mrca_ind
    
    def get_ancestor_trace(self, individual: Individual) -> List[Individual]:
        """Get the complete ancestry trace of an individual"""
        trace = []
        for ancestor_id in individual.ancestors:
            ancestor = self.get_individual_by_id(ancestor_id)
            if ancestor:
                trace.append(ancestor)
        return trace
    
    @staticmethod
    def run_single_simulation(size: int, genome_length: int, mutation_rate: int) -> int:
        """Helper method to run a single simulation for a given population size"""

        # Create a new population instance to avoid state conflicts
        pop = Population(initial_size=size, 
                        genome_length=genome_length,
                        mutation_rate=mutation_rate)
        
        # Evolve for some max number of generations -> 5000
        # E_v=2N, sig=2N, should 
        # for _ in range(max(5000, size*10)):
        #     pop.get_next_generation(size)
        #     # Check for convergence if not continue
        #     tmrca, _ = pop.time_to_most_recent_common_ancestor()
        #     if tmrca >= 0:
        #         return tmrca 

        for _ in range(round(size*5)):
            pop.get_next_generation(size)
            # Check for convergence if not continue

        tmrca, _ = pop.time_to_most_recent_common_ancestor()

        # will return -1 if non-convergence
        return tmrca 
        
    def analyze_population_size_vs_mrca_time(self, sizes: List[int], repetitions: int = 5) -> Dict:
        """Analyze relationship between population size and time to MRCA using parallel processing"""
        results = {pop_size: [] for pop_size in sizes}
        
        # Create all simulation parameters
        simulation_params = [(pop_size,) for pop_size in sizes for _ in range(repetitions)]
        
        # Run simulations in parallel or die trying, default to number of CPUs*5
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Should be no deadlocks as runs are fully independent
            futures = [executor.submit(self.run_single_simulation, pop_size, self.genome_length, self.mutation_rate) 
                      for pop_size, in simulation_params]
            
            # Collect results as they complete
            for (pop_size,), future in zip(simulation_params, futures):
                tmrca = future.result()
                if tmrca > 0:  # Non-convergence will return -1
                    # Python list is thread-safe for append
                    results[pop_size].append(tmrca)
        
        # Calculate averages
        averages = {pop_size: sum(times)/len(times) if times else 0 
                   for pop_size, times in results.items()}
        
        return averages, results
    
    def calculate_genetic_distances(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Calculate genetic distances between all individuals. """
        n_individuals = len(self.individuals)
        
        # Initialize the distance matrix with zeros
        genetic_distances = np.zeros((n_individuals, n_individuals))
        
        # Symmetric matrix of genetic distances
        for i in range(n_individuals):
            for j in range(i+1, n_individuals):
                distance = self.individuals[i].get_genetic_distance(self.individuals[j])
                genetic_distances[i, j] = distance
                genetic_distances[j, i] = distance
        
        # Get mean of lower triangular matrix of genetic distances
        mean_distance = np.mean(np.tril(genetic_distances, k=-1))
        
        return genetic_distances, mean_distance
    
    def get_all_individual_genomes(self) -> List[str]:
        """Get Genomes of all individuals in the population"""
        genomes = (individual.genome for individual in self.individuals)
        for generation in self.generations:
            new_genomes = (individual.genome for individual in generation)
            genomes = itertools.chain(genomes, new_genomes)
        return list(genomes)
    
    def __str__(self):
        current_gen = len(self.generations) - 1
        population_size = len(self.individuals)
        individuals_str = '\n'.join(f"Individual #{i:0.0f}, {v.id[:6]}: {v.genome}" for i,v in enumerate(self.individuals))
        return f"Generation #{current_gen}, Population size: {population_size}\n{individuals_str}"
