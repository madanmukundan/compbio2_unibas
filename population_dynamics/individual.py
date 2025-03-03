import random
import uuid

class Individual:
    gene_letters = 'ACTG'

    def __init__(self, genome: str,
                 parent = None,
                 generation: int = 0,
                 mutation_rate: float = 0.01):
        self.id = str(uuid.uuid4())
        self.genome = genome
        self.parent = parent  # Direct parent
        self.ancestors = []  # List of ancestor IDs
        self.generation = generation
        self.mutation_rate = mutation_rate
        
        if parent is not None:
            self.ancestors = parent.ancestors.copy()
            self.ancestors.append(parent.id)
    
    def mutate(self) -> 'Individual':
        """Create a descendant with mutation"""
        # Should be modified to pull from poisson distribution and 
        # use random.choice position to then select the position
        new_genome = list(self.genome)  # convert to list<char>
        if self.mutation_rate > 0:
            for i in range(len(new_genome)):
                if random.random() < self.mutation_rate:
                    new_genome[i] = random.choice(Individual.gene_letters)
        
        # Construct and return new mutated individual
        return Individual(
            genome=''.join(new_genome),
            parent=self,
            generation=self.generation + 1,
            mutation_rate=self.mutation_rate
            )
    
    def get_genetic_distance(self, other: 'Individual') -> int:
        """Calculate genetic distance (number of different characters)"""
        # Genetic distance as just the number of different nucleaotides
        return sum(c1 != c2 for c1, c2 in zip(self.genome, other.genome))
    
    def print_ancestor_tree(self):
        """Print a numbered list of all ancestors including their IDs and full genomes"""
        print(f"Ancestor tree for Individual {self.id[:6]}:")
        print(f"0. Self: {self.id[:6]} - {self.genome}")
        
        for i, ancestor_id in enumerate(reversed(self.ancestors), 1):
            if self.parent is not None and ancestor_id == self.parent.id:
                prefix = "Parent"
            else:
                prefix = f"Gen -{i}"
            print(f"{i}. {prefix}: {ancestor_id[:6]} - {self.parent.genome if self.parent else 'Unknown'}")
    
    def __str__(self):
        """Return first 6 digits of uuid as representation of individual"""
        return f"Individual {self.id[:6]} (Gen {self.generation})"
    
    def __repr__(self):
        """Return detailed string representation of individual"""
        return f"Individual ID: {self.id}\nGeneration: {self.generation}\nGenome: {self.genome}"

