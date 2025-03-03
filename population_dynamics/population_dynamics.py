import customtkinter as ctk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from population import Population

# Set appearance mode and default color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


class PopulationSimulatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Population Dynamics Simulator")
        self.root.geometry("1200x1000")
        
        # Initialize population
        self.population = Population(
                            initial_size=10,
                            genome_length=20,
                            mutation_rate=0.01
                            )
        
        self.selected_individuals = []
        
        self.setup_gui()
    
    def setup_gui(self):
        # Create main frame
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create top controls frame
        self.controls_frame = ctk.CTkFrame(self.main_frame)
        self.controls_frame.pack(fill="x", padx=10, pady=10)
        
        # Population size label
        self.pop_size_label = ctk.CTkLabel(
            self.controls_frame, 
            text=f"Population Size: {len(self.population.individuals)}"
            )
        self.pop_size_label.pack(side="left", padx=10)
        
        # Generation label
        self.generation_label = ctk.CTkLabel(
            self.controls_frame, 
            text=f"Generation: {len(self.population.generations)-1}"
            )
        self.generation_label.pack(side="left", padx=10)
        
        # Produce next generation button
        self.next_generation_cb = ctk.CTkButton(
            self.controls_frame,
            text="Next Generation",
            command=self.next_generation
            )
        self.next_generation_cb.pack(side="left", padx=10)
        
        # Parameters frame
        self.params_frame = ctk.CTkFrame(self.controls_frame)
        self.params_frame.pack(side="left", padx=10)
        
        # Population size entry
        self.pop_size_entry_label = ctk.CTkLabel(self.params_frame, text="New Population Size:")
        self.pop_size_entry_label.grid(row=0, column=0, padx=5, pady=5)
        self.pop_size_entry = ctk.CTkEntry(self.params_frame, width=60)
        self.pop_size_entry.insert(0, str(len(self.population.individuals)))
        self.pop_size_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Mutation rate entry
        self.mutation_rate_label = ctk.CTkLabel(self.params_frame, text="Mutation Rate:")
        self.mutation_rate_label.grid(row=1, column=0, padx=5, pady=5)
        self.mutation_rate_entry = ctk.CTkEntry(self.params_frame, width=60)
        self.mutation_rate_entry.insert(0, str(self.population.mutation_rate))
        self.mutation_rate_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Apply parameters button
        self.apply_params_button = ctk.CTkButton(
            self.params_frame, 
            text="Apply",
            command=self.apply_parameters
            )
        self.apply_params_button.grid(row=0, column=2, rowspan=2, padx=5, pady=5)
        
        # Create middle section with individuals display and genome info
        self.middle_frame = ctk.CTkFrame(self.main_frame)
        self.middle_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left side - Individual selection
        self.individuals_frame = ctk.CTkScrollableFrame(self.middle_frame, width=400)
        self.individuals_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        self.individuals_label = ctk.CTkLabel(
            self.individuals_frame, 
            text="Current Population",
            font=ctk.CTkFont(size=16, weight="bold")
            )
        self.individuals_label.pack(anchor="w", padx=5, pady=5)
        
        # Individual buttons will be added here
        self.individual_buttons = []
        self.update_individual_buttons()
        
        # Right side - Genome and analysis display
        self.info_frame = ctk.CTkFrame(self.middle_frame)
        self.info_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # Genome display
        self.genome_label = ctk.CTkLabel(
            self.info_frame, 
            text="Genome Information",
            font=ctk.CTkFont(size=16, weight="bold")
            )
        self.genome_label.pack(anchor="w", padx=5, pady=5)
        self.genome_text = ctk.CTkTextbox(self.info_frame, height=100)
        self.genome_text.pack(fill="x", padx=5, pady=5)
        
        # Analysis buttons
        self.analysis_frame = ctk.CTkFrame(self.info_frame)
        self.analysis_frame.pack(fill="x", padx=5, pady=5)
        
        self.mrca_button = ctk.CTkButton(
            self.analysis_frame,
            text="Find Most Recent Common Ancestor",
            command=self.find_mrca
            )
        self.mrca_button.pack(fill="x", padx=5, pady=5)
        
        self.genetic_distance_button = ctk.CTkButton(
            self.analysis_frame,
            text="Calculate Genetic Distance",
            command=self.calculate_genetic_distance
            )
        self.genetic_distance_button.pack(fill="x", padx=5, pady=5)
        
        self.ancestry_trace_button = ctk.CTkButton(
            self.analysis_frame,
            text="Show Ancestry Trace",
            command=self.show_ancestry_trace
            )
        self.ancestry_trace_button.pack(fill="x", padx=5, pady=5)
        
        self.population_mrca_button = ctk.CTkButton(
            self.analysis_frame,
            text="Time to Population MRCA",
            command=self.show_population_mrca_time
            )
        self.population_mrca_button.pack(fill="x", padx=5, pady=5)
        
        self.analyze_relationship_button = ctk.CTkButton(
            self.analysis_frame,
            text="Analyze Population Size vs MRCA Time",
            command=self.show_size_mrca_relationship
            )
        self.analyze_relationship_button.pack(fill="x", padx=5, pady=5)
        
        # Results display
        self.results_label = ctk.CTkLabel(
            self.info_frame, 
            text="Analysis Results",
            font=ctk.CTkFont(size=16, weight="bold")
            )
        self.results_label.pack(anchor="w", padx=5, pady=5)
        
        self.results_text = ctk.CTkTextbox(self.info_frame)
        self.results_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Area for plots
        self.plot_frame = ctk.CTkFrame(self.main_frame, height=300)
        self.plot_frame.pack(fill="x", padx=10, pady=10)
        self.plot_canvas = None
    
    def update_individual_buttons(self):
        # Clear previous buttons
        for button in self.individual_buttons:
            button.destroy()
        self.individual_buttons = []
        
        # Create buttons for each individual
        for idx, individual in enumerate(self.population.individuals):
            button = ctk.CTkButton(
                self.individuals_frame,
                # text=f"Individual {idx+1} (ID: {individual.id[:8]}) (Genome: {individual.genome})",
                text=f"Individual {idx+1} (Genome: {individual.genome})",
                command=lambda ind=individual: self.toggle_individual_selection(ind),
                fg_color="gray" if individual in self.selected_individuals else None
            )
            button.pack(fill="x", padx=5, pady=2)
            self.individual_buttons.append(button)
    
    def toggle_individual_selection(self, individual):
        if individual in self.selected_individuals:
            self.selected_individuals.remove(individual)
        else:
            self.selected_individuals.append(individual)
        
        self.update_individual_buttons()
        self.update_genome_display()
    
    def update_genome_display(self):
        self.genome_text.delete("1.0", "end")
        
        for idx, individual in enumerate(self.selected_individuals):
            self.genome_text.insert("end", f"Individual {idx+1} (ID: {individual.id[:8]}):\n")
            self.genome_text.insert("end", f"Generation: {individual.generation}\n")
            self.genome_text.insert("end", f"Genome: {individual.genome}\n\n")
    
    def next_generation(self):
        try:
            new_size = int(self.pop_size_entry.get())
            if new_size <= 0:
                raise ValueError("Population size must be positive")
        except ValueError:
            new_size = len(self.population.individuals)
        
        try:
            mutation_rate = float(self.mutation_rate_entry.get())
            if not 0 <= mutation_rate <= 1:
                raise ValueError("Mutation rate must be between 0 and 1")
            
            # Update mutation rate for future individuals
            self.population.mutation_rate = mutation_rate
            
        # Value error which was raised by us in previous step due to invalid entry  
        except ValueError:
            pass
        
        # Get next generation
        self.population.get_next_generation(new_size)
        
        # Clear selection
        self.selected_individuals = []
        
        # Update UI
        self.update_labels()
        self.update_individual_buttons()
        self.genome_text.delete("1.0", "end")
        self.results_text.delete("1.0", "end")
    
    def apply_parameters(self):
        try:
            mutation_rate = float(self.mutation_rate_entry.get())
            if not 0 <= mutation_rate <= 1:
                raise ValueError("Mutation rate must be between 0 and 1")
            
            self.population.mutation_rate = mutation_rate
            
            # Update in existing individuals
            for generation in self.population.generations:
                for individual in generation:
                    individual.mutation_rate = mutation_rate
            
            self.results_text.delete("1.0", "end")
            self.results_text.insert("end", "Parameters updated successfully.\n")
            
        except ValueError as e:
            self.results_text.delete("1.0", "end")
            self.results_text.insert("end", f"Error: {str(e)}\n")
    
    def update_labels(self):
        self.pop_size_label.configure(text=f"Population Size: {len(self.population.individuals)}")
        self.generation_label.configure(text=f"Generation: {len(self.population.generations) - 1}")
    
    def find_mrca(self):
        if len(self.selected_individuals) < 2:
            self.results_text.delete("1.0", "end")
            self.results_text.insert("end", "Please select at least 2 individuals to find their MRCA.\n")
            return
        
        mrca_id, mrca_gen, mrca_ind = self.population.find_most_recent_common_ancestor(self.selected_individuals)
        
        self.results_text.delete("1.0", "end")
        
        if mrca_id is None:
            self.results_text.insert("end", "No common ancestor found for the selected individuals.\n")
            return
        
        mrca = self.population.get_individual_by_id(mrca_id)
        if mrca:
            self.results_text.insert("end", f"Most Recent Common Ancestor:\n")
            self.results_text.insert("end", f"ID: {mrca.id}\n")
            self.results_text.insert("end", f"Generation: {mrca.generation}\n")
            self.results_text.insert("end", f"Genome: {mrca.genome}\n")
            
            current_gen = len(self.population.generations) - 1
            time_to_mrca = current_gen - mrca_gen
            
            self.results_text.insert("end", f"\nTime to MRCA: {time_to_mrca} generations ago\n")
    
    def calculate_genetic_distance(self):
        if len(self.selected_individuals) != 2:
            self.results_text.delete("1.0", "end")
            self.results_text.insert("end", "Please select exactly 2 individuals to calculate genetic distance.\n")
            return
        
        ind1, ind2 = self.selected_individuals
        distance = ind1.get_genetic_distance(ind2)
        
        self.results_text.delete("1.0", "end")
        self.results_text.insert("end", f"Genetic Distance: {distance} different positions\n")
        self.results_text.insert("end", f"Genome Length: {len(ind1.genome)}\n")
        self.results_text.insert("end", f"Similarity: {(1 - distance/len(ind1.genome))*100:.2f}%\n")
    
    def show_ancestry_trace(self):
        if len(self.selected_individuals) != 1:
            self.results_text.delete("1.0", "end")
            self.results_text.insert("end", "Please select exactly 1 individual to show ancestry trace.\n")
            return
        
        individual = self.selected_individuals[0]
        ancestors = self.population.get_ancestor_trace(individual)
        
        self.results_text.delete("1.0", "end")
        self.results_text.insert("end", f"Ancestry Trace for Individual {individual.id[:8]}:\n\n")
        
        if not ancestors:
            self.results_text.insert("end", "This individual has no ancestors (it's from the initial population).\n")
            return
        
        # Sort ancestors by generation
        ancestors.sort(key=lambda x: x.generation)
        
        for idx, ancestor in enumerate(ancestors):
            self.results_text.insert("end", f"{idx+1}. Generation {ancestor.generation}: {ancestor.id[:8]}\n")
            self.results_text.insert("end", f"   Genome: {ancestor.genome}\n")
    
    def show_population_mrca_time(self):
        tmrca, = self.population.time_to_most_recent_common_ancestor()
        
        self.results_text.delete("1.0", "end")
        
        if tmrca < 0:
            self.results_text.insert("end", "No common ancestor found for the current population.\n")
            self.results_text.insert("end", "This might be because the population is from the initial generation.\n")
        else:
            self.results_text.insert("end", f"Time to Most Recent Common Ancestor for the entire population: {tmrca} generations ago\n")
    
    def show_size_mrca_relationship(self):
        self.results_text.delete("1.0", "end")
        self.results_text.insert("end", "Analyzing relationship between population size and MRCA time...\n")
        self.results_text.insert("end", "This may take a moment...\n")
        self.root.update()
        
        # Define population sizes to analyze
        # sizes = [5, 10, 20, 30, 40, 50, 60, 70, 90, 100, 200, 300, 400, 500]
        # sizes = [i for i in range(1, 501, 100)]
        sizes = [5, 10, 20, 30, 40, 50]
        
        # Run analysis
        results = self.population.analyze_population_size_vs_mrca_time(sizes, repetitions=3)
        
        # Display results
        self.results_text.delete("1.0", "end")
        self.results_text.insert("end", "Relationship Between Population Size and TMRCA:\n\n")
        
        for size, avg_time in results.items():
            self.results_text.insert("end", f"Population Size {size}: Average TMRCA = {avg_time:.2f} generations\n")
        
        # Create and display plot
        self.create_relationship_plot(results)
    
    def create_relationship_plot(self, data):
        # Clear previous plot
        if self.plot_canvas:
            self.plot_canvas.get_tk_widget().destroy()
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 4))
        
        sizes = list(data.keys())
        times = [data[size] for size in sizes]
        
        ax.plot(sizes, times, 'o-', color='blue')
        ax.set_xlabel('Population Size')
        ax.set_ylabel('Time to MRCA (generations)')
        ax.set_title('Relationship Between Population Size and Time to MRCA')
        ax.grid(True)
        
        # Create canvas
        self.plot_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(fill="both", expand=True)


def main():
    root = ctk.CTk()
    # Composition over Inheritance, very nice
    app = PopulationSimulatorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()