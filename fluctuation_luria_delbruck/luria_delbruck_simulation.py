import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import threading
import pandas as pd
from LDSimulationRunner import LDSimulationRunner

# Set appearance mode and color theme for customtkinter
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class LuriaDelbruckApp(ctk.CTk):
    """GUI application for the Luria-Delbruck experiment simulation."""

    def __init__(self):
        """Initialize the application."""
        super().__init__()

        self.title("Luria-Delbruck Experiment Simulation")
        self.geometry("1200x800")
        self.minsize(1000, 700)

        # Initialize simulation runner
        self.simulation_runner = LDSimulationRunner()

        # Create main layout
        self.create_widgets()

        # Thread for running simulations
        self.simulation_thread = None
        self.is_simulation_running = False

    def create_widgets(self):
        """Create and arrange the GUI widgets."""
        # Main layout with two frames
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        # Left panel (controls)
        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Right panel (results and visualization)
        self.result_frame = ctk.CTkFrame(self)
        self.result_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Configure layouts
        self.setup_control_panel()
        self.setup_result_panel()

    def setup_control_panel(self):
        """Set up the control panel with input fields and buttons."""
        # Configure grid
        self.control_frame.grid_columnconfigure(0, weight=1)
        self.control_frame.grid_columnconfigure(1, weight=1)

        # Title
        title_label = ctk.CTkLabel(self.control_frame, text="Simulation Parameters",
                                  font=ctk.CTkFont(size=16, weight="bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(10, 20))

        # Population parameters
        row = 1
        ctk.CTkLabel(self.control_frame, text="Initial Population:").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        self.start_pop_var = ctk.StringVar(value="10")
        start_pop_entry = ctk.CTkEntry(self.control_frame, textvariable=self.start_pop_var)
        start_pop_entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")

        row += 1
        ctk.CTkLabel(self.control_frame, text="Target Population:").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        self.end_pop_var = ctk.StringVar(value="1000")
        end_pop_entry = ctk.CTkEntry(self.control_frame, textvariable=self.end_pop_var)
        end_pop_entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")

        # Division parameters
        row += 1
        ctk.CTkLabel(self.control_frame, text="Division Distribution:").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        self.division_dist_var = ctk.StringVar(value="gaussian")
        division_dist_menu = ctk.CTkOptionMenu(self.control_frame, values=["gaussian", "poisson"],
                                             variable=self.division_dist_var,
                                             command=self.on_distribution_change)
        division_dist_menu.grid(row=row, column=1, padx=5, pady=5, sticky="ew")

        row += 1
        self.dist_param_frame = ctk.CTkFrame(self.control_frame)
        self.dist_param_frame.grid(row=row, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        self.setup_distribution_params()

        # Bacterial parameters
        row += 1
        ctk.CTkLabel(self.control_frame, text="Mutation Rate:").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        self.mutation_rate_var = ctk.StringVar(value="1e-6")
        mutation_rate_entry = ctk.CTkEntry(self.control_frame, textvariable=self.mutation_rate_var)
        mutation_rate_entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")

        # Lifespan parameters
        row += 1
        ctk.CTkLabel(self.control_frame, text="Lifespan Distribution:").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        self.lifespan_dist_var = ctk.StringVar(value="fixed")
        lifespan_dist_menu = ctk.CTkOptionMenu(self.control_frame, values=["fixed", "gaussian", "poisson"],
                                             variable=self.lifespan_dist_var,
                                             command=self.on_lifespan_distribution_change)
        lifespan_dist_menu.grid(row=row, column=1, padx=5, pady=5, sticky="ew")

        row += 1
        self.lifespan_param_frame = ctk.CTkFrame(self.control_frame)
        self.lifespan_param_frame.grid(row=row, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        self.setup_lifespan_params()

        # Survival parameters
        row += 1
        survival_label = ctk.CTkLabel(self.control_frame, text="Survival Parameters",
                                     font=ctk.CTkFont(weight="bold"))
        survival_label.grid(row=row, column=0, columnspan=2, pady=(10, 5))

        row += 1
        ctk.CTkLabel(self.control_frame, text="Base Survival Probability:").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        self.base_survival_prob_var = ctk.StringVar(value="0.001")
        base_survival_entry = ctk.CTkEntry(self.control_frame, textvariable=self.base_survival_prob_var)
        base_survival_entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")

        row += 1
        ctk.CTkLabel(self.control_frame, text="Mutation Survival Boost:").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        self.mutation_survival_boost_var = ctk.StringVar(value="10.0")
        mutation_boost_entry = ctk.CTkEntry(self.control_frame, textvariable=self.mutation_survival_boost_var)
        mutation_boost_entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")

        # Simulation control
        row += 1
        ctk.CTkLabel(self.control_frame, text="Number of Runs:").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        self.num_runs_var = ctk.StringVar(value="10")
        num_runs_entry = ctk.CTkEntry(self.control_frame, textvariable=self.num_runs_var)
        num_runs_entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")

        row += 1
        self.parallel_var = ctk.BooleanVar(value=True)
        parallel_check = ctk.CTkCheckBox(self.control_frame, text="Run in Parallel",
                                       variable=self.parallel_var)
        parallel_check.grid(row=row, column=0, columnspan=2, padx=5, pady=10, sticky="w")

        # Run button
        row += 1
        self.run_button = ctk.CTkButton(self.control_frame, text="Run Simulations",
                                      command=self.run_simulations)
        self.run_button.grid(row=row, column=0, columnspan=2, padx=20, pady=20, sticky="ew")

        # Progress bar
        row += 1
        self.progress_bar = ctk.CTkProgressBar(self.control_frame)
        self.progress_bar.grid(row=row, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        self.progress_bar.set(0)

        # Status label
        row += 1
        self.status_var = tk.StringVar(value="Ready")
        status_label = ctk.CTkLabel(self.control_frame, textvariable=self.status_var)
        status_label.grid(row=row, column=0, columnspan=2, padx=5, pady=5)

        # Save results button
        row += 1
        self.save_button = ctk.CTkButton(self.control_frame, text="Save Results",
                                       command=self.save_results,
                                       state="disabled")
        self.save_button.grid(row=row, column=0, columnspan=2, padx=20, pady=20, sticky="ew")

    def setup_distribution_params(self):
        """Set up the parameters specific to the selected distribution."""
        # Clear existing widgets
        for widget in self.dist_param_frame.winfo_children():
            widget.destroy()

        # Configure grid
        self.dist_param_frame.grid_columnconfigure(0, weight=1)
        self.dist_param_frame.grid_columnconfigure(1, weight=1)

        distribution = self.division_dist_var.get()

        if distribution == "gaussian":
            # Gaussian parameters
            ctk.CTkLabel(self.dist_param_frame, text="Mean:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
            self.division_mean_var = ctk.StringVar(value="10")
            mean_entry = ctk.CTkEntry(self.dist_param_frame, textvariable=self.division_mean_var)
            mean_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

            ctk.CTkLabel(self.dist_param_frame, text="Std Dev:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
            self.division_std_var = ctk.StringVar(value="2")
            std_entry = ctk.CTkEntry(self.dist_param_frame, textvariable=self.division_std_var)
            std_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        elif distribution == "poisson":
            # Poisson parameter
            ctk.CTkLabel(self.dist_param_frame, text="Lambda:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
            self.division_lambda_var = ctk.StringVar(value="10")
            lambda_entry = ctk.CTkEntry(self.dist_param_frame, textvariable=self.division_lambda_var)
            lambda_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

    def on_distribution_change(self, _):
        """Handle change in the division distribution selection."""
        self.setup_distribution_params()

    def setup_lifespan_params(self):
        """Set up the parameters specific to the selected lifespan distribution."""
        # Clear existing widgets
        for widget in self.lifespan_param_frame.winfo_children():
            widget.destroy()

        # Configure grid
        self.lifespan_param_frame.grid_columnconfigure(0, weight=1)
        self.lifespan_param_frame.grid_columnconfigure(1, weight=1)

        distribution = self.lifespan_dist_var.get()

        if distribution == "fixed":
            # Fixed parameter
            ctk.CTkLabel(self.lifespan_param_frame, text="Lifespan:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
            self.lifespan_var = ctk.StringVar(value="20")
            lifespan_entry = ctk.CTkEntry(self.lifespan_param_frame, textvariable=self.lifespan_var)
            lifespan_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        elif distribution == "gaussian":
            # Gaussian parameters
            ctk.CTkLabel(self.lifespan_param_frame, text="Mean:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
            self.lifespan_mean_var = ctk.StringVar(value="20")
            mean_entry = ctk.CTkEntry(self.lifespan_param_frame, textvariable=self.lifespan_mean_var)
            mean_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

            ctk.CTkLabel(self.lifespan_param_frame, text="Std Dev:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
            self.lifespan_std_var = ctk.StringVar(value="5")
            std_entry = ctk.CTkEntry(self.lifespan_param_frame, textvariable=self.lifespan_std_var)
            std_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        elif distribution == "poisson":
            # Poisson parameter
            ctk.CTkLabel(self.lifespan_param_frame, text="Lambda:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
            self.lifespan_lambda_var = ctk.StringVar(value="20")
            lambda_entry = ctk.CTkEntry(self.lifespan_param_frame, textvariable=self.lifespan_lambda_var)
            lambda_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

    def on_lifespan_distribution_change(self, _):
        """Handle change in the lifespan distribution selection."""
        self.setup_lifespan_params()

    def setup_result_panel(self):
        """Set up the results panel with plots and statistics."""
        # Configure grid
        self.result_frame.grid_columnconfigure(0, weight=1)
        self.result_frame.grid_rowconfigure(0, weight=0)  # Header
        self.result_frame.grid_rowconfigure(1, weight=1)  # Plot
        self.result_frame.grid_rowconfigure(2, weight=0)  # Stats

        # Results header
        header_label = ctk.CTkLabel(self.result_frame, text="Simulation Results",
                                   font=ctk.CTkFont(size=16, weight="bold"))
        header_label.grid(row=0, column=0, pady=(10, 5))

        # Plot area
        self.plot_frame = ctk.CTkFrame(self.result_frame)
        self.plot_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

        # Initialize plot
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.text(0.5, 0.5, "Run a simulation to see results",
                    ha="center", va="center", transform=self.ax.transAxes)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Statistics panel
        self.stats_frame = ctk.CTkFrame(self.result_frame)
        self.stats_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        # Stats labels
        ctk.CTkLabel(self.stats_frame, text="Statistics",
                    font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=4, pady=5)

        # Row 1 - Mutant statistics
        ctk.CTkLabel(self.stats_frame, text="Total Runs:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.total_runs_var = tk.StringVar(value="0")
        ctk.CTkLabel(self.stats_frame, textvariable=self.total_runs_var).grid(row=1, column=1, padx=10, pady=5, sticky="w")

        ctk.CTkLabel(self.stats_frame, text="Avg. Mutant %:").grid(row=1, column=2, padx=10, pady=5, sticky="w")
        self.avg_mutant_var = tk.StringVar(value="0.0")
        ctk.CTkLabel(self.stats_frame, textvariable=self.avg_mutant_var).grid(row=1, column=3, padx=10, pady=5, sticky="w")

        # Row 2 - More mutant statistics
        ctk.CTkLabel(self.stats_frame, text="Median Mutant %:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.median_mutant_var = tk.StringVar(value="0.0")
        ctk.CTkLabel(self.stats_frame, textvariable=self.median_mutant_var).grid(row=2, column=1, padx=10, pady=5, sticky="w")

        ctk.CTkLabel(self.stats_frame, text="Mutant Variance:").grid(row=2, column=2, padx=10, pady=5, sticky="w")
        self.variance_var = tk.StringVar(value="0.0")
        ctk.CTkLabel(self.stats_frame, textvariable=self.variance_var).grid(row=2, column=3, padx=10, pady=5, sticky="w")
        
        # Row 3 - Survival statistics
        ctk.CTkLabel(self.stats_frame, text="Avg. Survival %:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.avg_survival_var = tk.StringVar(value="0.0")
        ctk.CTkLabel(self.stats_frame, textvariable=self.avg_survival_var).grid(row=3, column=1, padx=10, pady=5, sticky="w")
        
        ctk.CTkLabel(self.stats_frame, text="Avg. Surviving Mutant %:").grid(row=3, column=2, padx=10, pady=5, sticky="w")
        self.avg_surviving_mutant_var = tk.StringVar(value="0.0")
        ctk.CTkLabel(self.stats_frame, textvariable=self.avg_surviving_mutant_var).grid(row=3, column=3, padx=10, pady=5, sticky="w")

    def get_simulation_params(self):
        """Retrieve and validate simulation parameters from the GUI."""
        try:
            params = {
                "start_pop_size": int(self.start_pop_var.get()),
                "end_pop_size": int(self.end_pop_var.get()),
                "division_distribution": self.division_dist_var.get(),
                "mutation_rate": float(self.mutation_rate_var.get()),
                "lifespan_distribution": self.lifespan_dist_var.get(),
                "base_survival_prob": float(self.base_survival_prob_var.get()),
                "mutation_survival_boost": float(self.mutation_survival_boost_var.get())
            }

            # Division distribution parameters
            if params["division_distribution"] == "gaussian":
                params["division_mean"] = float(self.division_mean_var.get())
                params["division_std"] = float(self.division_std_var.get())
                params["division_lambda"] = 10  # Default value
            else:  # poisson
                params["division_lambda"] = float(self.division_lambda_var.get())
                params["division_mean"] = 10  # Default value
                params["division_std"] = 2  # Default value
                
            # Lifespan distribution parameters
            if params["lifespan_distribution"] == "fixed":
                params["lifespan"] = int(self.lifespan_var.get())
                params["lifespan_mean"] = 20  # Default value
                params["lifespan_std"] = 5  # Default value
                params["lifespan_lambda"] = 20  # Default value
            elif params["lifespan_distribution"] == "gaussian":
                params["lifespan"] = 20  # Default value
                params["lifespan_mean"] = float(self.lifespan_mean_var.get())
                params["lifespan_std"] = float(self.lifespan_std_var.get())
                params["lifespan_lambda"] = 20  # Default value
            else:  # poisson
                params["lifespan"] = 20  # Default value
                params["lifespan_mean"] = 20  # Default value
                params["lifespan_std"] = 5  # Default value
                params["lifespan_lambda"] = float(self.lifespan_lambda_var.get())

            # Validate
            if params["start_pop_size"] <= 0 or params["end_pop_size"] <= 0:
                raise ValueError("Population sizes must be positive")

            if params["mutation_rate"] < 0 or params["mutation_rate"] > 1:
                raise ValueError("Mutation rate must be between 0 and 1")
                
            if params["lifespan_distribution"] == "fixed" and params["lifespan"] <= 0:
                raise ValueError("Bacterial lifespan must be positive")
                
            if params["lifespan_distribution"] == "gaussian" and params["lifespan_std"] <= 0:
                raise ValueError("Lifespan standard deviation must be positive")
                
            if params["lifespan_distribution"] == "poisson" and params["lifespan_lambda"] <= 0:
                raise ValueError("Lifespan lambda must be positive")
                
            if params["base_survival_prob"] < 0 or params["base_survival_prob"] > 1:
                raise ValueError("Base survival probability must be between 0 and 1")
                
            if params["mutation_survival_boost"] < 0:
                raise ValueError("Mutation survival boost must be positive")

            return params, int(self.num_runs_var.get())

        except ValueError as e:
            messagebox.showerror("Invalid Parameters", str(e))
            return None, None

    def update_progress(self, current, total):
        """Update the progress bar and status label."""
        progress = current / total if total > 0 else 0
        self.progress_bar.set(progress)
        self.status_var.set(f"Running simulations: {current}/{total}")
        self.update_idletasks()

    def run_simulations(self):
        """Run the simulations with the current parameters."""
        if self.is_simulation_running:
            return

        params, num_runs = self.get_simulation_params()
        if params is None or num_runs is None:
            return

        # Disable controls during simulation
        self.run_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self.is_simulation_running = True

        # Reset progress
        self.progress_bar.set(0)
        self.status_var.set("Starting simulations...")

        # Start simulation thread
        def simulation_task():
            try:
                if self.parallel_var.get() and num_runs > 1:
                    results = self.simulation_runner.run_parallel_simulations(
                        params, num_runs, self.update_progress)
                else:
                    results = self.simulation_runner.run_sequential_simulations(
                        params, num_runs, self.update_progress)

                # Update GUI in the main thread
                self.after(0, lambda: self.update_results(results))
            except Exception as e:
                self.after(0, lambda: self.handle_simulation_error(str(e)))
            finally:
                self.is_simulation_running = False
                self.after(0, lambda: self.run_button.configure(state="normal"))

        self.simulation_thread = threading.Thread(target=simulation_task)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()

    def update_results(self, results):
        """Update the GUI with simulation results."""
        if not results:
            self.status_var.set("No results to display")
            return

        # Update plot
        self.ax.clear()
        self.simulation_runner.plot_results(self.ax)
        self.canvas.draw()

        # Update statistics
        self.total_runs_var.set(str(len(results)))

        mean_mutant = np.mean(self.simulation_runner.results)
        median_mutant = np.median(self.simulation_runner.results)
        variance_mutant = np.var(self.simulation_runner.results)
        mean_survival = np.mean(self.simulation_runner.survival_results)
        mean_surviving_mutant = np.mean(self.simulation_runner.surviving_mutant_results)

        self.avg_mutant_var.set(f"{mean_mutant:.4%}")
        self.median_mutant_var.set(f"{median_mutant:.4%}")
        self.variance_var.set(f"{variance_mutant:.6f}")
        self.avg_survival_var.set(f"{mean_survival:.4%}")
        self.avg_surviving_mutant_var.set(f"{mean_surviving_mutant:.4%}")

        # Update status
        self.status_var.set(f"Completed {len(results)} simulations")
        self.progress_bar.set(1.0)

        # Enable save button
        self.save_button.configure(state="normal")

    def handle_simulation_error(self, error_msg):
        """Handle errors that occur during simulation."""
        messagebox.showerror("Simulation Error", f"An error occurred: {error_msg}")
        self.status_var.set("Error occurred during simulation")
        self.progress_bar.set(0)

    def save_results(self):
        """Save the simulation results to a CSV file."""
        if not self.simulation_runner.results:
            messagebox.showinfo("No Results", "No simulation results to save")
            return

        try:
            # Create a data frame with results
            data = {
                "Run": list(range(1, len(self.simulation_runner.results) + 1)),
                "Mutant_Proportion": self.simulation_runner.results,
                "Survival_Proportion": self.simulation_runner.survival_results,
                "Surviving_Mutant_Proportion": self.simulation_runner.surviving_mutant_results,
                "Population_Size": [result["population_size"] for result in self.simulation_runner.raw_results],
                "Mutant_Count": [result["mutant_count"] for result in self.simulation_runner.raw_results],
                "Surviving_Count": [result["surviving_count"] for result in self.simulation_runner.raw_results],
                "Surviving_Mutant_Count": [result["surviving_mutant_count"] for result in self.simulation_runner.raw_results],
                "Total_Divisions": [result["total_divisions"] for result in self.simulation_runner.raw_results],
                "Total_Mutations": [result["total_mutations"] for result in self.simulation_runner.raw_results],
                "Simulation_Time": [result["final_time"] for result in self.simulation_runner.raw_results]
            }

            df = pd.DataFrame(data)

            # Save to file
            filename = f"luria_delbruck_results_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)

            messagebox.showinfo("Save Successful", f"Results saved to {filename}")

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save results: {str(e)}")


def main():
    """Main function to run the application."""
    # Create and run the application
    app = LuriaDelbruckApp()
    app.mainloop()


if __name__ == "__main__":
    main()
