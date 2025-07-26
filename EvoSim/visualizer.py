"""
Main script that controls the visualization utility of the speciation simulation
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np

from .speciation import check_speciation


class GridVisualizer:
    """
    Grid visualizer for the speciation simulation.
    """

    def __init__(self,
                 simulation,
                 *,
                 interval: int = 200,
                 show_energy: bool = False,
                 figsize: tuple[int, int] = (8, 8)):
        """
        Initialize the grid visualizer.

        Parameters:
            simulation: The simulation instance to be visualised.
            interval: Delay between frames in milliseconds.
            show_energy: If True, individuals are colour-mapped by their energy and a colour-bar is displayed. Otherwise, all individuals are rendered in blue.
            figsize: The size of the figure in inches.
        Returns:
            None
        """
        self.sim = simulation
        self.interval = interval
        self.show_energy = show_energy

        # Basic figure/axes setup
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self._init_axes()

        # Start the animation loop
        self._ani = animation.FuncAnimation(
            self.fig,
            self._update,
            interval=self.interval,
            blit=False,
            cache_frame_data=False,
        )


    def show(self):
        """Display the animation window."""
        plt.show()

    def _init_axes(self):
        """Initialize the graphical window"""
        # Set the limits of the axes
        self.ax.set_xlim(0, self.sim.size[0])
        self.ax.set_ylim(0, self.sim.size[1])
        self.ax.set_aspect("equal")
        self.ax.invert_yaxis()

        # Light grid lines
        self.ax.set_xticks(np.arange(0, self.sim.size[0] + 1, 1))
        self.ax.set_yticks(np.arange(0, self.sim.size[1] + 1, 1))
        self.ax.grid(True, lw=0.2, color="lightgray")

        # Remove axis labels for a cleaner look
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])


    def _draw_environment(self):
        """Draw environment components (barriers and food)"""
        for (x, y), cell in self.sim.grid.items():  # For each cell in the grid
            if cell["blocked"]:  # Check if the cell is blocked
                self.ax.add_patch(patches.Rectangle((x, y), 1, 1, color="black", zorder=0))
            elif cell["food"] is not None:  # Check if the cell contains food
                self.ax.add_patch(patches.Rectangle((x + 0.3, y + 0.3), 0.4, 0.4, color="green", alpha=0.6, zorder=0))


    def _update(self, frame):
        """Update the simulation"""
        if self.sim.extinct:  # Check if the population is extinct
            self.ax.set_title("Population extinct.")
            return []

        # Advance one simulation step
        self.sim.step()

        # Clear previous frame (but keep grid lines)
        self.ax.cla()
        self._init_axes()

        # Draw environment components first
        self._draw_environment()

        # Collect individual positions and energies (if requested)
        xs, ys, energies = [], [], []
        for (x, y), cell in self.sim.grid.items():
            ind = cell["individual"]
            if ind is None:
                continue
            xs.append(x + 0.5)
            ys.append(y + 0.5)
            energies.append(ind["energy"])

        # Draw individuals
        if xs:  # Check if there are any individuals
            scatter = self.ax.scatter(  # Draw the individuals
                xs,
                ys,
                c=energies if self.show_energy else "royalblue",
                cmap="plasma",
                s=80,
                edgecolors="k",
                linewidths=0.5,
                zorder=1,
            )
            if self.show_energy:  # Check if the energy should be shown
                if not hasattr(self, "_cbar"):
                    self._cbar = self.fig.colorbar(scatter, ax=self.ax, label="Energy")
                else:
                    self._cbar.update_normal(scatter)

        # Compute the speciation rate (0-1)
        rate = check_speciation(
            self.sim.grid,
            self.sim.size,
            self.sim.barrier_orientation,
            self.sim.barrier_position,
        )
        self.ax.set_title(f"Step {self.sim.steps} - Speciation rate: {rate:.3f}")

        return []
