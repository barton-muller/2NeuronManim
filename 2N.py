from manim import *
import numpy as np

class TwoNeuronRNNOrbit(Scene):
    def construct(self):
        # Create axes for the orbit plot
        axes = Axes(
            x_range=[-2, 2, 0.5], y_range=[-2, 2, 0.5],
            x_length=6, y_length=6,
            axis_config={"color": BLUE}
        )
        labels = axes.get_axis_labels(x_label="h1", y_label="h2")
        self.play(Create(axes), Write(labels))

        # Initial state of the two neurons
        h0 = np.array([1.0, 0.0])
        
        # Define RNN dynamics: Weight matrix and bias
        W = np.array([[0.5, -1.0], [1.0, 0.5]])
        b = np.array([0.0, 0.0])
        
        # Function for RNN dynamics
        def rnn_update(h):
            return np.tanh(np.dot(W, h) + b)

        # Simulate neuron activities over time
        orbit_points = [h0]
        for _ in range(50):  # Simulate for 50 time steps
            h_next = rnn_update(orbit_points[-1])
            orbit_points.append(h_next)

        # Convert the simulated points to Manim points on the 2D plane
        orbit_points_manim = [
            axes.c2p(h[0], h[1]) for h in orbit_points
        ]

        # Create the orbit as a line with a moving dot
        orbit_path = VMobject(color=YELLOW)
        orbit_path.set_points_as_corners(orbit_points_manim)
        moving_dot = Dot(orbit_points_manim[0], color=RED)

        # Animation: Draw the orbit path and move the dot along the path
        self.play(Create(orbit_path), FadeIn(moving_dot))
        self.play(MoveAlongPath(moving_dot, orbit_path), run_time=8, rate_func=linear)

        # Hold the final state for a moment
        self.wait(2)
