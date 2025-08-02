from agent_creator import AgentCreator
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Environment:
    """
    An 2D world where a swarm of agents interact.

    Parameters
    ----------
    n_agents : int
        The number of agents to use.
    step_size : float
        The amount of travel each agent does per step.
    grid_size : tuple of two floats
        The size of the 2D world.
        The first number represents the horizontal length.
        The second number represents the vertical length.
    agent_creator : Implementation of AgentCreator
        Object which produces agents.
    """

    def __init__(self, 
            n_agents: int, 
            step_size: float,
            grid_size: tuple,
            agent_creator: AgentCreator
        ) -> None:
        self.n_agents = n_agents
        self.step_size = step_size
        self.grid_size = grid_size
        self.agent_creator = agent_creator

        # Place holder list to store agents
        self.agents = []

    def __random_agent_assignment(self) -> None:
        """
        Assigns an enemy and friend by random sampling with replacements.
        """
        for i in range(self.n_agents):
            other_agents = list(range(0 , i)) + list(range(i+1, self.n_agents))
            friend_idx, enemy_idx = random.sample(other_agents, 2)
            self.agents[i].friend = self.agents[friend_idx]
            self.agents[i].enemy = self.agents[enemy_idx]

    def __neighbours_agent_assignment(self) -> None:
        """
        Assigns the previous agent in the list as friend and the next as enemy.
        """
        for i in range(self.n_agents):
            if i == 0:
                friend_idx, enemy_idx = -1, i+1
            elif i == self.n_agents-1:
                friend_idx, enemy_idx = -2, 0
            else:
                friend_idx, enemy_idx = i-1, i+1
            self.agents[i].friend = self.agents[friend_idx]
            self.agents[i].enemy = self.agents[enemy_idx]

    def reset(self, assignment_type: str) -> None:
        """
        Resets the environment by reinitialising the agents and their
        friend and enemy.

        Parameters
        ----------
        assignment_type : str
            How to assign friends and enemies.
            If 'random':
                Randomly samples with replacement.
            If 'neighbour':
                Selects previous agent as friend and the next as enemy.
        """
        self.agents = []

        # Initialize agents and their starting position
        for _ in range(self.n_agents):
            x = np.random.rand() * self.grid_size[0]
            y = np.random.rand() * self.grid_size[1]
            self.agents.append(
                self.agent_creator.create_agent(
                    x=x, 
                    y=y, 
                    step_size=self.step_size
                )
            )

        # Assign a friend and an enemy to each agent
        if assignment_type == "random":
            self.__random_agent_assignment()
        elif assignment_type == "neighbours":
            self.__neighbours_agent_assignment()

    def step(self) -> None:
        """
        Take one timestep in the environment.
        """
        # All agents update their perceptions simultaneously
        for agent in self.agents:
            agent.update_state()

        # All agents move simultaneously
        for agent in self.agents:
            agent.move()

    def format_agent_positions(self) -> np.ndarray:
        """
        Formats the agents' positions for plotting.

        Returns
        -------
        np.array
            A stacked array of all positions.
        """
        return np.stack([agent.position for agent in self.agents])

    def render(self, i):
        """
        Runs and renders one step of the environment.

        Parameters
        ----------
        i : int
            Not used. Only for matplotlib animation compatability.

        Returns
        -------
        matplotlib.collections.PathCollection
            A matlotlib scatter plot.
        """
        self.step()
        positions = self.format_agent_positions()
        self.sctr.set_offsets(positions)
        return self.sctr

    def init_render(self):
        """
        Initial render of the enrivoment.

        Returns
        -------
        matplotlib.collections.PathCollection
            A matlotlib scatter plot.
        """
        positions = self.format_agent_positions()
        self.sctr = plt.scatter(positions[:, 0], positions[:, 1])
        return self.sctr
        
    def run(self, xlim: tuple=None, ylim: tuple=None) -> None:
        """
        Runs the enironment.
        """
        fig, ax = plt.subplots()
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        ani = FuncAnimation(
            fig, 
            func=self.render, 
            init_func=self.init_render,
            interval=40
        )

        plt.show()
