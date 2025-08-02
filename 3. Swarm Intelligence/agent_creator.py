from abc import ABC, abstractmethod
from agents import Agent
from agents import ProtectiveAgent1
from agents import ProtectiveAgent2
from agents import HidingAgent1
from agents import HidingAgent2


class AgentCreator(ABC):
    """
    Abstract base class for implementing agent creators.
    """

    @abstractmethod
    def create_agent(self) -> Agent:
        """
        Returns an agent instance.

        Returns
        -------
        Agent
            Implementation of base class Agent.
        """
        pass


class ProtectiveAgent1Creator(AgentCreator):
    """
    Creates protective agents of type 1.
    """

    def create_agent(self, x: float, y: float, step_size: float) -> ProtectiveAgent1:
        """
        Creates a protective agent.

        Parameters
        ----------
        x : float
            Starting horizontal position.
        y : float
            Starting vertical position.
        step_size : float
            The amount of travel performed each turn.

        Returns
        -------
        ProtectiveAgent
            Agent who tries to stand between its friend and enemy.
        """
        return ProtectiveAgent1(x, y, step_size)

class ProtectiveAgent2Creator(AgentCreator):
    """
    Creates protective agents of type 2.
    """

    def create_agent(self, x: float, y: float, step_size: float) -> ProtectiveAgent2:
        """
        Creates a protective agent.

        Parameters
        ----------
        x : float
            Starting horizontal position.
        y : float
            Starting vertical position.
        step_size : float
            The amount of travel performed each turn.

        Returns
        -------
        ProtectiveAgent
            Agent who tries to stand between its friend and enemy.
        """
        return ProtectiveAgent2(x, y, step_size)


class HidingAgent1Creator(AgentCreator):
    """
    Creates hiding agents of type 1.

    Parameters
    ----------
    distance : float
        The distance away from the friend for the agent to travel. 
    """

    def __init__(self, distance: float) -> None:
        super().__init__()
        self.distance = distance

    def create_agent(self, x: float, y: float, step_size: float) -> HidingAgent1:
        """
        Creates a protective agent.

        Parameters
        ----------
        x : float
            Starting horizontal position.
        y : float
            Starting vertical position.
        step_size : float
            The amount of travel performed each turn.

        Returns
        -------
        ProtectiveAgent
            Agent who tries to stand between its friend and enemy.
        """
        return HidingAgent1(x, y, step_size, self.distance)


class HidingAgent2Creator(AgentCreator):
    """
    Creates hiding agents of type 2.
    """

    def create_agent(self, x: float, y: float, step_size: float) -> HidingAgent2:
        """
        Creates a protective agent.

        Parameters
        ----------
        x : float
            Starting horizontal position.
        y : float
            Starting vertical position.
        step_size : float
            The amount of travel performed each turn.

        Returns
        -------
        ProtectiveAgent
            Agent who tries to stand between its friend and enemy.
        """
        return HidingAgent2(x, y, step_size)