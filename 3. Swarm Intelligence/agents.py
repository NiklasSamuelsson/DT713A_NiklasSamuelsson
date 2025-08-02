from abc import ABC, abstractmethod
import numpy as np


class Agent(ABC):
    """
    Base class for a very simple autonomous agent with a friend
    and an enemy.

    Parameters
    ----------
    x : float
        Starting horizontal position.
    y : float
        Starting vertical position.
    step_size : float
        The amount of travel performed each turn.
    """

    def __init__(self, x: float, y: float, step_size: float) -> None:

        # Initiate agent position
        self.position = (x, y)
        self.step_size = step_size

        # Placeholder for the reference to the agent's friend and enemy
        self.friend = None
        self.enemy = None

        self.friend_position = None
        self.enemy_position = None

    def _calculate_point_on_vector(self, destination: tuple) -> tuple:
        """
        Get the new point where to travel.

        Parameters
        ----------
        destination : tuple
            The destinaton to go to.

        Returns
        -------
        tuple
            New position where to travel.
        """
        v = np.array(self.position, dtype=float)
        u = np.array(destination, dtype=float)
        n = v - u
        n_magnitude = np.linalg.norm(n, 2)
        if n_magnitude == 0:
            n = 0
        else:
            n /= n_magnitude

        # Make sure we don't overstep the destination
        distance = min(n_magnitude, self.step_size)
        new_position = v - distance * n

        return new_position

    @abstractmethod
    def _get_new_position(self) -> tuple:
        """
        Calculate the new agent position.

        Returns
        -------
        tuple
            New position.
        """
        pass

    def update_state(self) -> None:
        """
        Save the position of the friend and the enemy.
        """
        self.friend_position = self.friend.position
        self.enemy_position = self.enemy.position

    def move(self) -> None:
        """
        The agent moves.
        """
        self.position = self._get_new_position()


class ProtectiveAgent1(Agent):
    """
    An agent who tries to put itself in the midpoint between its friend
    and its enemy.

    Parameters
    ----------
    x : float
        Starting horizontal position.
    y : float
        Starting vertical position.
    step_size : float
        The amount of travel performed each turn.
    """

    def __init__(self, x: float, y: float, step_size: float) -> None:
        super().__init__(x, y, step_size)

    def _get_destination(self) -> tuple:
        """
        Calculates the midpoint of a straight line between 
        the friend and the enemy.

        Returns
        -------
        tuple(float, float)
            Position of the midpoint.
        """
        midpoint_x = (self.friend_position[0] + self.enemy_position[0]) / 2
        midpoint_y = (self.friend_position[1] + self.enemy_position[1]) / 2
        return (midpoint_x, midpoint_y)

    def _get_new_position(self) -> tuple:
        """
        Calculates the new position in order for the agent to put inself
        in the middle between the friend and the enemy.

        Returns
        -------
        tuple
            New position.
        """
        destination = self._get_destination()
        new_position = self._calculate_point_on_vector(destination)

        return (new_position[0], new_position[1])


class ProtectiveAgent2(ProtectiveAgent1):
    """
    An agent who tries to put itself between its friend and its enemy by
    taking the shortest path.

    Parameters
    ----------
    x : float
        Starting horizontal position.
    y : float
        Starting vertical position.
    step_size : float
        The amount of travel performed each turn.
    """

    def __init__(self, x: float, y: float, step_size: float) -> None:
        super().__init__(x, y, step_size)

    def _get_destination(self) -> tuple:
        """
        Gets the destination by taking the shortest path to be on the
        line between the friend and the enemy.

        Returns
        -------
        tuple
            The destination where to go.
        """
        # Represent points as vectors
        f = np.array(self.friend_position, dtype=float)
        e = np.array(self.enemy_position, dtype=float)
        p = np.array(self.position, dtype=float)

        # Calculate distance between all points
        d_fe = np.linalg.norm(f-e, 2)
        d_fp = np.linalg.norm(f-p, 2)
        d_ep = np.linalg.norm(e-p, 2)

        # Check if obtuse
        # Behind friend
        if d_ep**2 > d_fe**2 + d_fp**2:
            point = f
        # Behind enemy
        elif d_fp**2 > d_fe**2 + d_ep**2:
            point = e
        # Between friend and enemy
        else:
            # TODO: must be a much neater way of doing all this...
            # First check if the friend and enemy form a vertical line
            if e[0]-f[0] == 0:
                # If so we now that the intercept will happen at their x location
                # and at self's y location since we already have checked that
                # self is not outside the line
                x = e[0]
                y = self.position[1]
            # The same case with horizontal lines
            elif e[1]-f[1] == 0:
                x = self.position[1]
                y = e[1]
            else:
                # Calculate slope of the two lines (between enemy and friend
                # and the perpendicular line passing through self position)
                m1 = (e[1]-f[1])/(e[0]-f[0])
                m2 = -1/m1

                # Calculate intercept
                b1 = e[1] - m1*e[0]
                b2 = p[1] - m2*p[0]

                # Yields intersection point in line
                x = (b1-b2)/(m2-m1)
                y = m1*x + b1

            point = (x, y)

        return (point[0] ,point[1])


class HidingAgent1(ProtectiveAgent1):
    """
    An agent which tries to hide behind its friend by taking moving to a 
    set distance behind its friend.

    Parameters
    ----------
    x : float
        Starting horizontal position.
    y : float
        Starting vertical position.
    step_size : float
        The amount of travel performed each turn.
    distance : float
        The distance away from the friend to travel.
    """

    def __init__(self, x: float, y: float, step_size: float, distance: float) -> None:
        super().__init__(x, y, step_size)
        self.distance = distance

    def _get_destination(self) -> tuple:
        """
        Gets the destination by to a set distance behind its friend.

        Returns
        -------
        tuple
            The destination where to go.
        """
        # Represent points as vectors
        f = np.array(self.friend_position, dtype=float)
        e = np.array(self.enemy_position, dtype=float)

        # TODO: very similar to ProtectiveAgent1, generalize this
        v = f - e
        u = v / np.linalg.norm(v, 2)

        # Check in which direction we should move
        point = f + self.distance*u
        d_fe = np.linalg.norm(f-e, 2)
        d_ep = np.linalg.norm(e-point, 2)
        
        # The distance between the new point and the enemy should be
        # greater than the distance between the friend and the enemy
        if d_fe > d_ep:
            point = f - self.distance*u

        return (point[0] ,point[1])


class HidingAgent2(ProtectiveAgent1):
    """
    An agent which tries to hide behind its friend by taking the shortest
    path to safety.

    Parameters
    ----------
    x : float
        Starting horizontal position.
    y : float
        Starting vertical position.
    step_size : float
        The amount of travel performed each turn.
    """

    def __init__(self, x: float, y: float, step_size: float) -> None:
        super().__init__(x, y, step_size)

    def _get_destination(self) -> tuple:
        """
        Gets the destination by taking the shortest path to be behind the
        friend on the line through the friend and the enemy.

        Returns
        -------
        tuple
            The destination where to go.
        """
        # Represent points as vectors
        f = np.array(self.friend_position, dtype=float)
        e = np.array(self.enemy_position, dtype=float)
        p = np.array(self.position, dtype=float)

        # Calculate distance between all points
        d_fe = np.linalg.norm(f-e, 2)
        d_fp = np.linalg.norm(f-p, 2)
        d_ep = np.linalg.norm(e-p, 2)

        # TODO: very similar to ProtectiveAgent2, generalize this
        # Check if obtuse
        # Behind friend
        if d_ep**2 > d_fe**2 + d_fp**2:
            # TODO: must be a much neater way of doing all this...
            # First check if the friend and enemy form a vertical line
            if e[0]-f[0] == 0:
                # If so we now that the intercept will happen at their x location
                # and at self's y location since we already have checked that
                # self is not outside the line
                x = e[0]
                y = self.position[1]
            # The same case with horizontal lines
            elif e[1]-f[1] == 0:
                x = self.position[1]
                y = e[1]
            else:
                # Calculate slope of the two lines (between enemy and friend
                # and the perpendicular line passing through self position)
                m1 = (e[1]-f[1])/(e[0]-f[0])
                m2 = -1/m1

                # Calculate intercept
                b1 = e[1] - m1*e[0]
                b2 = p[1] - m2*p[0]

                # Yields intersection point in line
                x = (b1-b2)/(m2-m1)
                y = m1*x + b1

            point = (x, y)
        # Behind enemy
        elif d_fp**2 > d_fe**2 + d_ep**2:
            point = f
        # Between friend and enemy
        else:
            point = f

        return (point[0] ,point[1])