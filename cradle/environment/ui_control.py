"""Base class for UI Control."""
import abc

class UIControl(abc.ABC):
    """Interface for UI control."""

    @abc.abstractmethod
    def pause_game(self, env_name: str, ide_name: str) -> None:
        """
        Pause the game.
        Args:
            env_name: Environment name.
            ide_name: IDE name.
        Returns:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def unpause_game(self, env_name: str, ide_name: str) -> None:
        """
        Args:
            env_name: Environment name.
            ide_name: IDE name.
        Returns:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def switch_to_game(self, env_name: str, ide_name: str) -> None:
        """
        Switch to game.
        Args:
            env_name: Environment name.
            ide_name: IDE name.
        Returns:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def exit_back_to_pause(self, env_name: str, ide_name: str) -> None:
        """
        Exit back to pause.
        Args:
            env_name: Environment name.
            ide_name: IDE name.
        Returns:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def exit_back_to_game(self, env_name: str, ide_name: str) -> None:
        """
        Exit back to game.
        Args:
            env_name: Environment name.
            ide_name: IDE name.
        Returns:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def take_screenshot(self, tid : float, screen_region : tuple[int, int, int, int] = None) -> str:
        """
        Take a screenshot.
        Args:
            tid: Task ID.
            screen_region: Screen region.
        Returns:
            Screenshot path.
        """
        raise NotImplementedError