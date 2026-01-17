from abc import ABC, abstractmethod


class SocialScraper(ABC):
    """
    Abstract Base Class (Interface) for all social media scrapers.
    Follows the Dependency Inversion Principle.
    """

    @abstractmethod
    def get_payload(self, target: str) -> dict:
        """
        Scrapes the target (handle, URL, or ID) and returns a
        standardized 'Identity Payload' dictionary.

        :param target: The string identifier for the channel/user/community.
        :return: A dictionary containing the structured community data.
        """
        pass