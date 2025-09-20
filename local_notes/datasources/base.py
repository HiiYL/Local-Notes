from abc import ABC, abstractmethod
from typing import List

from ..models import Document


class DataSource(ABC):
    @abstractmethod
    def fetch(self) -> List[Document]:
        """Fetch documents from the source."""
        raise NotImplementedError
