from abc import ABC


class SimpleClient(ABC):
    from abc import abstractmethod

    """
    Abstract class for interacting with backend data storage system.
    Eg., BigTableClient for using big table as storage.
    """

    @abstractmethod
    def create_table(self) -> None:
        """Initialize the table and store associated meta."""

    @abstractmethod
    def write_metadata(self, metadata):
        """Update stored metadata."""

    @abstractmethod
    def read_metadata(self):
        """Read stored metadata."""

    @abstractmethod
    def read_entries(
        self,
        entry_ids,
        properties=None,
        start_time=None,
        end_time=None,
        end_time_inclusive=False,
    ):
        """Read entries and their properties."""

    @abstractmethod
    def read_entry(
        self,
        entry_id,
        properties=None,
        start_time=None,
        end_time=None,
        end_time_inclusive=False,
    ):
        """Read a single entry and it's properties."""

    @abstractmethod
    def write_entries(self, entries):
        """Writes/updates entries (IDs along with properties)."""
