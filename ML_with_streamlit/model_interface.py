from abc import ABC, abstractmethod
from matplotlib.figure import Figure


class Model(ABC):

    @abstractmethod
    def get_available_datasets(self) -> tuple:
        """Get the available datasets for each type of machine learning algorithm."""
    
    @abstractmethod
    def get_available_models(self) -> list:
        """Get the available models for the chosen type of machine learning algorithm."""

    @abstractmethod
    def set_dataset(self, dataset_name: str) -> None:
        """Set the dataset and store its important data."""

    @abstractmethod
    def set_model(self, model_name: str) -> None:
        """Set the model name."""

    @abstractmethod
    def get_sidebar(self) -> dict:
        """Get sidebar widgets to get the model parameters."""
    
    @abstractmethod
    def set_params(self, params: dict) -> None:
        """Set the model parameters."""
    
    @abstractmethod
    def get_model_info(self) -> str:        
        """Run the model and return its results."""

    @abstractmethod
    def get_dataset_info(self) -> tuple:
        """Get information about the dataset."""
    
    @abstractmethod
    def get_figure(self) -> Figure:
        """"""