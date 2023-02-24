import streamlit as st
from matplotlib.figure import Figure

from model_interface import Model


class PageInterface:
    """Class for the interface of the web app."""
    def __init__(self) -> None:
        self.available_model_types = ("Classification",)
        self.model = None

    @staticmethod
    def get_model_object(model_type: str) -> Model:
        """Import the model class according to the given model type."""
        mod = __import__(model_type.lower(), fromlist=[model_type])
        cls = getattr(mod, model_type)
        return cls

    def set_model(self, model_type: str) -> None:
        """Set the model object."""
        self.model = self.get_model_object(model_type)()

    @staticmethod
    def write_intro_section() -> None:
        """Write the introduction section in the web app interface."""
        st.title("Your Machine Learning Platform")

        video_url = "https://www.youtube.com/watch?v=Klqn--Mu2pE"
        st.write("- This project started with the tutorial on [this video](%s)" % video_url)

        st.markdown("""
        - Next steps:
            - ~~Code refactoring~~
            - Review the docstrings
            - New dataset option: create your own with a set of parameters
            - More options to choose what to see in the plot (not necessarily PCA)
            - More information about the algorithm
            - More parameters to set
            - Other types of algorithms (clustering, regression)
            - Section of resources to learn about the algorithm
        """)

    @staticmethod
    def write_model_section(model_info: str, dataset_info: str, fig: Figure) -> None:
        """Write the model section in the web app interface."""
        st.subheader("Algorithm Info")
        st.markdown(model_info)
        st.pyplot(fig)
        st.subheader("Dataset Info")
        st.markdown(dataset_info)

    def build_general_sidebar(self) -> tuple:
        """Add general widgets to the sidebar, for selection of model type, dataset and model."""
        model_type = st.sidebar.selectbox("Select Model Type", self.available_model_types)

        self.set_model(model_type)

        dataset_name = st.sidebar.selectbox("Select Dataset", self.model.get_available_datasets())
        model_name = st.sidebar.selectbox("Select Model", self.model.get_available_models())

        return dataset_name, model_name

    def build_model_sidebar(self, model_sidebar: dict) -> dict:
        """Add model's specific widgets to the sidebar, for selection of parameters."""
        params = {}
        for widget, configs in model_sidebar.items():
            for config in configs:
                params[config["label"]] = getattr(st.sidebar, widget)(**config)

        return params

    def build_interface(self):
        """Build the whole web app interface."""
        self.write_intro_section()

        dataset_name, model_name = self.build_general_sidebar()
        self.model.set_dataset(dataset_name)
        self.model.set_model(model_name)

        model_sidebar = self.model.get_sidebar()
        params = self.build_model_sidebar(model_sidebar)
        self.model.set_params(params)

        model_info = self.model.get_model_info()
        fig = self.model.get_figure()

        dataset_info = self.model.get_dataset_info()

        self.write_model_section(model_info, dataset_info, fig)


if __name__ == "__main__":
    page = PageInterface()
    page.build_interface()
