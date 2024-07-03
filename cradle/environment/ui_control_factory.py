import importlib

from cradle.utils import Singleton


class UIControlFactory(metaclass=Singleton):

    def __init__(self):
        self._builders = {}


    def register_builder(self, key, builder_str):

        try:

            # Split the module and class name
            module_name, class_name = builder_str.rsplit('.', 1)

            # Dynamically import the module
            module = importlib.import_module(module_name)

            # Get the class from the module
            builder_class = getattr(module, class_name)

            self._builders[key] = builder_class

        except (ImportError, AttributeError) as e:
            raise ValueError(f"Class '{class_name}' not found in module '{module_name}'") from e


    # A SkillRegistry takes a skill_config and an embedding provider
    def create(self, key, **kwargs):

        builder = self._builders.get(key)

        if not builder:
            raise ValueError(key)

        return builder(**kwargs)
