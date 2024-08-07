

class Config():
    def __init__(self) -> None:
        # TODO: read configs from yaml
        
        self._template_path = "./templates"

        pass


    @property
    def template_path(self) -> str:
        return self._template_path

    @template_path.setter
    def template_path(self, new_path:str):
        # TODO
        pass