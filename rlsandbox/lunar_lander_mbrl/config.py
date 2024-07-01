from argparse import Namespace


class Config(Namespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __iter__(self):
        return iter(self._get_kwargs())
