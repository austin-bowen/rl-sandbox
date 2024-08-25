from rlsandbox.walker.utils import Wrapper


class OptimizerWrapper(Wrapper):
    @property
    def lr(self) -> float:
        return self.param_groups[0]['lr']

    @lr.setter
    def lr(self, value: float) -> None:
        for param_group in self.param_groups:
            param_group['lr'] = value
