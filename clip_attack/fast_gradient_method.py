import eagerpy as ep
from foolbox.attacks.gradient_descent_base import L1BaseGradientDescent
from foolbox.attacks.gradient_descent_base import L2BaseGradientDescent
from foolbox.attacks.gradient_descent_base import LinfBaseGradientDescent
from foolbox.models.base import Model
from foolbox.criteria import Misclassification, TargetedMisclassification
from foolbox.attacks.base import T
from typing import Callable, Union, Any
import torch

class L1FastGradientAttack(L1BaseGradientDescent):
    """Fast Gradient Method (FGM) using the L1 norm

    Args:
        random_start : Controls whether to randomly start within allowed epsilon ball.
    """

    def __init__(self, *, random_start: bool = False):
        super().__init__(
            rel_stepsize=1.0,
            steps=1,
            random_start=random_start,
        )

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Misclassification, TargetedMisclassification, T],
        *,
        epsilon: float,
        **kwargs: Any,
    ) -> T:
        if hasattr(criterion, "target_classes"):
            raise ValueError("unsupported criterion")

        return super().run(
            model=model, inputs=inputs, criterion=criterion, epsilon=epsilon, **kwargs
        )


class L2FastGradientAttack(L2BaseGradientDescent):
    """Fast Gradient Method (FGM)

    Args:
        random_start : Controls whether to randomly start within allowed epsilon ball.
    """

    def __init__(self, *, random_start: bool = False):
        super().__init__(
            rel_stepsize=1.0,
            steps=1,
            random_start=random_start,
        )

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Misclassification, TargetedMisclassification, T],
        *,
        epsilon: float,
        **kwargs: Any,
    ) -> T:
        if hasattr(criterion, "target_classes"):
            raise ValueError("unsupported criterion")

        return super().run(
            model=model, inputs=inputs, criterion=criterion, epsilon=epsilon, **kwargs
        )


class LinfFastGradientAttack(LinfBaseGradientDescent):
    """Fast Gradient Sign Method (FGSM)

    Args:
        random_start : Controls whether to randomly start within allowed epsilon ball.
    """

    def __init__(self, *, random_start: bool = False):
        super().__init__(
            rel_stepsize=1.0,
            steps=1,
            random_start=random_start,
        )

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Misclassification, TargetedMisclassification, T],
        *,
        epsilon: float,
        **kwargs: Any,
    ) -> T:
        if hasattr(criterion, "target_classes"):
            raise ValueError("unsupported criterion")

        return super().run(
            model=model, inputs=inputs, criterion=criterion, epsilon=epsilon, **kwargs
        )

    def get_loss_fn(self, model: Model, labels: ep.Tensor) -> Callable[[ep.Tensor], ep.Tensor]:
        def loss_fn(inputs: ep.Tensor) -> ep.Tensor:
            logits = model(inputs)
            # final = logits[torch.arange(labels.shape[0]), labels]
            return ep.crossentropy(logits, labels).mean()
        return loss_fn