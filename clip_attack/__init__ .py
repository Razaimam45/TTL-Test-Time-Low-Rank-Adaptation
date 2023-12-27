import foolbox 
from foolbox.attacks.base import Attack  # noqa: F401

# FixedEpsilonAttack subclasses
from foolbox.attacks.contrast import L2ContrastReductionAttack  # noqa: F401
from foolbox.attacks.virtual_adversarial_attack import VirtualAdversarialAttack  # noqa: F401
from foolbox.attacks.ddn import DDNAttack  # noqa: F401
from .projected_gradient_descent import (  # noqa: F401
    L1ProjectedGradientDescentAttack,
    L2ProjectedGradientDescentAttack,
    LinfProjectedGradientDescentAttack,
    L1AdamProjectedGradientDescentAttack,
    L2AdamProjectedGradientDescentAttack,
    LinfAdamProjectedGradientDescentAttack,
)

from foolbox.attacks.basic_iterative_method import (  # noqa: F401
    L1BasicIterativeAttack,
    L2BasicIterativeAttack,
    LinfBasicIterativeAttack,
    L1AdamBasicIterativeAttack,
    L2AdamBasicIterativeAttack,
    LinfAdamBasicIterativeAttack,
)
from .fast_gradient_method import (  # noqa: F401
    L1FastGradientAttack,
    L2FastGradientAttack,
    LinfFastGradientAttack,
)

from foolbox.attacks.additive_noise import (  # noqa: F401
    L2AdditiveGaussianNoiseAttack,
    L2AdditiveUniformNoiseAttack,
    L2ClippingAwareAdditiveGaussianNoiseAttack,
    L2ClippingAwareAdditiveUniformNoiseAttack,
    LinfAdditiveUniformNoiseAttack,
    L2RepeatedAdditiveGaussianNoiseAttack,
    L2RepeatedAdditiveUniformNoiseAttack,
    L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack,
    L2ClippingAwareRepeatedAdditiveUniformNoiseAttack,
    LinfRepeatedAdditiveUniformNoiseAttack,
)
from foolbox.attacks.sparse_l1_descent_attack import SparseL1DescentAttack  # noqa: F401

# MinimizatonAttack subclasses
from foolbox.attacks.inversion import InversionAttack  # noqa: F401
from foolbox.attacks.contrast_min import (  # noqa: F401
    BinarySearchContrastReductionAttack,
    LinearSearchContrastReductionAttack,
)
from foolbox.attacks.carlini_wagner import L2CarliniWagnerAttack  # noqa: F401
from foolbox.attacks.newtonfool import NewtonFoolAttack  # noqa: F401
from foolbox.attacks.ead import EADAttack  # noqa: F401
from foolbox.attacks.blur import GaussianBlurAttack  # noqa: F401
from foolbox.attacks.spatial_attack import SpatialAttack  # noqa: F401
from foolbox.attacks.deepfool import L2DeepFoolAttack, LinfDeepFoolAttack  # noqa: F401
from foolbox.attacks.saltandpepper import SaltAndPepperNoiseAttack  # noqa: F401
from foolbox.attacks.blended_noise import LinearSearchBlendedUniformNoiseAttack  # noqa: F401
from foolbox.attacks.binarization import BinarizationRefinementAttack  # noqa: F401
from foolbox.attacks.dataset_attack import DatasetAttack  # noqa: F401
from foolbox.attacks.boundary_attack import BoundaryAttack  # noqa: F401
from foolbox.attacks.hop_skip_jump import HopSkipJumpAttack  # noqa: F401
from foolbox.attacks.brendel_bethge import (  # noqa: F401
    L0BrendelBethgeAttack,
    L1BrendelBethgeAttack,
    L2BrendelBethgeAttack,
    LinfinityBrendelBethgeAttack,
)
from foolbox.attacks.fast_minimum_norm import (  # noqa: F401
    L0FMNAttack,
    L1FMNAttack,
    L2FMNAttack,
    LInfFMNAttack,
)
from foolbox.attacks.gen_attack import GenAttack  # noqa: F401
from foolbox.attacks.pointwise import PointwiseAttack  # noqa: F401

# from .blended_noise import LinearSearchBlendedUniformNoiseAttack  # noqa: F401
# from .brendel_bethge import (  # noqa: F401
#     L0BrendelBethgeAttack,
#     L1BrendelBethgeAttack,
#     L2BrendelBethgeAttack,
#     LinfinityBrendelBethgeAttack,
# )
# from .additive_noise import L2AdditiveGaussianNoiseAttack  # noqa: F401
# from .additive_noise import L2AdditiveUniformNoiseAttack  # noqa: F401
# from .additive_noise import LinfAdditiveUniformNoiseAttack  # noqa: F401
# from .additive_noise import L2RepeatedAdditiveGaussianNoiseAttack  # noqa: F401
# from .additive_noise import L2RepeatedAdditiveUniformNoiseAttack  # noqa: F401
# from .additive_noise import LinfRepeatedAdditiveUniformNoiseAttack  # noqa: F401
# from .saltandpepper import SaltAndPepperNoiseAttack  # noqa: F401

FGM = L2FastGradientAttack
FGSM_clip = LinfFastGradientAttack
L1PGD = L1ProjectedGradientDescentAttack
L2PGD = L2ProjectedGradientDescentAttack
LinfPGD = LinfProjectedGradientDescentAttack
PGD_clip = LinfPGD

L1AdamPGD = L1AdamProjectedGradientDescentAttack
L2AdamPGD = L2AdamProjectedGradientDescentAttack
LinfAdamPGD = LinfAdamProjectedGradientDescentAttack
AdamPGD = LinfAdamPGD
