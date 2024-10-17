import sys

if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal

from functools import partial
from typing import Callable, Dict, Optional, Sequence, Union

from reservoirpy.activationsfunc import get_function, identity, tanh
from reservoirpy.mat_gen import bernoulli, normal
from reservoirpy.node import Node
from reservoirpy.type import Weights
from reservoirpy.utils.random import noise, rand_generator
from reservoirpy.utils.validation import is_array
from reservoirpy.nodes.reservoirs.base import forward_external, forward_internal, initialize, initialize_feedback

import sys

if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal

from functools import partial
from typing import Callable, Dict, Optional, Sequence, Union
import numpy as np
from NeuronModel import NeuronModel

class CustomReservoir(Node):
    def __init__(
        self,
        units: int = None,
        neuron_model: NeuronModel = None,  # 任意のニューロンモデル
        lr: float = 1.0,
        sr: Optional[float] = None,
        input_bias: bool = True,
        noise_rc: float = 0.0,
        noise_in: float = 0.0,
        noise_fb: float = 0.0,
        noise_type: str = "normal",
        noise_kwargs: Dict = None,
        input_scaling: Union[float, Sequence] = 1.0,
        bias_scaling: float = 1.0,
        fb_scaling: Union[float, Sequence] = 1.0,
        input_connectivity: float = 0.1,
        rc_connectivity: float = 0.1,
        fb_connectivity: float = 0.1,
        Win: Union[Weights, Callable] = bernoulli,
        W: Union[Weights, Callable] = normal,
        Wfb: Union[Weights, Callable] = bernoulli,
        bias: Union[Weights, Callable] = bernoulli,
        fb_activation: Union[str, Callable] = identity,
        activation: Union[str, Callable] = tanh,
        equation: Literal["internal", "external"] = "internal",
        forward_fn: Optional[Callable] = None,  # Allowing custom forward functions
        input_dim: Optional[int] = None,
        feedback_dim: Optional[int] = None,
        seed=None,
        **kwargs,
    ):
        if units is None and not is_array(W):
            raise ValueError(
                "'units' parameter must not be None if 'W' parameter is not "
                "a matrix."
            )

        if neuron_model is None:
            raise ValueError("A neuron model must be provided.")

        # If a custom forward function is provided, use it; otherwise, default to internal/external logic
        if forward_fn is not None:
            forward = forward_fn
        elif neuron_model is not None:
            forward = neuron_model.step
        else:
            if equation == "internal":
                forward = forward_internal
            elif equation == "external":
                forward = forward_external
            else:
                raise ValueError(
                    "'equation' parameter must be either 'internal' or 'external'."
                )

        if type(activation) is str:
            activation = get_function(activation)
        if type(fb_activation) is str:
            fb_activation = get_function(fb_activation)

        rng = rand_generator(seed)

        noise_kwargs = dict() if noise_kwargs is None else noise_kwargs

        # カスタムパラメータの初期化
        self.neuron_model = neuron_model
        self.membrane_potentials = np.zeros((units, 1))  # 各ニューロンの膜電位を0で初期化

        super(CustomReservoir, self).__init__(
            fb_initializer=partial(
                initialize_feedback,
                Wfb_init=Wfb,
                fb_scaling=fb_scaling,
                fb_connectivity=fb_connectivity,
                seed=seed,
            ),
            params={
                "W": None,
                "Win": None,
                "Wfb": None,
                "bias": None,
                "internal_state": None,
                "neuron_model": neuron_model,  # カスタムパラメータとして保持
            },
            hypers={
                "lr": lr,
                "sr": sr,
                "input_scaling": input_scaling,
                "bias_scaling": bias_scaling,
                "fb_scaling": fb_scaling,
                "rc_connectivity": rc_connectivity,
                "input_connectivity": input_connectivity,
                "fb_connectivity": fb_connectivity,
                "noise_in": noise_in,
                "noise_rc": noise_rc,
                "noise_out": noise_fb,
                "noise_type": noise_type,
                "activation": activation,
                "fb_activation": fb_activation,
                "units": units,
                "noise_generator": partial(noise, rng=rng, **noise_kwargs),
            },
            forward=forward,
            initializer=partial(
                initialize,
                sr=sr,
                input_scaling=input_scaling,
                bias_scaling=bias_scaling,
                input_connectivity=input_connectivity,
                rc_connectivity=rc_connectivity,
                W_init=W,
                Win_init=Win,
                bias_init=bias,
                input_bias=input_bias,
                seed=seed,
            ),
            output_dim=units,
            feedback_dim=feedback_dim,
            input_dim=input_dim,
            **kwargs,
        )
