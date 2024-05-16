from .domain_adaptation import DomainAdaptationExperiment
from ..models._ot import _OT

from typing import Dict, Optional, List, Tuple, Union
import torch.nn as nn


class AblationStudy(DomainAdaptationExperiment):
    def __init__(
        self,
        model: Dict[str, _OT],
        exp_name: str,
        log_dir: str,
        classifier: Optional[nn.Module] = None,
        **kwargs
    ):
        super().__init__(model, exp_name, log_dir)