from dataclasses import dataclass, field
from enum import Enum


class InferModelComb(Enum):
    DET = 0  # Detection Model
    REC = 1  # Recognition Model
    CLS = 2  # Classifier Model
    DET_REC = 3  # Detection, Classifier And Detection Model
    DET_CLS_REC = 4  # Detection, Recognition Model


class ShapeType(Enum):
    STATIC_SHAPE = 0
    DYNAMIC_SHAPE = 1
    DYNAMIC_BATCHSIZE = 2
    DYNAMIC_IMAGESIZE = 3


SupportedTaskOrder = {
    InferModelComb.DET: [InferModelComb.DET],
    InferModelComb.REC: [InferModelComb.REC],
    InferModelComb.DET_REC: [InferModelComb.DET, InferModelComb.REC],
    InferModelComb.DET_CLS_REC: [InferModelComb.DET, InferModelComb.CLS, InferModelComb.REC],
}


class ConnectType(Enum):
    MODULE_CONNECT_ONE = 0
    MODULE_CONNECT_CHANNEL = 1
    MODULE_CONNECT_PAIR = 2
    MODULE_CONNECT_RANDOM = 3


@dataclass
class ModuleOutputInfo:
    module_name: str
    connect_type: ConnectType
    output_queue_list_size: int
    output_queue_list: list = field(default_factory=lambda: [])


@dataclass
class ModuleInitArgs:
    pipeline_name: str
    module_name: str
    instance_id: -1


@dataclass
class ModuleDesc:
    module_name: str
    module_count: int


@dataclass
class ModuleConnectDesc:
    module_send_name: str
    module_recv_name: str
    connect_type: ConnectType = field(default_factory=lambda: ConnectType.MODULE_CONNECT_RANDOM)


@dataclass
class ModulesInfo:
    module_list: list = field(default_factory=lambda: [])
    input_queue_list: list = field(default_factory=lambda: [])
