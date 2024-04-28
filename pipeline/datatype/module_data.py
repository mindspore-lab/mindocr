from dataclasses import dataclass, field
from enum import Enum


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
    module_type: str
    module_name: str
    instance_id: -1


@dataclass
class ModuleDesc:
    module_type: str  # 节点类型，如HandoutNode
    module_name: str  # 节点名，如1，该节点唯一标识为 {module_type}{model_name}
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
