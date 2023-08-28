from dataclasses import dataclass


@dataclass
class StopSign:
    stop: bool = True


@dataclass
class ProfilingData:
    module_name: str = ""
    instance_id: int = ""
    device_id: int = 0
    process_cost_time: float = 0.0
    send_cost_time: float = 0.0
    image_total: int = -1
