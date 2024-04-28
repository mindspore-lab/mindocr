from enum import Enum


class TaskType(Enum):
    DET = 0  # Detection Model
    CLS = 1  # Classification Model
    REC = 2  # Recognition Model
    DET_REC = 3  # Detection And Detection Model
    DET_CLS_REC = 4  # Detection, Classification and Recognition Model
    LAYOUT = 5  # Layout Model
    LAYOUT_DET = 6
    LAYOUT_DET_REC = 7
    LAYOUT_DET_CLS_REC = 8


SUPPORTED_TASK_BASIC_MODULE = {
    TaskType.DET: [TaskType.DET],
    TaskType.CLS: [TaskType.CLS],
    TaskType.REC: [TaskType.REC],
    TaskType.DET_REC: [TaskType.DET_REC],
    TaskType.DET_CLS_REC: [TaskType.DET_CLS_REC],
    TaskType.LAYOUT: [TaskType.LAYOUT],
    TaskType.LAYOUT_DET: [TaskType.LAYOUT_DET],
    TaskType.LAYOUT_DET_REC: [TaskType.LAYOUT_DET_REC],
    TaskType.LAYOUT_DET_CLS_REC: [TaskType.LAYOUT_DET_CLS_REC],
}
