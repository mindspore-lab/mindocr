import dataclasses
from typing import List, Tuple


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    def __init__(
        self,
        system: str = None,
        roles: Tuple[str, str] = ("user", "assistant"),
        messages: List[Tuple[str, str]] = None,
        sep: str = "<|im_end|>",
    ):
        self.system = (
            "<|im_start|>{system}\n{message}{sep}\n".format(
                system="system",
                message="You should follow the instructions carefully and explain your answers in detail.",
                sep=sep,
            )
            if system is None
            else system
        )
        self.roles = roles
        self.messages = list() if messages is None else messages
        self.sep = sep

    def get_messages(self):
        return self.messages

    def get_prompt(self):
        ret = self.system if self.system else ""
        for role, message in self.messages:
            ret += "<|im_start|>{role}\n{message}{sep}\n".format(role=role, message=message, sep=self.sep)
        return ret

    def add_message(self, role, message):
        if role not in self.roles:
            raise ValueError("role must be in {}.".format(self.roles))
        self.messages.append((role, message))
