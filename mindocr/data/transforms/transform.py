from typing import Any, Dict, Tuple

import numpy as np


class Transform:
    """Transform the input data into the output data.

    Inputs:
        data: Data tuples need to be transformed

    Outputs:
        result: Transformed data tuples

    Note:
        This is an abstract class, child class must implement `transform` method.
    """

    def __init__(self) -> None:
        self._updated_columns = self.get_updated_columns()

    @property
    def updated_columns(self):
        return self._updated_columns
    
    @updated_columns.setter
    def updated_columns(self, columns):
        self._updated_columns = columns

    def get_updated_columns(self):
        """Set the names of the updated columns in the stage"""
        raise NotImplementedError("Child class must implement this method.")
    
    def transform(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Transform the state into the transformed state. state is a dictionay
        storing the informaton of the image and labels, the returned states is
        the updated dictionary storing the updated image and labels.

        Args:
            state: Stored information of image and labels

        Returns:
            Updated inforamtion of image and labels based on the transformation
        """
        raise NotImplementedError("Child class must implement this method.")

    def __call__(self, *args: Any) -> Tuple[np.ndarray, ...]:
        """This simply does the following process
        1. Pack the column names and data tuples into a dictionary
        2. Calling the tranform method on the dictionary
        3. Unpack the dictionay and return the data tuples only"""
        # pack the arguments
        states = dict(zip(self._updated_columns, args))
        transformed_states = self.transform(states)
        states.update(transformed_states)

        # unpack the argument for mindspore dataset API
        final_states = {k: np.asarray(states[k]) for k in self._updated_columns}
        tuples = tuple(final_states.values())
        return tuples
