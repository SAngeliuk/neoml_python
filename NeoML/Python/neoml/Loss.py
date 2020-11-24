""" Copyright © 2017-2020 ABBYY Production LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
--------------------------------------------------------------------------------------------------------------
"""

import neoml.PythonWrapper as PythonWrapper
import neoml.Dnn as Dnn
import neoml.Utils as Utils


class Loss(Dnn.Layer):
    """
    """
    def __init__(self, internal):
        if not isinstance(internal, PythonWrapper.LossLayer):
            raise ValueError('The `internal` must be PythonWrapper.LossLayer')

        super().__init__(internal)

    @property
    def last_loss(self):
        """
        """
        return self._internal.get_last_loss()

    @property
    def loss_weight(self):
        """
        """
        return self._internal.get_loss_weight()

    @loss_weight.setter
    def loss_weight(self, weight):
        """
        """
        self._internal.set_loss_weight(weight)

    @property
    def train_labels(self):
        """
        """
        return self._internal.get_train_labels()

    @train_labels.setter
    def train_labels(self, train):
        """
        """
        self._internal.set_train_labels(train)

    @property
    def max_gradient(self):
        """
        """
        return self._internal.get_max_gradient()

    @max_gradient.setter
    def max_gradient(self, max_value):
        """
        """
        self._internal.set_max_gradient(max_value)

# ----------------------------------------------------------------------------------------------------------------------


class CrossEntropyLoss(Loss):
    """Implements a layer that calculates the loss value as cross-entropy between the result and the standard
    """
    def __init__(self, input_layers, softmax=True, loss_weight=1.0, name=None):

        if type(input_layers) is PythonWrapper.CrossEntropyLoss:
            super().__init__(input_layers)
            return

        layers, outputs = Utils.check_input_layers(input_layers, (2, 3))

        internal = PythonWrapper.CrossEntropyLoss(str(name), layers, outputs, bool(softmax), float(loss_weight))
        super().__init__(internal)

    @property
    def apply_softmax(self):
        """
        """
        return self._internal.get_apply_softmax()

    @apply_softmax.setter
    def apply_softmax(self, value):
        """
        """
        self._internal.set_apply_softmax(int(value))

# ----------------------------------------------------------------------------------------------------------------------


class BinaryCrossEntropyLoss(Loss):
    """
    """
    def __init__(self, input_layers, name=None):

        if type(input_layers) is PythonWrapper.BinaryCrossEntropyLoss:
            super().__init__(input_layers)
            return

        layers, outputs = Utils.check_input_layers(input_layers, (2, 3))

        internal = PythonWrapper.BinaryCrossEntropyLoss(str(name), layers, outputs)
        super().__init__(internal)

    @property
    def positive_weight(self):
        """
        """
        return self._internal.get_positive_weight()

    @positive_weight.setter
    def positive_weight(self, weight):
        """
        """
        self._internal.set_positive_weight(weight)
