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


class EltwiseSum(Dnn.Layer):
    """
    """
    def __init__(self, input_layers, name=None):

        if type(input_layers) is PythonWrapper.EltwiseSum:
            super().__init__(input_layers)
            return

        layers, outputs = Utils.check_input_layers(input_layers, 0)

        internal = PythonWrapper.EltwiseSum(str(name), layers, outputs)
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class EltwiseMul(Dnn.Layer):
    """
    """
    def __init__(self, input_layers, name=None):

        if type(input_layers) is PythonWrapper.EltwiseMul:
            super().__init__(input_layers)
            return

        layers, outputs = Utils.check_input_layers(input_layers, 0)

        internal = PythonWrapper.EltwiseMul(str(name), layers, outputs)
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class EltwiseNegMul(Dnn.Layer):
    """
    """
    def __init__(self, input_layers, name=None):

        if type(input_layers) is PythonWrapper.EltwiseNegMul:
            super().__init__(input_layers)
            return

        layers, outputs = Utils.check_input_layers(input_layers, 0)

        internal = PythonWrapper.EltwiseNegMul(str(name), layers, outputs)
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class EltwiseMax(Dnn.Layer):
    """
    """
    def __init__(self, input_layers, name=None):

        if type(input_layers) is PythonWrapper.EltwiseMax:
            super().__init__(input_layers)
            return

        layers, outputs = Utils.check_input_layers(input_layers, 0)

        internal = PythonWrapper.EltwiseMax(str(name), layers, outputs)
        super().__init__(internal)
