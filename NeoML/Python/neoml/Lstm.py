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


class Lstm(Dnn.Layer):
    """
    """
    def __init__(self, input_layer, hidden_size=1, dropout_rate=0.0, recurrent_activation="sigmoid", reverse_seq=False,
                 name=None):

        if type(input_layer) is PythonWrapper.Lstm:
            super().__init__(input_layer)
            return

        layers, outputs = Utils.check_input_layers(input_layer, 1)

        if hidden_size <= 0:
            raise ValueError('The `hidden_size` must be > 0.')

        recurrent_activation_index = ["linear", "elu", "relu", "leaky_relu", "abs", "sigmoid", "tanh", "hard_tanh",
                                      "hard_sigmoid", "power", "hswish", "gelu"].index(recurrent_activation)

        internal = PythonWrapper.Lstm(str(name), layers[0], int(outputs[0]), int(hidden_size), float(dropout_rate),
                                      recurrent_activation_index, bool(reverse_seq))
        super().__init__(internal)
