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
--------------------------------------------------------------------------------------------------------------*/
"""

import neoml.PythonWrapper as PythonWrapper
import neoml.Dnn as Dnn
import neoml.Utils as Utils


class Pooling(Dnn.Layer):
    """
    """
    def __init__(self, internal):
        if not isinstance(internal, PythonWrapper.Pooling):
            raise ValueError('The `internal` must be PythonWrapper.Pooling')

        super().__init__(internal)

    @property
    def filter_size(self):
        """
        """
        return self._internal.get_filter_height(), self._internal.get_filter_width()

    @filter_size.setter
    def filter_size(self, filter_size):
        """
        """
        if len(filter_size) != 2:
            raise ValueError('The `filter_size` must contain two values (h, w).')

        self._internal.set_filter_height(int(filter_size[0]))
        self._internal.set_filter_width(int(filter_size[1]))

    @property
    def stride_size(self):
        """
        """
        return self._internal.get_stride_height(), self._internal.get_stride_width()

    @stride_size.setter
    def stride_size(self, stride_size):
        """
        """
        if len(stride_size) != 2:
            raise ValueError('The `filter_size` must contain two values (h, w).')

        self._internal.set_stride_height(int(stride_size[0]))
        self._internal.set_stride_width(int(stride_size[1]))

# ----------------------------------------------------------------------------------------------------------------------


class MaxPooling(Pooling):
    """
    """
    def __init__(self, input_layers, filter_size=(3, 3), stride_size=(1, 1), name=None):

        if type(input_layers) is PythonWrapper.MaxPooling:
            super().__init__(input_layers)

        layers, outputs = Utils.check_input_layers(input_layers, 1)

        if len(filter_size) != 2:
            raise ValueError('The `filter_size` must contain two values (h, w).')

        if len(stride_size) != 2:
            raise ValueError('The `stride_size` must contain two values (h, w).')

        internal = PythonWrapper.MaxPooling(str(name), layers[0], outputs[0], int(filter_size[0]),
                                            int(filter_size[1]), int(stride_size[0]), int(stride_size[1]))
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class MeanPooling(Pooling):
    """
    """
    def __init__(self, input_layers, filter_size=(3, 3), stride_size=(1, 1), name=None):

        if type(input_layers) is PythonWrapper.MeanPooling:
            super().__init__(input_layers)
            return

        layers, outputs = Utils.check_input_layers(input_layers, 1)

        if len(filter_size) != 2:
            raise ValueError('The `filter_size` must contain two values (h, w).')

        if len(stride_size) != 2:
            raise ValueError('The `stride_size` must contain two values (h, w).')

        internal = PythonWrapper.MeanPooling(str(name), layers[0], outputs[0], int(filter_size[0]),
                                             int(filter_size[1]), int(stride_size[0]), int(stride_size[1]))
        super().__init__(internal)
