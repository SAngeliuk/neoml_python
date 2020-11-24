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


class AttentionDecoder(Dnn.Layer):
    """AttentionDecoderLayer implements a layer that converts the input sequence
        into the output sequence, not necessarily of the same length
    """
    scores = ["additive", "dot_product"]

    def __init__(self, input_layers, score, hidden_size, output_object_size, output_seq_len, name=None):

        if type(input_layers) is PythonWrapper.AttentionDecoder:
            super().__init__(input_layers)
            return

        layers, outputs = Utils.check_input_layers(input_layers, 2)

        score_index = self.scores.index(score)

        if output_object_size <= 0:
            raise ValueError('The `output_object_size` must be > 0.')

        if output_seq_len <= 0:
            raise ValueError('The `output_seq_len` must be > 0.')

        if hidden_size <= 0:
            raise ValueError('The `hidden_size` must be > 0.')

        internal = PythonWrapper.AttentionDecoder(str(name), layers[0], int(outputs[0]), layers[1], int(outputs[1]),
                                                  score_index, int(output_object_size), int(output_seq_len),
                                                  int(hidden_size))
        super().__init__(internal)

    @property
    def score(self):
        """Get the estimate function.
        """
        return self.scores[self._internal.get_score()]

    @score.setter
    def score(self, new_score):
        """Set the estimate function.
        """
        score_index = self.scores.index(new_score)
        self._internal.set_score(score_index)

    @property
    def output_seq_len(self):
        """Get the length of the output sequence.
        """
        return self._internal.get_output_seq_len()

    @output_seq_len.setter
    def output_seq_len(self, output_seq_len):
        """Set the length of the output sequence
        """
        self._internal.set_output_seq_len(int(output_seq_len))

    @property
    def output_object_size(self):
        """Get the number of channels.
        """
        return self._internal.get_output_object_size()

    @output_object_size.setter
    def output_object_size(self, output_object_size):
        """Set the number of channels.
        """
        self._internal.set_output_object_size(int(output_object_size))

    @property
    def hidden_layer_size(self):
        """Get the size of the hidden layer.
        """
        return self._internal.get_hidden_layer_size()

    @hidden_layer_size.setter
    def hidden_layer_size(self, hidden_layer_size):
        """Set the size of the hidden layer.
        """
        self._internal.set_hidden_layer_size(int(hidden_layer_size))
