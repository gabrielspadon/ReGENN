# coding=utf-8
#
#  Copyright 2020, Gabriel Spadon, all rights reserved.
#  This code is under GNU General Public License v3.0.
#      www.spadon.com.br & gabriel@spadon.com.br
#
#  Joint research between:
#     [USP] University of Sao Paulo - Sao Carlos, SP - Brazil;
#  [GaTech] Georgia Institute of Technology - Atlanta, GA - USA;
#    [UIUC] University of Illinois Urbana-Champaign - Champaign, IL - USA;
#     [UGA] Université Grenoble Alpes - Grenoble-Alpes, France; and,
#     [DAL] Dalhousie University ‑ Halifax, NS - Canada.
#
#  Contributions:
#  * Gabriel Spadon: idea, design, and development;
#  * Shenda Hong & Bruno Brandoli: discussion and validation; and,
#  * Stan Matwin, Jose F. Rodrigues-Jr & Jimeng Sun: discussion, validation, and idea refinement.

import torch

from layers import GSEvolution, Autoregression

from torch.nn import RNN, GRU, LSTM, TransformerEncoderLayer
from torch.nn import ELU, Hardshrink, Hardtanh, Identity, LeakyReLU
from torch.nn import PReLU, ReLU, ReLU6, RReLU, SELU, CELU, GELU, Sigmoid
from torch.nn import Softplus, Softshrink, Softsign, Tanh, Tanhshrink, LogSigmoid

# Stable Neural Network Architecture

class ReGENN(torch.nn.Module):
    """
    Recurrent Graph Evolution Neural Network (ReGENN)
    - In this architecture, the Time Gate layer comes before the Variable Sequencer layer.
    - It is a more straight forward way of thinking about the problem but might suffer from gradient vanishing.
    - Residual connections are limited in this case as we change the time-axis in the early stages of the pipeline.
    """

    def __init__(self, window, stride, variables, gate, sequencer, autoregression, evolution_function, output_function,
                 dropout, bidirectional_gate, bidirectional_sequencer, encoder, bias, ablation_mode, **kwargs):
        """
        Initializes the layers of the network with the user-input arguments.
        :param window: integer
            The number of time-steps to look in the past.
        :param stride: integer
            The number of future time-steps to predict.
        :param variables: integer
            The number of co-occurring variables in the dataset.
        :param gate: string
            Defines the architecture used by the Time Gate.
        :param sequencer: string
            Defines the architecture used by the Variable Sequencer.
        :param autoregression: boolean
            Uses no autoregressive component if set to False.
        :param evolution_function: string
            Activation function to be used in the evolution layer.
        :param output_function: string
            Activation function to be used on the output.
        :param dropout: float
            Dropout probability for zeroing a neuron output.
        :param bidirectional_gate: boolean
            Sets the Time Gate as bidirectional if set to True.
        :param bidirectional_sequencer: boolean
            Sets the Variable Sequencer as bidirectional if set to True.
        :param encoder: boolean
            Disable encoding layers if set to False.
        :param bias: boolean
            Sets bias vectors permanently to zeros if set to False.
        :param ablation_mode: boolean
            Deactivates all non-vanilla layers.
        """
        super(ReGENN, self).__init__()
        # Attributes
        self.bias = bias
        self.window = window
        self.stride = stride
        self.encoder = encoder
        self.variables = variables
        self.ablation_mode = ablation_mode
        # Parametrized Layers
        # [+] Graph evolution for multi-multivariate learning (input graph)
        self.SourceEvolution = GSEvolution(variables=variables, in_time=window, out_time=window, time_dot=True,
                                           activation=evolution_function, dropout=dropout, batch_size=32,
                                           bias=bias) if not ablation_mode else None
        # [+] TimeEncoder for time-embedding learning
        self.TimeEncoder = TransformerEncoderLayer(d_model=window, nhead=1, dropout=dropout) if self.encoder else None
        # [+] TimeGate for mapping a window-sized input to a stride-sized output
        self.TimeGate = eval(gate)(window, stride, batch_first=True, bidirectional=bidirectional_gate, bias=bias)
        # [+] VariableSequencer for time-based variable scaling
        self.VariableSequencer = eval(sequencer)(input_size=variables, hidden_size=variables,
                                                 batch_first=True, bidirectional=bidirectional_sequencer,
                                                 bias=bias) if sequencer is not None else None
        # [+] Graph evolution for multi-multivariate learning (output graph)
        self.TargetEvolution = GSEvolution(variables=variables, in_time=stride, out_time=stride, time_dot=False,
                                           activation=None, dropout=.0, batch_size=32,
                                           bias=bias) if not ablation_mode else None
        # [+] Autoregression for linearly scaling the output
        self.Autoregression = Autoregression(window=window, stride=stride, bias=bias) if autoregression else None
        # Regularizers
        self.TimeDropout = torch.nn.Dropout(p=dropout)  # Between the time gate and variable sequencer
        self.VariableDropout = torch.nn.Dropout(p=dropout) if sequencer is not None else None  # Before Graph Evolution
        self.OutputActivation = eval(output_function)()  # Output Activation is persistent throughout the ablation test

    def forward(self, x, uuid=None):
        """
        Defines the computation performed at every call.
        :param x: array-like of shape (samples, window, variables)
            Observations from the past window-sized time-steps.
        :param uuid: integer
            A unique identifier, but persistent across different epochs, for the current observations (default: None).
        :return: array-like of shape (samples, stride, variables)
            Predictions for the next stride-sized time-steps.
        """

        # 1. Graph evolution for multi-multivariate learning (input graph)

        if not self.ablation_mode:
            # Outputs the results first and the evolved graph after that
            se, sg = self.SourceEvolution(x, uuid=uuid, initial_state=None)
        # Skipping Graph Evolution
        else: se = x

        # Time to the last axis
        se = se.permute(0, 2, 1)

        # 2. TimeEncoder for time-embedding learning

        tex = self.TimeEncoder(se) if self.encoder else se

        # 3. TimeGate for mapping a window-sized input to a stride-sized output

        tx, _ = self.TimeGate(tex)
        if self.TimeGate.bidirectional:
            # Merging the two branches into a single one
            tx = tx.view(-1, self.variables, 2, self.stride).sum(axis=2)
        tx = tx.permute(0, 2, 1)  # Permuting the time axis to the original position
        tx = self.TimeDropout(tx)  # Randomly zeroing some of the neurons

        # 4. VariableSequencer for time-based variable scaling

        if self.VariableSequencer is not None:
            vx, _ = self.VariableSequencer(tx)
            if self.VariableSequencer.bidirectional:
                # Merging the two branches into a single one
                vx = vx.view(-1, self.stride, 2, self.variables).sum(axis=2)
            vx = self.VariableDropout(vx) + tx  # Dropout & Residual Connection
        # Skipping Variable Sequencer
        else: vx = tx

        # 5. Graph evolution for multi-multivariate learning (output graph)

        if not self.ablation_mode:
            # Outputs the results first and the evolved graph after that
            te, _ = self.TargetEvolution(vx, uuid=None, initial_state=sg)
        # Skipping Graph Evolution
        else: te = vx

        # 6. Autoregressive linear-scaling component

        if self.Autoregression is not None:
            # Linear highway-like component
            ar = self.Autoregression(x)
            # Parallel scaled output
            return self.OutputActivation(te + ar)

        # 7. Output without parallel scaling
        return self.OutputActivation(te)
