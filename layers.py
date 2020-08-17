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
import warnings
from torch.nn.parameter import Parameter

from torch.nn import Dropout, AlphaDropout
from torch.nn import RNN, GRU, LSTM, TransformerEncoderLayer
from torch.nn import ELU, Hardshrink, Hardtanh, Identity, LeakyReLU
from torch.nn import PReLU, ReLU, ReLU6, RReLU, SELU, CELU, GELU, Sigmoid
from torch.nn import Softplus, Softshrink, Softsign, Tanh, Tanhshrink, LogSigmoid

class Autoregression(torch.nn.Module):
    """
    Autoregression (AR)
        A linear layer used to dilate or contract a window-sized input to a stride-sized output.
        The layer assumes that there is a linear mapping between the input and the output.
    """

    def __init__(self, window, stride, bias=True):
        """
        Initializes the layer.
        :param window: integer
            The number of time-steps to look in the past.
        :param stride: integer
            The number of future time-steps to predict.
        :param bias: boolean
            Set bias vectors permanently to zeros if False (default: True).
        """
        super(Autoregression, self).__init__()
        # Attributes
        self.window = window
        self.stride = stride
        # Trainable bias
        self.bias = Parameter(torch.Tensor(stride, 1)).requires_grad_(bias)
        # Trainable weights
        self.weights = Parameter(torch.Tensor(window, stride))
        # Initializing parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize weights following X. Glorot & Y. Bengio (2010) and bias as zeros.
        """
        # Bias initialization
        torch.nn.init.zeros_(self.bias)
        # Weights initialization
        torch.nn.init.xavier_uniform_(self.weights)

    def forward(self, x):
        """
        Defines the computation performed at every call.
        :param x: array-like of shape (samples, window, variables)
            Observations from the past window-sized time-steps.
        :return: array-like of shape (samples, stride, variables)
            Predictions for the next stride-sized time-steps.
        """
        return torch.einsum("bd,abc->adc", self.weights, x) + self.bias

    def extra_repr(self):
        """
        Print customized extra information about the module.
        :return: string
            Details about the layer parameters.
        """
        return "window={}, stride={}, bias={}".format(
            self.window, self.stride, self.bias.requires_grad
        )

class GSEvolution(torch.nn.Module):
    """
    Graph Soft Evolution (GSE)
        The soft evolution learns a shared adjacency matrix that generalizes across the training samples.
        The evolved adjacency matrix is a representation from a multi-sample variables' co-occurrence graph.
    """
    def __init__(self, variables, in_time, out_time, time_dot=True, activation=None, dropout=.1, batch_size=32, bias=True):
        """
        Initializes the layer.
        :param variables: integer
            The number of co-occurring variables in the dataset.
        :param in_time: integer
            The size of the time-dimension of the input tensor when time-dot is true.
        :param out_time: integer
            The size of the time-dimension of the output tensor when time-dot is true.
        :param time_dot: boolean
            Whether to use a feed-forward layer on the time-axis before the output (default: True).
        :param activation: string
            The activation function to be used before the feed-forward layer (default: None).
        :param dropout: float
            The dropout probability for zeroing a neuron output (default: .1).
        :param batch_size: integer
            The number of adjacency matrices to be built simultaneously. Numerical variations will occur
            when increasing or decreasing the batch size. Be advised that, if determinism is
            imperative, one should always use the same batch-size (default: 32).
        :param bias: boolean
            Set bias vectors permanently to zeros if False (default: True).
        """
        super(GSEvolution, self).__init__()
        # Asserting parameters
        assert variables > 1, "There must be at least two co-occurring variables."
        assert batch_size > 0, "The batch size for this layer should be larger than one."
        # Attributes
        self.scale = 0  # TODO: Hardcoded because it is too unstable.
        self.epochs = 0
        self.batches = 0
        self.warned = False
        self.warming = True
        self.in_time = in_time
        self.out_time = out_time
        self.training = True
        self.time_dot = time_dot
        self.variables = variables
        self.batch_size = batch_size
        self.periodicity = 0  # TODO: Hardcoded because it is too unstable.
        self.evolution_delay = self.periodicity
        # Trainable bias
        self.time_bias = Parameter(torch.Tensor(out_time, 1)).requires_grad_(bias & time_dot)
        self.cosine_bias = Parameter(torch.Tensor(variables, variables)).requires_grad_(bias)
        self.adjacency_bias = Parameter(torch.Tensor(variables, 1)).requires_grad_(bias)
        # Trainable weights
        self.time_weights = Parameter(torch.Tensor(in_time, out_time)).requires_grad_(time_dot)
        self.cosine_weights = Parameter(torch.Tensor(variables, variables))
        self.adjacency_weights = Parameter(torch.Tensor(variables, variables))
        # Graph gradient-less variables
        self.adjacency = Parameter(torch.zeros(variables, variables)).requires_grad_(False)
        self.shared_adjacency = Parameter(torch.zeros(variables, variables)).requires_grad_(False)
        # Regularizers
        self.dropout = (torch.nn.AlphaDropout if activation == "SELU" else torch.nn.Dropout)(p=dropout)
        self.activation = eval(activation)() if activation is not None else Identity()
        # Initializing parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize weights following X. Glorot & Y. Bengio (2010) and bias as zeros.
        """
        # Bias initialization
        torch.nn.init.zeros_(self.time_bias)
        torch.nn.init.zeros_(self.cosine_bias)
        torch.nn.init.zeros_(self.adjacency_bias)
        # Weights initialization
        torch.nn.init.xavier_uniform_(self.time_weights)
        torch.nn.init.xavier_uniform_(self.cosine_weights)
        torch.nn.init.xavier_uniform_(self.adjacency_weights)

    @staticmethod
    def cosine_similarity_matrix(x, eps=1e-12):
        """
        Creates a similarity matrix for a given matrix using the cosine similarity.
            Source code adapted from https://stackoverflow.com/a/41906708.
        :param x: array-like of shape (variables, features)
            A matrix with variables matching the layer variable's attribute and any number of features.
        :param eps: float
            Small value to avoid division by zero (default: 1e-12).
        :return: array-like of shape (variables, variables)
            A squared similarity-matrix from the input.
        """
        norm = torch.norm(x, dim=1, keepdim=True)
        norm = torch.clamp(norm, min=eps)
        ebba = torch.matmul(x, x.t())
        return (ebba / norm) / norm.t()

    @staticmethod
    def __adjacency_batch(x):
        """
        Creates an adjacency matrix for the input batch.
        :param x: array-like of shape (samples, time, variables)
            Input with variables matching the layer variable's attribute.
        :return: array-like of shape (variables, variables)
            A shared adjacency matrix among the samples of the batch.
        """
        # Creating the adjacency masks
        batch_mask = (x > 0).type(x.dtype)
        # Squaring the number of variables in the batch
        adjacency = x.repeat(x.shape[2], 1, 1, 1)
        # Building adjacency matrices using matrix-mask multiplication
        adjacency = torch.einsum("dac, cdab->bc", batch_mask, adjacency)
        # Filling the diagonals with the existing self-loops
        self_loops = torch.clamp((adjacency.diagonal() * 2) - adjacency.sum(axis=1), min=0)
        adjacency.as_strided([adjacency.shape[0]], [adjacency.shape[0] + 1]).copy_(self_loops)
        # Outputs the batch-shared adjacency matrix
        return adjacency

    def preset_adjacency(self, adjacency):
        """
        Set a previously loaded adjacency matrix as the starting matrix and evolution goes from there.
        :param adjacency: array-like of shape (variables, variables)
            An adjacency matrix that will be used instead of the one extracted from the training data.
        """
        with torch.no_grad():
            self.evolution_delay = -1  # Blocking evolution from the training set
            self.adjacency.data = adjacency  # Set a previously learned matrix as the adjacency matrix

    def evolve_adjacency(self, x, uuid=None, initial_state=None):
        """
        Creates an adjacency matrix for the current batch and use it to evolve the layer's shared adjacency matrix.
            There is a link whenever two variables appear on the same sample and time-step.
            There is an incoming link for each outgoing one, but weights may differ.
            The variables-count weights the links in the batch.
        :param x: array-like of shape (samples, time, variables)
            An input with variables matching the layer variable's attribute.
        :param uuid: integer
            A strictly-positive identifier and persistent across epochs for the current input (default: None).
        :param initial_state: array-like of shape (variables, variables)
            The initial hidden state used for adjacency relearning (default: None).
        :return: array-like of shape (variables, variables)
            The layer's shared adjacency matrix.
        """
        with torch.no_grad():
            # It takes place only during training
            if self.training and self.evolution_delay >= 0:
                # Evolves during the first batch of each new epoch
                if uuid == 0:
                    # Proceeds with the evolution if not in cool-down
                    if self.epochs == int(self.evolution_delay):
                        self.warming = False  # Stopping the warming-up time
                        # Delaying future evolutions
                        if self.periodicity > 0:
                            periodic_delay = self.evolution_delay + self.periodicity
                            scaled_delay = periodic_delay + (self.evolution_delay * self.scale)
                            # Fixing bad parametrization; otherwise, evolution might never happen again
                            self.evolution_delay = max(self.epochs + 1, scaled_delay)
                    self.epochs += 1  # Registering a new epoch as completed
                # Using the initial hidden state during warming-up
                if self.warming and initial_state is not None:
                    return initial_state.data  # It must be a parameter
                # Evolving the current adjacency matrix if not already evolved
                elif uuid == self.batches or uuid is None:
                    # Slicing the data into smaller sample-batches
                    adjacency = [self.__adjacency_batch(x_batch) for x_batch in torch.split(x, self.batch_size, dim=0)]
                    # Joining all matrices in one final matrix
                    adjacency = torch.sum(torch.stack(adjacency), dim=0)
                    # Storing results to avoid re-computation
                    if uuid is not None:
                        # Merging the adjacency matrix of the current batch with the shared one
                        self.shared_adjacency.data = self.shared_adjacency.data + adjacency
                        # The previous matrix was merged with all other ones
                        adjacency = self.shared_adjacency.data.to(x.device)
                        # Registering a new batch to avoid re-merging
                        self.batches += 1
                    # Returning the adjacency matrix
                    return adjacency
            # Using the pre-computed adjacency matrix
            return self.adjacency.data

    def forward(self, x, uuid=None, initial_state=None):
        """
        Defines the computation performed at every call based on the following cases:
        1) UUID is not None and Initial State is None:
            It will create a new adjacency matrix straight ahead. This mode is more suited to be used
            on the first layer of the network to learn the shared adjacency matrix from the input.
        2) UUID is not None and Initial State is not None:
            It will use the adjacency matrix provided as the initial state to warm-up the weights of the layer before
            updating the adjacency matrix over and over again. Such updates are preset during initialization using the
            Evolution Steps and Step Scale attributes. This mode is tricky to use once synchronizing layers is not an
            ordinary task and will definitively increase the uncertainty of the model. One can see this as like an
            annealing optimization, in which we are relearning the adjacency matrix from the hidden weights itself.
        3) UUID is None and Initial State is not None:
            It will use the provided initial state as the layer's adjacency matrix and won't change it with further
            updates. It is the most stable and fast mode of operation. In this mode, the layer can be used for a
            variety of applications but mainly to stacked adjacency-relearning. In this case, instead of learning the
            adjacency from all the hidden weights, we are relearning it from the hidden weights of the last GSE layer.
        4) UUID is None and Initial State is None:
            This is the debugging mode, in which a new adjacency matrix will be created at the beginning
            of every epoch. It should be avoided in real applications as it is extremely slow.
        NOTICE: Some of the layer parameters were intentionally disabled because they are too unstable and generate
                a huge amount of uncertainty for the whole training process (see the class initialization parameters).
        :param x: array-like of shape (samples, time, variables)
            An input with variables matching the layer variable's attribute.
        :param uuid: integer
            A strictly-positive identifier and persistent across epochs for the current input (default: None).
            Notice that it should start from zero and increase until the number of the last training batch.
        :param initial_state: array-like of shape (variables, variables)
            The initial hidden state used at adjacency relearning (default: None).
        :return: array-like of shape ([samples, time, variables], [variables, variables]]
            An evolved version of the input followed by the evolved adjacency matrix.
        """
        # Layer's sanity self-check
        with torch.no_grad():
            if initial_state is not None: assert initial_state.shape == self.adjacency.shape, "Incompatible shape."
            if uuid is not None: assert uuid >= 0, "The identifier should be strictly positive or None."
            assert x.shape[2] == self.variables, "The number of variables mismatch."
            if not self.warned and (uuid is None and initial_state is None):
                warnings.warn("Beware that you are running the Evolution Layer on Debugging Mode.", Warning)
                self.warned = True  # Avoiding spamming warnings at each call
        # Warming-up and evolution (gradient-less call)
        self.adjacency.data = self.evolve_adjacency(x, uuid, initial_state)
        # Adjacency matrix (re)learning and intermediate non-linear representation
        adjacency = torch.addmm(self.adjacency_bias, self.adjacency, self.adjacency_weights)
        weights = self.cosine_similarity_matrix(adjacency)  # Similarity matrix transformation
        weights = torch.mul(weights, self.cosine_weights) + self.cosine_bias  # Data scaling
        # Producing the final output for the layer
        output = torch.einsum("ab,cdb->cda", weights, x)  # Evolution without feed-forward scaling
        output = self.dropout(output)  # Dropout before time scaling the output
        output = self.activation(output)  # Using a non-linear activation whenever requested
        if self.time_dot:
            output = torch.add(torch.einsum("bd,abc->adc", self.time_weights, output), self.time_bias)
        # Returns the evolved input and adjacency matrix
        return output, adjacency.detach()

    def extra_repr(self):
        """
        Print customized extra information about the module.
        :return: string
            Details about the layer parameters.
        """
        return ("variables={}, in_time={}, out_time={}, time_dot={}, activation={}, dropout={}, batch_size={}, bias={}"
                .format(self.variables, self.in_time, self.out_time, self.time_dot, self.activation.__class__.__name__,
                        self.dropout, self.batch_size, self.adjacency_bias.requires_grad))
