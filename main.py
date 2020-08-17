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

import argparse
import copy
import os
import gc
import platform
import pprint
import random
import sys
import uuid
import warnings
from argparse import RawTextHelpFormatter

import time
import numpy as np
from torch.backends import cudnn

from losses import *
from torch.optim import *
from networks import ReGENN
from neural_trainer import NeuralTrainer


def __main__():
    description = """
        A toolset for multi-multivariate time-series forecasting.
                            ._
                              '.
                       --._     \    .-.
                           '. .--;  /  |
                           ,;/ ^  |`\  /.-"".
                          ;' \  _/   |'    .'
                         /    `.;I> /_.--.` )
                        |    .'`  .'( _.Y/`;
                         \_.'---'`   `\  `-/`-.
                           /_-.`-.._ _/`\  ;-./
                          |   -./  ;.__.'`\\
                           `--'   (._ _.'`|
                                  /     ./
                                 ; `--';'
                                 `;-,-'
                                 ,
        Copyright 2020, Gabriel Spadon, all rights reserved.
        This code is under GNU General Public License v3.0.
            www.spadon.com.br & gabriel@spadon.com.br

        Joint research between:
           [USP] University of Sao Paulo - Sao Carlos, SP - Brazil;
        [GaTech] Georgia Institute of Technology - Atlanta, GA - USA;
          [UIUC] University of Illinois Urbana-Champaign - Champaign, IL - USA;
           [UGA] Université Grenoble Alpes - Grenoble-Alpes, France; and,
           [DAL] Dalhousie University ‑ Halifax, NS - Canada.

        Contributions:
        * Gabriel Spadon: idea, design, and development;
        * Shenda Hong & Bruno Brandoli: discussion and validation; and,
        * Stan Matwin, Jose F. Rodrigues-Jr & Jimeng Sun: discussion, validation, and idea refinement.
    """
    # Yielding program description
    parser = argparse.ArgumentParser(prog="GSNeural", description=description, formatter_class=RawTextHelpFormatter)

    # Minimization criteria and evaluation metrics
    metrics = [
        "MAELoss", "MALELoss", "MSELoss", "MSLELoss", "RMSELoss", "RMSLELoss"
    ]

    # In-layer and output activation functions
    activation = [
        "CELU", "ELU", "GELU", "Hardshrink", "Hardtanh", "Identity", "LeakyReLU", "LogSigmoid", "PReLU", "ReLU",
        "ReLU6", "RReLU", "SELU", "Sigmoid", "Softplus", "Softshrink", "Softsign", "Swish", "Tanh", "Tanhshrink"
    ]

    # [+] Positional Arguments
    parser.add_argument("input", type=str, nargs="?", help="A PyTorch tri-dimensional tensor of shape (samples, time, variables).")
    parser.add_argument("stride", type=int, help="The number of time-steps to predict.")
    parser.add_argument("validation-samples", type=float, help="Percentage of samples reserved for assessing sample-generalization.")
    parser.add_argument("validation-stride", type=int, help="The number of time-steps reserved for assessing time-generalization.")
    parser.add_argument("watch-axis", type=int, nargs="?", choices=[0, 1, 2, 3], help="The data to which the schedulers will be watching - 0: training data; 1: reserved samples, 2: reserved time-fold of the training data, and 3: reserved time-fold of the reserved samples.")
    parser.add_argument("window", type=int, help="The number of previous time-steps to look during training and testing.")

    # [-] Optional Arguments
    parser.add_argument("-aM", "--ablation-mode", action="store_true", help="Deactivates all non-vanilla layers (default: %(default)s).")
    parser.add_argument("-AR", "--autoregression", action="store_true", help="Activates an autoregressive component in parallel with the network (default: %(default)s).")
    parser.add_argument("-batch", "--batch-size", type=int, default=32, help="Number of samples to use in each training batch (default: %(default)s).")
    parser.add_argument("-b", "--bias", action="store_true", help="Learns additive bias vectors whenever it is supported (default: %(default)s).")
    parser.add_argument("-bG", "--bidirectional-gate", action="store_true", help="Sets the time gate as bidirectional (default: %(default)s).")
    parser.add_argument("-bS", "--bidirectional-sequencer", action="store_true", help="Sets the variable sequencer as bidirectional (default: %(default)s).")
    parser.add_argument("-norm", "--clip-norm", type=float, default=10, help="The gradient norm used as the clipping threshold (default: %(default)s).")
    parser.add_argument("--cost-list", type=str, nargs="+", default=["MAELoss", "RMSELoss", "MSLELoss"], choices=metrics, help="A list of three cost-functions used to evaluate the results (default: %(default)s).")
    parser.add_argument("--criterion", type=str, default="MAELoss", nargs="?", choices=metrics, help="The optimizer's minimization criterion (default: %(default)s).")
    parser.add_argument("-dM", "--debugging-mode", action="store_true", help="Turns on the debugging mode, i.e., redirects all the output to the STDOUT (default: %(default)s).")
    parser.add_argument("-drop", "--dropout", type=float, default=.1, help="Dropout probability for zeroing the output of a neuron (default: %(default)s).")
    parser.add_argument("-stop", "--early-stop", type=int, default=250, help="The number of non-improving epochs to wait before stop training (default: %(default)s).")
    parser.add_argument("--epochs", type=int, default=2500, help="The number of iterations over the whole dataset (default: %(default)s).")
    parser.add_argument("-eF", "--evolution-function", type=str, default=None, nargs="?", choices=activation, help="Activation function to be used in the evolution layer (default: %(default)s).")
    parser.add_argument("-prefix", "--file-prefix", type=str, default=None, nargs="?", help="The prefix appended to the output if saving any file (default: %(default)s).")
    parser.add_argument("-g", "--gate", type=str, default="LSTM", nargs="?", choices=["RNN", "GRU", "LSTM"], help="Defines the architecture used by the time gate (default: %(default)s).")
    parser.add_argument("--gpu", type=int, default=-1, help="Which GPU to use - automatically picks one when entering a negative number (default: %(default)s).")
    parser.add_argument("--iterator", type=str, default="time", nargs="?", choices=["time", "batch"], help="The data iterator to be used during training (default: %(default)s).")
    parser.add_argument("--k-fold", type=int, default=0, help="Number of folds to use during k-fold cross-validation (default: %(default)s).")
    parser.add_argument("-lr", "--learning-rate", type=float, default=.001, help="Learning rate of the optimizer (default: %(default)s).")
    parser.add_argument("--load-weights", type=str, nargs="?", default=None, help="Path to a pre-trained ReGENN model, from which we will use pre-learned weights to initialize the current model weights (default: %(default)s).")
    parser.add_argument("--network-type", type=str, nargs="?", default="ReGENN", choices=["ReGENN"], help="The network architecture to be used (default: %(default)s).")
    parser.add_argument("--network-uuid", type=str, nargs="?", default=None, help="Unique identifier to label the neural network (default: %(default)s).")
    parser.add_argument("-nE", "--no-encoder", action="store_true", help="Deactivates encoding layers (default: %(default)s).")
    parser.add_argument("-nS", "--no-sequencer", action="store_true", help="Deactivates the variable sequencer (default: %(default)s).")
    parser.add_argument("--normalization-axis", type=int, default=2, nargs="?", choices=[0, 1, 2], help="When normalizing by the maximum value, it indicates the axis to which the dataset will be normalized - 0: sample-axis; 1: time-axis; and, 2: variable-axis (default: %(default)s).")
    parser.add_argument("--normalization-type", type=str, default="maximum", nargs="?", choices=["linear", "maximum", "logarithm"], help="Which normalization to apply onto the dataset (default: %(default)s).")
    parser.add_argument("--optimizer", type=str, nargs="?", default="Adam", help="A valid PyTorch Optimizer (default: %(default)s).")
    parser.add_argument("-oF", "--output-function", type=str, nargs="?", default="ReLU", choices=activation, help="Activation function to be used on the output (default: %(default)s).")
    parser.add_argument("--random-seed", type=int, default=0, help="Which random seed to use - sets no seed when entering a negative number (default: %(default)s).")
    parser.add_argument("--requested-memory", type=int, default=500, help="The minimum amount of memory (in MB) required from the GPU (default: %(default)s).")
    parser.add_argument("--save-graph", action="store_true", help="Saves a snapshot of all graphs - source and target graph evolution - at each epoch (default: %(default)s).")
    parser.add_argument("--save-model", action="store_true", help="Saves a snapshot of the network after training (default: %(default)s).")
    parser.add_argument("--save-output", type=str, nargs="?", default=None, help="Full path used for saving everything logged into the STDOUT (default: %(default)s).")
    parser.add_argument("-factor", "--scheduler-factor", type=float, default=.95, help="Multiplicative factor by which the learning rate will be reduced (default: %(default)s).")
    parser.add_argument("--scheduler-min-lr", type=float, default=.0, help="A lower bound for the learning rate (default: %(default)s).")
    parser.add_argument("-patience", "--scheduler-patience", type=int, default=25, help="The number of non-improving epochs to wait before reducing the learning rate (default: %(default)s).")
    parser.add_argument("-threshold", "--scheduler-threshold", type=float, default=.1, help="The threshold used to define a significant change in the minimization criterion (default: %(default)s).")
    parser.add_argument("-s", "--sequencer", type=str, default="LSTM", nargs="?", choices=["RNN", "GRU", "LSTM"], help="Defines the architecture used by the variable sequencer (default: %(default)s).")
    parser.add_argument("--test-every", type=int, default=5, help="The frequency in which the network will be tested on the reserved data fold (default: %(default)s).")
    parser.add_argument("--variable-graph", type=str, nargs="?", default=None, help="Path to a pre-learned variable graph that will be used instead of creating a graph from the training data (default: %(default)s).")
    parser.add_argument("--working-directory", type=str, nargs="?", default="./", help="Full path to which all generated data will be saved to (default: %(default)s).")

    # Parsing arguments
    args = parser.parse_args()
    args_copy = copy.deepcopy(args)

    # Disables the sequencer on the target network
    args.sequencer = None if args.no_sequencer else args.sequencer

    # Disables the encoder layer on the target network
    args.encoder = False if args.no_encoder else True

    # Creates a new network unique identifier in case none is provided
    args.network_uuid = str(uuid.uuid4()) if args.network_uuid is None else args.network_uuid

    # Creating required directories to save the output
    os.makedirs(args.working_directory, exist_ok=True)
    os.makedirs("%s/graph" % args.working_directory, exist_ok=True)
    os.makedirs("%s/output" % args.working_directory, exist_ok=True)
    os.makedirs("%s/sandbox" % args.working_directory, exist_ok=True)
    os.makedirs("%s/snapshots" % args.working_directory, exist_ok=True)

    # If no filename is provided, we will build one that is unique
    output_file = (str(args.file_prefix) + " | ") if args.file_prefix is not None else ""
    output_file += "Ablation | " if args.ablation_mode else ""
    output_file += args.network_type + " |"
    output_file += " [AE] " if args.encoder else " "

    output_file += "b" if args.bidirectional_gate else "u"
    output_file += args.gate
    if not args.no_sequencer:
        output_file += "-b" if args.bidirectional_sequencer else "-u"
        output_file += args.sequencer

    output_file += "-AR" if args.autoregression else ""
    output_file += " +b" if args.bias else " -b"
    output_file += " | " + args.network_uuid

    # String package for printing dictionaries and regular strings
    pp = pprint.PrettyPrinter(compact=True)

    if not args.debugging_mode:  # Debug output will always be on the STDOUT
        filepath = ("%s/output/%s.txt" % (args.working_directory, output_file)) if args.save_output is None else args.save_output
        sys.stdout = open(filepath, "w+", buffering=1)  # All output will be redirected to this file

    # Enforcing uniqueness for solo runs but not for cross-validation as we can start-stop training procedures
    assert not os.path.exists("%s/snapshots/%s.pth" % (args.working_directory, args.network_uuid)), "Network UUID is not unique!"

    # We can clip predictions to evaluate a smaller period, but the model is bounded to predict stride-sized periods
    assert args.stride >= args.validation_stride, "The number of reserved strides shouldn't be higher than the stride."

    selected_gpu = False
    while not selected_gpu:
        temporary_file = "%s/sandbox/.%s.gpu" % (args.working_directory, args.network_uuid)  # List of available GPUs
        os.system("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free | sed 's/[^0-9]*//g'> %s" % temporary_file)
        gpu_memory = np.array([int(line) for line in open(temporary_file, "r").readlines()])[::-1]
        os.system("rm %s" % temporary_file)  # Removes the temporary file

        if -1 < args.gpu < len(gpu_memory):
            # Checking if the selected GPU has enough memory available
            if gpu_memory[args.gpu] >= args.requested_memory:
                gpu_id = args.gpu  # User-selected GPU
                selected_gpu = True  # Leave the GPU selector

        if not selected_gpu:
            # Automatically selects the GPU with enough memory available
            if np.max(gpu_memory) >= args.requested_memory:
                gpu_id = len(gpu_memory) - np.argmax(gpu_memory) - 1
                selected_gpu = True  # Leave the GPU selector
            # Entering sleep mode before repeating the process once more
            else: time.sleep(int(os.getpid()))  # PID is Unique

    # Setting reproducible seeds
    if args.random_seed >= 0:
        # Might impact performance
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    # Identifiers for the current Python process
    print("[Running on %s - PID #%d and GPU #%d]" % (platform.node().title(), os.getpid(), gpu_id), end="\n\n")
    print("Your training UUID is \"%s\"." % args.network_uuid)
    print("All results will be saved under that UUID.", end="\n\n")

    # Sending everything to the selected GPU
    with torch.cuda.device("cuda:%d" % gpu_id):

        # The shape must be (samples, time, variables)
        dataset = torch.load("%s/%s" % (args.working_directory, args.input)).pin_memory()
        assert (3 >= dataset.ndim >= 2), "Incompatible number of dimensions."
        dataset = torch.unsqueeze(dataset, dim=0) if dataset.ndim == 2 else dataset
        # Verbosing the dataset information
        print("Dataset(samples=%d, time-steps=%d, variables=%d)" % dataset.shape, end="\n\n")

        # Limited to three cost-functions for enhancing output visualization
        assert len(args.cost_list) == 3, "Make sure to select three cost-functions from the list."
        args.cost_list = [eval(cost_function)().cuda() for cost_function in args.cost_list]

        # Validating training-, development-, and testing-related variables
        assert args.validation_stride >= 0 and 1. >= args.validation_samples >= 0. and args.window > 0 and args.stride > 0 and \
               args.k_fold >= 0 and dataset.shape[0] > (dataset.shape[0] * args.validation_samples) and \
               dataset.shape[1] >= (args.window + args.validation_stride + (2 * args.stride)) and \
               (args.validation_stride > 0 or args.validation_samples > 0 or args.k_fold > 0), \
            "Unfeasible parametrization for the current dataset."

        # Checking the fit-watch parameter used by the schedulers
        assert not ((args.watch_axis == 1 or args.watch_axis == 3) and args.validation_samples == 0) and \
               not ((args.watch_axis == 2 or args.watch_axis == 3) and args.validation_stride == 0), \
            "Make some data available for testing on time- and/or sample-reserved data."

        if args.k_fold > 0:
            if args.validation_samples > 0:
                # k-fold cross-validation will automatically split training and test data
                warnings.warn("The cross-validation will overwrite the reserved-sample parameter.", Warning)
            # Running cross-validation mode
            cross_train(dataset, args)
        else:
            # Running single-run training mode
            train_once(dataset, args)

        # Printing user-input parameters for further reproducibility and also sanity-check
        print("\n[HYPERPARAMETERS]", end="\n\n")
        print(pp.pformat(vars(args_copy)))


def __train(dataset, samples, fold_id, args):
    """
    Vanilla training.
    :param dataset: array-like of shape (samples, time, variables)
        The complete dataset for training and testing the network.
    :param samples: array-like of shape ([set-samples, dev-samples], 1)
        List of training and test sample indices.
    :param fold_id: integer
        The number of the current dataset fold.
    :param args: parsed dictionary
        A set of parsed command-line arguments.
    :return: array-like of shape (loss, [trainer, network, args])
        Test loss and all variables required to reproduce it.
    """
    # Instancing the minimization criterion
    args.criterion = eval(args.criterion)().cuda()
    # Creating a new instance of the neural network
    network = eval(args.network_type)(variables=dataset.shape[2], **vars(args)).cuda()
    # Overwrite the network initialization with a pre-trained set of weights
    if args.load_weights and not args.ablation_mode:
        state_dicts = torch.load(args.load_weights, map_location=lambda storage, loc: storage)
        _drop = torch.nn.Dropout(p=.2)
        for k in state_dicts.keys():
            if isinstance(state_dicts[k], dict):
                for v in state_dicts[k].keys():
                    if isinstance(state_dicts[k][v], torch.Tensor):
                        state_dicts[k][v] = _drop(state_dicts[k][v])
        network.load_state_dict(state_dicts["model_dict"])
    # Using a pre-learned graph to the network source-evolution layer
    if args.variable_graph and not args.ablation_mode:
        graph = torch.load(args.variable_graph, map_location=lambda storage, loc: storage)
        graph = torch.from_numpy(graph).cuda()
        network.SourceEvolution.preset_adjacency(graph)
    # Verbosing network-init configuration
    if fold_id == 0:
        # No need to verbose the network architecture at every fold
        print("[NETWORK-SUMMARY]\n\n" + str(network), end="\n\n")
    # Instancing the user-selected optimizer
    args.optimizer = eval(args.optimizer)(network.parameters(), lr=args.learning_rate)
    # Sending data to the NeuralTrainer
    trainer = NeuralTrainer(network=network, dataset=dataset, samples=samples, **vars(args))
    # Training the network and testing as soon as a better model is found
    loss = trainer.train()
    # Retrieves the network with the best parameters
    network = trainer.network
    # Returns the test loss and the best network
    return loss, (trainer, network, args)


def train_once(dataset, args):
    """
    Trains the network following the user's training-test split approach.
    :param dataset: array-like of shape (samples, time, variables)
        The complete dataset for training and testing the network.
    :param args: parsed dictionary
        A set of parsed command-line arguments.
    """
    # Shuffling and slicing the dataset samples
    items = int(dataset.shape[0] * args.validation_samples)
    samples = np.random.permutation(dataset.shape[0])
    x_set, x_dev = samples[items:], samples[:items]
    # Training the network using x-set and assessing generalization with x-dev
    test_loss, (trainer, network, args) = __train(dataset, (x_set, x_dev), fold_id=0, args=args)
    # Saving all network-related variables when requested
    if args.save_model:
        # Preparing all the state dictionaries - enables reproducibility and start-stop training
        args_dict, model_dict, trainer_dict, scheduler_dict, optimizer_dict = __state_dict(trainer, args)
        # Saving all state dictionaries
        torch.save({
            "test_loss": test_loss,
            "args_dict": args_dict,
            "model_dict": model_dict,
            "trainer_dict": trainer_dict,
            "scheduler_dict": scheduler_dict,
            "optimizer_dict": optimizer_dict,
        }, "%s/snapshots/%s.pth" % (args.working_directory, args.network_uuid))
    # Creating a metric summary to speed-up results assessment
    print("\n[SUMMARY]", end="\n\n")
    # String formatter for summary building
    tokenizer = lambda instance, idx: str(tuple(zip(instance.metric_storage[idx], instance.metric_deviation[idx])))
    # The first conditional case deals with regular regression and the second one with regular time-series
    metric_list = [0, 1, 2, 3, 4, 5]  # However, as default, we will return results from all six sets of data
    metric_list = [0, 1, 4, 5] if args.validation_samples > 0 and args.validation_stride == 0 else metric_list
    metric_list = [0, 2, 4] if args.validation_samples == 0 and args.validation_stride > 0 else metric_list
    # The summary layout depends on the reserved data for validation
    tk = str(tuple(tokenizer(trainer, idx) for idx in metric_list))
    # After a few tweaks, the summary string will be ready to be pasted to a spreadsheet
    print("%s\t%s" % (tk.replace("(", "").replace(")", "").replace(", ", "\t").replace("'", ""), args.network_uuid))


def __state_dict(trainer, args):
    """
    Builds a state dictionary from a neural trainer instance. In the case of NeuralTrainer changes,
    this method might fail. The method is outside the proper class because it will destroy all
    major attributes within it. This means that this method should be the last thing you call.
    neural_trainer: object
        A NeuralTrainer instance.
    :param args: parsed dictionary
        A set of parsed command-line arguments.
    return: array-like of shape (model-sd, trainer-sd, scheduler-sd, optimizer-sd)
        An array with the state dictionaries extracted from the model, trainer, scheduler, and optimizer.
    """
    # Tensors to NumPy & GPU to CPU
    trainer.criterion = trainer.criterion.name
    trainer.scale = trainer.scale.cpu().numpy()
    trainer.network_uuid = str(trainer.network_uuid)
    trainer.data_iterator = trainer.data_iterator.__name__
    trainer.cost_list = [cost_function.name for cost_function in trainer.cost_list]
    # Will be handled as outside variables
    model_dict = trainer.network.cpu().state_dict()
    optimizer_dict = trainer.optimizer.state_dict()
    scheduler_dict = trainer.scheduler.state_dict()
    # Rollback parsed arguments to raw-arguments
    args.cost_list = trainer.cost_list
    args.criterion = trainer.criterion
    args.network_uuid = trainer.network_uuid
    args.optimizer = args.optimizer.__class__.__name__
    # Deleting all the objects that should not be saved
    del trainer.dataset  # The dataset is private and won't be saved
    del trainer.network, trainer.scheduler, trainer.optimizer  # Will save the state dictionary
    return vars(args), model_dict, trainer.__dict__, scheduler_dict, optimizer_dict


if __name__ == "__main__":
    # Calls the main function when scripting
    __main__()
