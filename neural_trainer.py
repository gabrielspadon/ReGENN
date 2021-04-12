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

import os
import time

import torch
import traceback
import numpy as np
from numpy import array_split as batchify
from torch.optim.lr_scheduler import ReduceLROnPlateau


class NeuralTrainer:
    """
    ReGENN training and testing routines.
    """

    def __init__(self, network, dataset, samples, **kwargs):
        """
        Trains the neural network using user-defined parameters.
        :param network: object
            A ReGENN instance.
        :param dataset: array-like of shape (samples, time, variables)
            The complete dataset for training and testing the network.
        :param samples: array-like of shape ([set-indices, dev-indices], 1)
            List of training and development samples.
        """
        self.network = network  # ReGENN pre-instantiated object
        self.dataset = dataset  # Complete training/test dataset
        self.set_samples = samples[0]  # Training/test sample indices
        self.dev_samples = samples[1]  # Development sample indices

        # The dataset must match the internal tensor type
        self.dataset = self.dataset.to(torch.float32)

        for key, value in kwargs.items():
            # Mapping all kwargs to attributes
            setattr(self, key, value)

        # Whether the network will be tested on unseen samples or not
        self.unknown_samples = self.validation_samples > 0# or self.k_fold > 0

        if self.iterator == "time":
            # Shows to the network how time-series evolve over time
            # >>> After that will iterate over different time series
            self.data_iterator = self.time_iterator
        else:
            # Shows to the network how different time-series vary on the same time-step
            # >>> After that iterates over different time-steps, that is, through time
            self.data_iterator = self.batch_iterator

        self.metric_storage = [[], [], [], [], [], []]  # List of metrics observed in the best model
        self.metric_deviation = [[], [], [], [], [], []]  # Deviation of metrics observed in the best model
        self.min_loss = np.inf  # Store the best loss observed during the validation
        self.forward_index = 0  # The number of forward calls during the first epoch

        # These are the number of elements in the:
        self.z = self.dataset.shape[0]  # sample axis (z-axis)
        self.y = self.dataset.shape[1]  # time axis (y-axis)
        self.x = self.dataset.shape[2]  # variable axis (x-axis)

        # These are CONSTANTS:
        self.SET_ID = 0  # Training/Test loss-matrix index
        self.DEV_ID = 1  # Development loss-matrix index

        # These will track the last:
        self.time_id = 0  # time-batch
        self.epoch_id = 0  # epoch
        self.batch_id = 0  # sample-batch

        # The number of training time-batches for a window-sized input
        self.time_test = self.y - (self.window + self.stride)  # x-start for testing (stride will be used)
        self.time_dev = self.time_test - self.validation_stride  # x-start for dev (dev-stride is the stride)
        self.time_train = self.time_dev - self.stride + 1  # number of time batches (stride will be used)

        # Causing an exception in case time-dev is used where it is not supposed to
        self.time_dev = -1 if self.validation_stride == 0 else self.time_dev

        # Number of batches existing on the sample axis of the dataset
        self.sample_batches = max(len(self.set_samples) // self.batch_size, 1)

        # Number of cost functions to use for evaluation
        self.costs = len(self.cost_list)

        # Training variables - there is just one time-batch for the time-development and test sets
        self.trn_loss = np.zeros((2, self.epochs, self.time_train, self.sample_batches, self.costs))  # Training
        self.tst_loss = np.zeros((2, self.epochs, 1, self.sample_batches, self.costs))  # Test on stride-reserved data
        self.dev_loss = np.zeros((2, self.epochs, 1, self.sample_batches, self.costs))  # Development on stride-reserved data

        # Disables the scheduler by setting a unreachable patience
        self.scheduler_patience = self.scheduler_patience if self.scheduler_patience > 0 else self.epochs * 2

        # Scheduler for reducing the learning rate when a metric has stopped improving
        self.scheduler = ReduceLROnPlateau(optimizer=self.optimizer,  # A PyTorch compatible optimizer
                                           patience=self.scheduler_patience,  # Number of epochs to wait before acting
                                           factor=self.scheduler_factor,  # Factor by which the lr will be reduced
                                           threshold=self.scheduler_threshold,  # Threshold for measuring improvement
                                           min_lr=self.scheduler_min_lr,  # A lower bound for the learning rate
                                           verbose=False,  # Verbose scheduling information to the STDOUT
                                           eps=1e-12)  # Minimum decay applied to the learning rate

        # Logarithmic scale will also min-max scale the dataset
        if self.normalization_type == "logarithm":
            self.dataset = torch.log1p(self.dataset)

        # Handling different user-selected scales
        if self.normalization_type == "linear":
            # Scale by one means no scaling at all
            self.scale = torch.ones(self.z, 1, 1)
        else:
            # Scaling the dataset based on the max-sample value
            self.scale = self.dataset.max(dim=self.normalization_axis, keepdim=True).values.clamp_min(1.)

        # Final normalization based on user input
        self.dataset = self.dataset / self.scale

    def __slice_scale(self, data_batch, t_start, t_end):
        """
        Slices the scale array so that the scale dimension is compatible with the size of the data set.
        param data_batch: array-like of shape (batch-size,)
            List of indices in the current data batch.
        param t_start: integer
            Starting time-step of the current slice.
        param t_end: integer
            Ending time-step of the current slice.
        return: array-like of varying shape.
            An array of sliced scales according to the normalization criterion.
        """
        if self.normalization_axis == 1 or self.normalization_type == "linear":
            # Time normalization requires sub-sampling
            return self.scale[data_batch, :, :].pin_memory()
        if self.normalization_axis == 0:
            # Sample normalization requires time-slicing
            return self.scale[:, t_start:t_end, :].pin_memory()
        # Variable normalization requires both sub-sampling and time-slicing
        return self.scale[data_batch, t_start:t_end, :].pin_memory()

    def time_iterator(self, test_iteration=False, time_validation=False, shuffle=True):
        """
        Iterates over the time-axis training the network through time.
        :param test_iteration: boolean
            Iterates over the test set if the value is set to True (default: False).
        :param time_validation: boolean
            Iterates over the stride-reserved set if the value is set to True (default: False).
        :param shuffle: boolean
            Whether to shuffle the dataset before iterating over it (default: True).
        :return: array-like of shape ([x-set, y-set, x-dev, y-dev], 1)
            X and Y sets for training or testing followed, when available, by X and Y sets of reserved samples.
        """
        assert not (not test_iteration and time_validation), "There is no stride-reserved set for the training data."
        # Shuffling mandatory-samples for batching data
        self.set_samples = np.random.permutation(self.set_samples) if shuffle else self.set_samples
        # Batching the data on the mandatory-sample set
        batched_data = batchify(self.set_samples, self.sample_batches)
        # Dealing with sample-generalization
        if self.unknown_samples:
            # Shuffling the sample-generalization data
            self.dev_samples = np.random.permutation(self.dev_samples) if shuffle else self.dev_samples
            # Batching additional sample data
            batched_data = list(zip(batched_data, batchify(self.dev_samples, self.sample_batches)))
        # Iterating over the data batches
        for self.batch_id, set_batch in enumerate(batched_data):
            # Dealing with sample-generalization
            if self.unknown_samples:
                # In this case, we have a tuple of sets
                set_batch, dev_batch = set_batch
            # Handling training first
            if not test_iteration:
                # The output will be None whenever no data exist
                mini_batch = np.array([None, None, None, None])
                # Number of time-batches that fit in the dataset
                for self.time_id in range(self.time_train):
                    x_start, x_end = self.time_id, (self.time_id + self.window)  # Features indices
                    y_start, y_end = x_end, (x_end + self.stride)  # Labels indices
                    # Slicing the time-window for training and testing (known samples)
                    mini_batch[[0, 1]] = (self.dataset[set_batch, x_start:x_end].cuda(),
                                          self.dataset[set_batch, y_start:y_end].cuda())
                    # Dealing with sample-generalization
                    if self.unknown_samples:
                        # Slicing the time-window for development (unknown samples)
                        mini_batch[[2, 3]] = (self.dataset[dev_batch, x_start:x_end].cuda(),
                                              self.dataset[dev_batch, y_start:y_end].cuda())
                        # Scaling factor for the development set
                        self.dev_scale = self.__slice_scale(dev_batch, y_start, y_end).cuda()
                    # Scaling factors for the training/test set
                    self.set_scale = self.__slice_scale(set_batch, y_start, y_end).cuda()
                    # Yielding the final batch
                    yield mini_batch
            # Time development and regular testing are handled in here
            else:
                # The output will be None whenever no data exist
                mini_batch = np.array([None, None, None, None])
                # Time development uses reserved-stride instead of stride
                start_step, end_step = self.time_dev, self.validation_stride
                if not time_validation:
                    # Testing is not so frequent as the development test
                    start_step, end_step = self.time_test, self.stride
                # There are no time-batches, just a single set of features and labels
                x_start, x_end = start_step, (start_step + self.window)  # Features indices
                y_start, y_end = x_end, (x_end + end_step)  # Labels indices
                # Slicing the samples that are within the mandatory batch
                mini_batch[[0, 1]] = (self.dataset[set_batch, x_start:x_end].cuda(),
                                      self.dataset[set_batch, y_start:y_end].cuda())
                # Dealing with sample-generalization
                if self.unknown_samples:
                    # Slicing the samples that are within the generalization batch
                    mini_batch[[2, 3]] = (self.dataset[dev_batch, x_start:x_end].cuda(),
                                          self.dataset[dev_batch, y_start:y_end].cuda())
                    # Scaling factor for the development set
                    self.dev_scale = self.__slice_scale(dev_batch, y_start, y_end).cuda()
                # Scaling factors for the training/test set
                self.set_scale = self.__slice_scale(set_batch, y_start, y_end).cuda()
                # Yielding the final batch
                yield mini_batch

    def batch_iterator(self, test_iteration=False, time_validation=False, shuffle=True):
        """
        Iterates over the sample-axis training the network through the sample space.
        :param test_iteration: boolean
            Iterates over the test set if the value is set to True (default: False).
        :param time_validation: boolean
            Iterates over the stride-reserved set if the value is set to True (default: False).
        :param shuffle: boolean
            Whether to shuffle the dataset before iterating over it (default: True).
        :return: array-like of shape ([x-set, y-set, x-dev, y-dev], 1)
            X and Y sets for training or testing followed, when available, by X and Y sets of reserved samples.
        """
        assert not (not test_iteration and time_validation), "There is no stride-reserved set for the training data."
        # Shuffling mandatory-samples for batching data
        self.set_samples = np.random.permutation(self.set_samples) if shuffle else self.set_samples
        # Batching the data on the mandatory-sample set
        batched_data = batchify(self.set_samples, self.sample_batches)
        # Dealing with sample-generalization
        if self.unknown_samples:
            # Shuffling the sample-generalization data
            self.dev_samples = np.random.permutation(self.dev_samples) if shuffle else self.dev_samples
            # Batching additional sample data
            batched_data = list(zip(batched_data, batchify(self.dev_samples, self.sample_batches)))
        # Handling training first
        if not test_iteration:
            # Number of time-batches that fit in the dataset
            for self.time_id in range(self.time_train):
                x_start, x_end = self.time_id, (self.time_id + self.window)
                y_start, y_end = x_end, (x_end + self.stride)
                # The output will be None whenever no data exist
                mini_batch = np.array([None, None, None, None])
                # Iterating over the data batches
                for self.batch_id, set_batch in enumerate(batched_data):
                    # Dealing with sample-generalization
                    if self.unknown_samples:
                        # In this case, we have a tuple of sets
                        set_batch, dev_batch = set_batch
                        # Scaling factor for the development set
                        self.dev_scale = self.__slice_scale(dev_batch, y_start, y_end).cuda()
                        # Slicing the samples that are within the generalization batch
                        mini_batch[[2, 3]] = (self.dataset[dev_batch, x_start:x_end].cuda(),
                                              self.dataset[dev_batch, y_start:y_end].cuda())
                    # Scaling factors for the training set
                    self.set_scale = self.__slice_scale(set_batch, y_start, y_end).cuda()
                    # Slicing the samples that are within the mandatory batch
                    mini_batch[[0, 1]] = (self.dataset[set_batch, x_start:x_end].cuda(),
                                          self.dataset[set_batch, y_start:y_end].cuda())
                    # Yielding the final batch
                    yield mini_batch
        # Time development and regular testing are handled in here
        else:
            # The output will be None whenever no data exist
            mini_batch = np.array([None, None, None, None])
            # Time development uses reserved-stride instead of stride, and both are bounded to different time-steps
            start_step, end_step = (self.time_dev, self.validation_stride) if time_validation else (self.time_test, self.stride)
            # There are no time-batches, just a single set of features and labels
            x_start, x_end = start_step, (start_step + self.window)
            y_start, y_end = x_end, (x_end + end_step)
            # Iterating over the data batches
            for self.batch_id, set_batch in enumerate(batched_data):
                # Dealing with sample-generalization
                if self.unknown_samples:
                    # In this case, we have a tuple of sets
                    set_batch, dev_batch = set_batch
                    # Scaling factor for the development set
                    self.dev_scale = self.__slice_scale(dev_batch, y_start, y_end).cuda()
                # Scaling factors for the test set
                self.set_scale = self.__slice_scale(set_batch, y_start, y_end).cuda()
                # Slicing the samples that are within the mandatory batch
                mini_batch[[0, 1]] = (self.dataset[set_batch, x_start:x_end].cuda(),
                                      self.dataset[set_batch, y_start:y_end].cuda())
                # Dealing with sample-generalization
                if self.unknown_samples:
                    # Slicing the samples that are within the generalization batch
                    mini_batch[[2, 3]] = (self.dataset[dev_batch, x_start:x_end].cuda(),
                                          self.dataset[dev_batch, y_start:y_end].cuda())
                # Yielding the final batch
                yield mini_batch

    def __train(self, x, y):
        """
        PyTorch training routine.
        :param x: array-like of shape (samples, window, variables)
            Observations from the past window-sized time-steps.
        :param y: array-like of shape (samples, stride, variables)
            Predictions for the next stride-sized time-steps.
        """
        # 1. Setting the network to training mode
        self.network.train()
        # 2. Zeroing the gradient buffers
        self.optimizer.zero_grad()
        # 3. Forward propagation
        y_pred = self.network(x, self.forward_index)
        # 4. Incrementing the number of forward calls
        self.forward_index += 1
        # 5. Rollback min-max normalization (always use set-scale)
        y_pred, y = y_pred * self.set_scale, y * self.set_scale
        # 6. Rollback logarithm scaling
        if self.normalization_type == "logarithm":
            y_pred, y = torch.expm1(y_pred), torch.expm1(y)
        # 7. Computing the resulting loss
        loss = self.criterion(y_pred, y)
        # 8. Computing the gradients
        loss.backward()
        # 9. Compute and clip gradients norms
        if self.clip_norm > 0:
            gradient_norm = 0
            # 9.1. Not all parameters have gradients
            for param in self.network.parameters():
                if param.grad is not None:
                    gradient_norm += torch.pow(param.grad.data.norm(), 2.)
            # 9.2. Clipping norms whenever required
            gradient_norm = torch.sqrt(gradient_norm)
            if gradient_norm > 0:
                shrinkage = self.clip_norm / gradient_norm
                if shrinkage < 1:
                    for param in self.network.parameters():
                        if param.grad is not None:
                            param.grad.data.mul_(shrinkage)
        # 10. Updating parameters
        self.optimizer.step()

    def __test(self, x, y, sample_validation=False, to_float=True):
        """
        PyTorch testing routine.
        :param x: array-like of shape (samples, window, variables)
            Observations from the past window-sized time-steps.
        :param y: array-like of shape (samples, [reserved-]stride, variables)
            Predictions for the next [reserved-]stride-sized time-steps.
        :param sample_validation: boolean
            Changes the data scale to the sample-development scale if set to True (default: False).
        :param to_float: boolean
            Whether to return the result as a floating-point number (default: True).
        :return: array-like of shape ([criterion, cost-list], 1)
            The criterion loss and a set of pre-defined cost-function between the prediction and observation.
        """
        # 1. Setting the network to evaluation mode
        self.network.eval()
        # 2. Retrieving the correct scale for the input data
        scale = self.dev_scale if sample_validation else self.set_scale
        # 3. Ignoring all gradients for the next call
        with torch.no_grad():
            # 4. Forward propagation
            y_pred = self.network(x, self.forward_index)
        # 5. Clipping the test-stride to match the stride-reserved size
        if y_pred.shape[1] != y.shape[1]:  # This only takes place when the reserved-stride is smaller than the test-stride
            y_pred = y_pred[:, :y.shape[1], :]  # Clips time from left to right, that is, from past to future
        # 6. Rollback min-max normalization
        y_pred, y = y_pred * scale, y * scale
        # 7. Rollback logarithm scaling
        if self.normalization_type == "logarithm":
            y_pred, y = torch.expm1(y_pred), torch.expm1(y)
        # 8. Computes the criterion and all pre-defined cost functions between the prediction and observation
        return self.criterion(y_pred, y, to_float), [cost(y_pred, y, to_float) for cost in self.cost_list]

    def verbose(self, epoch_duration=0., validation_duration=0., is_train=True):
        """
        Verbose epoch results on-the-fly by averaging the time and batch axis of the losses matrices.
        :param epoch_duration: float
            The duration of the last epoch (default: 0.).
        :param validation_duration: float
            The duration of the validation process on the stride-reserved set (default: 0.).
        :param is_train: boolean
            Verbose test information if the value is set to False (default: True).
        """
        # Calculates the mean and standard deviation of a given loss matrix
        calculate_metrics = lambda matrix: zip(matrix.mean(axis=(0, 1)), matrix.std(axis=(0, 1)))
        # Outputs the metrics for the current epoch in a printable format
        tuplenizer = lambda matrix, data_id: tuple(np.hstack(list(calculate_metrics(matrix[data_id, self.epoch_id]))))
        # The string of results containing user-selected cost functions
        cost_string = " - ".join([cost_function.name + ": %9.4f (+/- %9.4f)" for cost_function in self.cost_list])
        # Picking which data type of results will be printed out
        loss_matrix = self.trn_loss if is_train else self.tst_loss
        # Preparing the opening string
        opening = "Epoch #%04d" % self.epoch_id if is_train else "       Test"
        # String for the mandatory samples of the dataset
        trn_samples = ("{M}" if self.watch_axis == 0 else "[M]") + " %s" % (cost_string % tuplenizer(loss_matrix, 0))
        # String for the non-mandatory development samples
        res_samples = ""  # It will remain empty if no data is reserved
        if self.validation_samples > 0:# or self.k_fold > 0:  # True whenever we have the mandatory and sample-reserved sets
            res_samples = ("{R}" if self.watch_axis == 1 else "[R]") + " %s" % (cost_string % tuplenizer(loss_matrix, 1))
        # Verbosing the results to the STDOUT
        print("%s - duration: %06.2fs %s %s" % (opening, epoch_duration, trn_samples, res_samples))
        # Results from the stride-reserved set will be printed bellow the training results
        if is_train and self.validation_stride > 0:
            # String for the mandatory samples of the dataset
            dev_trn = ("{M}" if self.watch_axis == 2 else "[M]") + " %s" % (cost_string % tuplenizer(self.dev_loss, 0))
            # String for the non-mandatory development samples
            dev_res = ""  # It will remain empty if no data is reserved
            if self.validation_samples > 0:# or self.k_fold > 0:  # True whenever we have the mandatory and sample-reserved sets
                dev_res = ("{R}" if self.watch_axis == 3 else "[R]") + " %s" % (cost_string % tuplenizer(self.dev_loss, 1))
            # Verbosing the results to the STDOUT
            print("& Dev #%04d - duration: %06.2fs %s %s" % (self.epoch_id, validation_duration, dev_trn, dev_res))

    def train(self):
        """
        Adapted training routine for time-series forecasting.
        :return: float
            The average criterion loss between the test and development sets.
        """
        # The path to which the best model will be saved
        file_path = "%s/sandbox/.%s" % (self.working_directory, self.network_uuid)
        print("[TRAINING]", end="\n\n")  # Start training
        best_epoch = 0  # The epoch in which the best loss was observed
        early_stop_count = 0  # Controls when to stop training if the network stops improving
        test_every_count = 0  # Controls when to test the network in the training set
        try:  # Stop training at any time by hitting CTRL + C
            for self.epoch_id in range(self.epochs):
                # [START] ----------------------------------------------------------------------------------------------
                epoch_start = time.time()
                self.forward_index = 0  # Forward count
                set_losses = []  # Stores training losses
                dev_losses = []  # Stores development losses
                # [REGULAR TRAINING AND SAMPLE-DEVELOPMENT ROUTINE] ----------------------------------------------------
                s, d = np.inf, np.inf  # Initial losses - s will always be overwritten, but d might remain as it is
                # Iterating over the training set and collecting metrics whenever necessary
                for x_set, y_set, x_dev, y_dev in self.data_iterator(test_iteration=False, time_validation=False):
                    self.__train(x_set, y_set)  # >>>>>>>>>>>>>>> TRAINING TAKES PLACE ONLY IN HERE <<<<<<<<<<<<<<<<<<<<
                    # Evaluating the results on the main dataset
                    s, self.trn_loss[self.SET_ID, self.epoch_id, self.time_id, self.batch_id] = self.__test(x_set, y_set, sample_validation=False)
                    # Dealing with sample-generalization
                    if self.unknown_samples:
                        # Evaluating the results on sample-reserved data
                        d, self.trn_loss[self.DEV_ID, self.epoch_id, self.time_id, self.batch_id] = self.__test(x_dev, y_dev, sample_validation=True)
                    set_losses.append([s, d])
                # [TIME-DEVELOPMENT ROUTINE FOR ASSESSING TIME GENERALIZATION] -----------------------------------------
                if self.validation_stride > 0:
                    s, d = np.inf, np.inf  # Initial losses - s will always be overwritten, but d might remain as it is
                    validation_start = time.time()
                    # Iterating over the stride-reserved set to collect metrics on time-generalization
                    for x_set, y_set, x_dev, y_dev in self.data_iterator(test_iteration=True, time_validation=True):
                        # Evaluating the results on the main dataset (time-id is zero because we have only one time-batch)
                        s, self.dev_loss[self.SET_ID, self.epoch_id, 0, self.batch_id] = self.__test(x_set, y_set, sample_validation=False)
                        # Dealing with sample-generalization
                        if self.unknown_samples:
                            # Evaluating the results on the sample-reserved set (time-id is zero because we have only one time-batch)
                            d, self.dev_loss[self.DEV_ID, self.epoch_id, 0, self.batch_id] = self.__test(x_dev, y_dev, sample_validation=True)
                        dev_losses.append([s, d])
                    validation_end = time.time()
                # ------------------------------------------------------------------------------------------------------
                # Selecting the axis to which the early-stopping and learning-rate schedulers will be watching.
                # In case the value is higher than one, it will watch the stride-reserved set; otherwise, the training set.
                losses, index = (dev_losses, self.watch_axis - 2) if self.watch_axis > 1 else (set_losses, self.watch_axis)
                # Taking the mean of losses to schedule the training process
                mean_loss = np.array(losses)[:, index].mean()
                self.scheduler.step(mean_loss)  # updating loss
                # Updating scheduling counts
                test_every_count += 1  # Confirming epoch as completed
                early_stop_count += 1  # Registering improvement count
                epoch_end = time.time()
                # ------------------------------------------------------------------------------------------------------
                # Verbose the new best result we had just seen
                if mean_loss < self.min_loss:
                    print("Epoch #%04d - Found a better development loss of %.4f." % (self.epoch_id, mean_loss))
                # ------------------------------------------------------------------------------------------------------
                # Verbose the results of the current epoch
                if self.validation_stride > 0:
                    # There is an intermediate validation inside the epoch
                    validation_duration = validation_end - validation_start
                    epoch_duration = (validation_start - epoch_start) + (epoch_end - validation_end)
                    self.verbose(epoch_duration=epoch_duration, validation_duration=validation_duration)
                else:
                    # There is no validation loss
                    epoch_duration = epoch_end - epoch_start
                    self.verbose(epoch_duration=epoch_duration)
                # ------------------------------------------------------------------------------------------------------
                # Save the current model if it is the best one seen so far
                if mean_loss < self.min_loss:
                    best_epoch = self.epoch_id  # The epoch in which the best loss occurred
                    test_every_count = 0  # Resetting the count for testing the network
                    early_stop_count = 0  # Resetting the count for early stopping
                    self.test()  # Calling the network testing routine
                    # Saving all states to the disk as this is our best result
                    torch.save(self.network.state_dict(), "%s.network" % file_path)
                    torch.save(self.optimizer.state_dict(), "%s.optimizer" % file_path)
                    torch.save(self.scheduler.state_dict(), "%s.scheduler" % file_path)
                    self.min_loss = mean_loss  # Replaces the best loss
                    # [MEAN] -------------------------------------------------------------------------------------------
                    # Storing the best metrics for the current best model ==============================================
                    self.metric_storage[0] = self.trn_loss[self.SET_ID, self.epoch_id].mean(axis=(0, 1))  # Known samples
                    self.metric_storage[1] = self.trn_loss[self.DEV_ID, self.epoch_id].mean(axis=(0, 1))  # Unknown samples
                    # These cases might be straight zeros if the dataset has no reserved stride data ===================
                    self.metric_storage[2] = self.dev_loss[self.SET_ID, self.epoch_id].mean(axis=(0, 1))  # Known samples
                    self.metric_storage[3] = self.dev_loss[self.DEV_ID, self.epoch_id].mean(axis=(0, 1))  # Unknown samples
                    # Storing the metrics observed in the test set - metrics for the best model ========================
                    self.metric_storage[4] = self.tst_loss[self.SET_ID, self.epoch_id].mean(axis=(0, 1))  # Known samples
                    self.metric_storage[5] = self.tst_loss[self.DEV_ID, self.epoch_id].mean(axis=(0, 1))  # Unknown samples
                    # [STD] --------------------------------------------------------------------------------------------
                    # Storing the best metrics for the current best model===============================================
                    self.metric_deviation[0] = self.trn_loss[self.SET_ID, self.epoch_id].std(axis=(0, 1))  # Known samples
                    self.metric_deviation[1] = self.trn_loss[self.DEV_ID, self.epoch_id].std(axis=(0, 1))  # Unknown samples
                    # These cases might be straight zeros if the dataset has no reserved stride data ===================
                    self.metric_deviation[2] = self.dev_loss[self.SET_ID, self.epoch_id].std(axis=(0, 1))  # Known samples
                    self.metric_deviation[3] = self.dev_loss[self.DEV_ID, self.epoch_id].std(axis=(0, 1))  # Unknown samples
                    # Storing the metrics observed in the test set - metrics for the best model ========================
                    self.metric_deviation[4] = self.tst_loss[self.SET_ID, self.epoch_id].std(axis=(0, 1))  # Known samples
                    self.metric_deviation[5] = self.tst_loss[self.DEV_ID, self.epoch_id].std(axis=(0, 1))  # Unknown samples
                    # --------------------------------------------------------------------------------------------------
                # Testing the network on-the-fly after each epoch
                if self.test_every == test_every_count:
                    test_every_count = 0  # Resetting the count for testing the network
                    self.test()  # Calling the network testing routine
                # Stopping early as the network stopped improving
                if self.early_stop == early_stop_count:
                    # Will raise an exception handled bellow
                    raise KeyboardInterrupt
                # Save intermediate graphs to the disk whenever requested by the user
                if self.save_graph:
                    with torch.no_grad():
                        SE, TE = self.network.SourceEvolution, self.network.TargetEvolution
                        # Input Graph (pre-set or learned from the training data)
                        ss = self.network.SourceEvolution.adjacency
                        # Learning Representation (shared graph with the next layer)
                        sh = torch.addmm(SE.adjacency_bias, ss, SE.adjacency_weights)
                        # Hidden Source Graph
                        so = torch.mul(SE.cosine_similarity_matrix(sh), SE.cosine_weights) + SE.cosine_bias
                        # Graph Shared by the Source Evolution
                        ts = self.network.TargetEvolution.adjacency
                        # Learning Representation (not shared graph)
                        th = torch.addmm(TE.adjacency_bias, ts, TE.adjacency_weights)
                        # Hidden Target Graph
                        to = torch.mul(TE.cosine_similarity_matrix(th), TE.cosine_weights) + TE.cosine_bias
                        # Saving Graphs
                        torch.save(ss.cpu().numpy(), "%s/graph/%s SRC-S-%04d.pth" % (self.working_directory, self.file_prefix, self.epoch_id))
                        torch.save(sh.cpu().numpy(), "%s/graph/%s SRC-H-%04d.pth" % (self.working_directory, self.file_prefix, self.epoch_id))
                        torch.save(so.cpu().numpy(), "%s/graph/%s SRC-O-%04d.pth" % (self.working_directory, self.file_prefix, self.epoch_id))
                        torch.save(ts.cpu().numpy(), "%s/graph/%s TGT-S-%04d.pth" % (self.working_directory, self.file_prefix, self.epoch_id))
                        torch.save(th.cpu().numpy(), "%s/graph/%s TGT-H-%04d.pth" % (self.working_directory, self.file_prefix, self.epoch_id))
                        torch.save(to.cpu().numpy(), "%s/graph/%s TGT-O-%04d.pth" % (self.working_directory, self.file_prefix, self.epoch_id))
                # [END] ------------------------------------------------------------------------------------------------
        except KeyboardInterrupt:
            # Checking whether the training was interrupted by natural causes or not
            if self.early_stop != early_stop_count:
                # In this case, the last epoch is incomplete, so we need to discard its values
                self.trn_loss = self.trn_loss[:, 0:self.epoch_id]  # Training losses
                self.tst_loss = self.tst_loss[:, 0:self.epoch_id]  # Test on reserved data
                self.dev_loss = self.dev_loss[:, 0:self.epoch_id]  # Time development on stride-reserved data
                self.epoch_id -= 1  # Rollback to the last epoch
                print("The last epoch was discarded as it was incomplete.")
            print("Stopping the training early (...)")
        except:
            print(traceback.format_exc())  # Will when during debugging
            # Cleans temporary files in case of any broad exception
            for file_type in ["network", "optimizer", "scheduler"]:
                if os.path.exists("%s.%s" % (file_path, file_type)):
                    os.remove("%s.%s" % (file_path, file_type))
            exit(2020)  # Forcing training exit with code 2020
        # Restoring all states to the moment when we got the best result
        self.network.load_state_dict(torch.load("%s.network" % file_path))
        self.optimizer.load_state_dict(torch.load("%s.optimizer" % file_path))
        self.scheduler.load_state_dict(torch.load("%s.scheduler" % file_path))
        print("\n[TESTING]", end="\n\n")  # Verbosing again the results of the best epoch
        self.epoch_id = best_epoch  # Replacing the current epoch for the best epoch
        self.verbose(epoch_duration=0, is_train=False)  # Show the test results
        # Cleans temporary files without checking as all files exist
        for file_type in ["network", "optimizer", "scheduler"]:
            os.remove("%s.%s" % (file_path, file_type))
        # Returning final result to the main class
        return self.tst_loss[self.SET_ID, self.epoch_id, ..., 0].mean()

    def test(self):
        """
        Adapted testing routine for time-series forecasting.
        :return: float
            The average criterion loss between the test and development sets.
        """
        losses = []
        test_start = time.time()
        # Iterating over the test set and collecting metrics
        for x_set, y_set, x_dev, y_dev in self.data_iterator(test_iteration=True, time_validation=False):
            # Evaluating the results on the main dataset (time-id is zero because we have only one time-batch)
            s, self.tst_loss[self.SET_ID, self.epoch_id, 0, self.batch_id] = self.__test(x_set, y_set, sample_validation=False)
            losses.append(s)  # The criterion-loss on the main dataset
            # Dealing with sample-generalization
            if self.unknown_samples:
                # Evaluating the results on the sample-reserved set (time-id is zero because we have only one time-batch)
                d, self.tst_loss[self.DEV_ID, self.epoch_id, 0, self.batch_id] = self.__test(x_dev, y_dev, sample_validation=True)
                losses.append(d)  # The criterion-loss on the sample-reserved set
        test_duration = time.time() - test_start
        # Verbosing the average and standard deviation of the results
        self.verbose(epoch_duration=test_duration, is_train=False)
        # Averaging the criterion losses seen on the test set
        return np.array(losses).mean()