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


class MAELoss(torch.nn.Module):
    """
    Mean Absolute Error (MAE)
    """

    def __init__(self):
        super(MAELoss, self).__init__()
        self.MAELoss = torch.nn.L1Loss()
        self.name = "MAE"

    def forward(self, y_pred, y_true, to_numpy=False):
        """
        Calculating the metric loss.
        :param y_pred: array-like of shape (n_nodes) or (n_nodes, n_samples)
            Predicted values - i.e., network output.
        :param y_true: array-like of shape (n_nodes) or (n_nodes, n_samples)
            Observed values - i.e., ground truth.
        :param to_numpy: boolean
            Whether to return the result as a NumPy variable.
        :return: float or tensor
            The loss between the predicted and observed values.
        """
        assert (y_pred.shape == y_true.shape), "Shape mismatch."
        loss = self.MAELoss(y_pred, y_true)
        return float(loss) if to_numpy else loss


class MALELoss(torch.nn.Module):
    """
    Mean Absolute Logarithm Error (MALE)
    """

    def __init__(self):
        super(MALELoss, self).__init__()
        self.MAELoss = torch.nn.L1Loss()
        self.name = "MALE"

    def forward(self, y_pred, y_true, to_numpy=False):
        """
        Calculating the metric loss.
        :param y_pred: array-like of shape (n_nodes) or (n_nodes, n_samples)
            Predicted values - i.e., network output.
        :param y_true: array-like of shape (n_nodes) or (n_nodes, n_samples)
            Observed values - i.e., ground truth.
        :param to_numpy: boolean
            Whether to return the result as a NumPy variable.
        :return: float or tensor
            The loss between the predicted and observed values.
        """
        assert (y_pred.shape == y_true.shape), "Shape mismatch."
        loss = self.MAELoss(torch.log1p(y_pred), torch.log1p(y_true))
        return float(loss) if to_numpy else loss


class MSELoss(torch.nn.Module):
    """
    Mean Squared Error (MSE)
    """

    def __init__(self):
        super(MSELoss, self).__init__()
        self.MSELoss = torch.nn.MSELoss()
        self.name = "MSE"

    def forward(self, y_pred, y_true, to_numpy=False):
        """
        Calculating the metric loss.
        :param y_pred: array-like of shape (n_nodes) or (n_nodes, n_samples)
            Predicted values - i.e., network output.
        :param y_true: array-like of shape (n_nodes) or (n_nodes, n_samples)
            Observed values - i.e., ground truth.
        :param to_numpy: boolean
            Whether to return the result as a NumPy variable.
        :return: float or tensor
            The loss between the predicted and observed values.
        """
        assert (y_pred.shape == y_true.shape), "Shape mismatch."
        loss = self.MSELoss(y_pred, y_true)
        return float(loss) if to_numpy else loss


class MSLELoss(torch.nn.Module):
    """
    Mean Squared Logarithmic Error (MSLE)
    """

    def __init__(self):
        super(MSLELoss, self).__init__()
        self.MSELoss = torch.nn.MSELoss()
        self.name = "MSLE"

    def forward(self, y_pred, y_true, to_numpy=False):
        """
        Calculating the metric loss.
        :param y_pred: array-like of shape (n_nodes) or (n_nodes, n_samples)
            Predicted values - i.e., network output.
        :param y_true: array-like of shape (n_nodes) or (n_nodes, n_samples)
            Observed values - i.e., ground truth.
        :param to_numpy: boolean
            Whether to return the result as a NumPy variable.
        :return: float or tensor
            The loss between the predicted and observed values.
        """
        assert (y_pred.shape == y_true.shape), "Shape mismatch."
        loss = self.MSELoss(torch.log1p(y_pred), torch.log1p(y_true))
        return float(loss) if to_numpy else loss


class RMSELoss(torch.nn.Module):
    """
    Root Mean Squared Error (RMSE)
    """

    def __init__(self):
        super(RMSELoss, self).__init__()
        self.MSELoss = torch.nn.MSELoss()
        self.name = "RMSE"

    def forward(self, y_pred, y_true, to_numpy=False):
        """
        Calculating the metric loss.
        :param y_pred: array-like of shape (n_nodes) or (n_nodes, n_samples)
            Predicted values - i.e., network output.
        :param y_true: array-like of shape (n_nodes) or (n_nodes, n_samples)
            Observed values - i.e., ground truth.
        :param to_numpy: boolean
            Whether to return the result as a NumPy variable.
        :return: float or tensor
            The loss between the predicted and observed values.
        """
        assert (y_pred.shape == y_true.shape), "Shape mismatch."
        loss = torch.sqrt(self.MSELoss(y_pred, y_true))
        return float(loss) if to_numpy else loss


class RMSLELoss(torch.nn.Module):
    """
    Root Mean Squared Logarithmic Error (RMSLE)
    """

    def __init__(self):
        super(RMSLELoss, self).__init__()
        self.MSELoss = torch.nn.MSELoss()
        self.name = "RMSLE"

    def forward(self, y_pred, y_true, to_numpy=False):
        """
        Calculating the metric loss.
        :param y_pred: array-like of shape (n_nodes) or (n_nodes, n_samples)
            Predicted values - i.e., network output.
        :param y_true: array-like of shape (n_nodes) or (n_nodes, n_samples)
            Observed values - i.e., ground truth.
        :param to_numpy: boolean
            Whether to return the result as a NumPy variable.
        :return: float or tensor
            The loss between the predicted and observed values.
        """
        assert (y_pred.shape == y_true.shape), "Shape mismatch."
        loss = torch.sqrt(self.MSELoss(torch.log1p(y_pred), torch.log1p(y_true)))
        return float(loss) if to_numpy else loss


class R2Score(torch.nn.Module):
    """
    Coefficient of Determination (R2Score)
    """

    def __init__(self):
        super(R2Score, self).__init__()
        self.name = "R2Score"

    def forward(self, y_pred, y_true, to_numpy=False, eps=1e-08):
        """
        Calculating the metric loss.
        :param y_pred: array-like of shape (n_nodes) or (n_nodes, n_samples)
            Predicted values - i.e., network output.
        :param y_true: array-like of shape (n_nodes) or (n_nodes, n_samples)
            Observed values - i.e., ground truth.
        :param to_numpy: boolean
            Whether to return the result as a NumPy variable.
        :param eps: float
            Small value to avoid division by zero.
        :return: float or tensor
                The loss between the predicted and observed values.
        """
        assert (y_pred.shape == y_true.shape), "Shape mismatch."
        numerator = torch.sum(torch.pow(torch.sub(y_true, y_pred), 2), dim=-2, dtype=torch.float64)
        denominator = torch.sum(torch.pow(torch.sub(y_true, y_true.mean(dim=0)), 2), dim=-2, dtype=torch.float64)
        loss = torch.mean(torch.sub(1.0, torch.div(numerator + eps, denominator + eps)), dtype=torch.float64)
        return float(loss) if to_numpy else loss
