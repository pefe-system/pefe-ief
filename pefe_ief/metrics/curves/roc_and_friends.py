import numpy as np
from numpy import ndarray
from typing import Callable
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
import os

class ROCAndFriendsPlotter:
    def __init__(self, figure_image_file_prefix, y_test, y_probs, autoshow=False, autosave=True, roc_curve_func=roc_curve):
        # type: (str, ndarray, ndarray, bool, bool,      Callable[[ndarray, ndarray], tuple[float, float, ndarray]]     ) -> None
        self.figure_image_file_prefix = figure_image_file_prefix
        self.y_test = y_test
        self.y_probs = y_probs
        self.autoshow = autoshow
        self.autosave = autosave
        self.roc_curve_func = roc_curve_func

        fpr, tpr, thresholds = roc_curve_func(y_test, y_probs)
        fpr = np.array(fpr)
        tpr = np.array(tpr)
        roc_auc = auc(fpr, tpr)

        self.fpr = fpr
        self.tpr = tpr
        self.roc_auc = roc_auc
        self.fnr = 1 - tpr # tpr + fnr = 1
        self.tnr = 1 - fpr # tnr + fpr = 1
        self.thresholds = thresholds
    
    def plot_ROC(self):
        """
        "Receiver operating characteristic"
        Returns the path to the rendered file, if any
        """
        figure_image_file_prefix = self.figure_image_file_prefix
        fpr = self.fpr
        tpr = self.tpr
        roc_auc = self.roc_auc

        ROC_FILE_PATH = os.path.abspath(f"{figure_image_file_prefix}_IEF_ROC.png")

        plt.figure()
        plt.plot(fpr, tpr, color='blue', label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # random chance line

        plt.xscale("log")
        plt.yscale("log")
        ticks = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        plt.xticks(ticks, [r"$10^{-6}$", r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"])
        plt.yticks(ticks, [r"$10^{-6}$", r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"])

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")

        if self.autosave:
            plt.savefig(ROC_FILE_PATH, dpi=300, bbox_inches='tight')

        if self.autoshow:
            plt.show()
        
        return ROC_FILE_PATH if self.autosave else None
    
    def plot_DET(self):
        """
        "Detection Error Tradeoff curve"
        """
        figure_image_file_prefix = self.figure_image_file_prefix
        fpr = self.fpr
        fnr = self.fnr

        DET_FILE_PATH = os.path.abspath(f"{figure_image_file_prefix}_IEF_DET.png")

        plt.figure()
        plt.plot(fpr, fnr, color='blue', label=f"DET curve")
        plt.plot([0, 1], [1, 0], color="gray", linestyle="--")  # random chance line

        plt.xscale("log")
        plt.yscale("log")
        ticks = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        plt.xticks(ticks, [r"$10^{-6}$", r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"])
        plt.yticks(ticks, [r"$10^{-6}$", r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"])

        plt.xlabel("False Positive Rate")
        plt.ylabel("False Negative Rate")
        plt.title("DET Curve")
        plt.legend(loc="lower right")

        if self.autosave:
            plt.savefig(DET_FILE_PATH, dpi=300, bbox_inches='tight')

        if self.autoshow:
            plt.show()

        return DET_FILE_PATH if self.autosave else None
    
    def plot_Actual_Positives(self):
        """Actual positives"""
        figure_image_file_prefix = self.figure_image_file_prefix
        tpr = self.tpr
        fnr = self.fnr
        thresholds = self.thresholds

        FILE_PATH = os.path.abspath(f"{figure_image_file_prefix}_IEF_TPR_FNR_per_threshold_aka_Actual_Positives.png")

        plt.figure()
        plt.plot(thresholds, tpr, color="blue", label="TPR")
        plt.plot(thresholds, fnr, color="orange", label="FNR")
        plt.xlabel("Thresholds")
        plt.title("Actual Positives")
        plt.legend(loc="lower right")

        if self.autosave:
            plt.savefig(FILE_PATH, dpi=300, bbox_inches='tight')
        
        if self.autoshow:
            plt.show()
        
        return FILE_PATH if self.autosave else None
    
    def plot_Actual_Negatives(self):
        """Actual negatives"""
        figure_image_file_prefix = self.figure_image_file_prefix
        tnr = self.tnr
        fpr = self.fpr
        thresholds = self.thresholds

        FILE_PATH = os.path.abspath(f"{figure_image_file_prefix}_IEF_TNR_FFR_per_threshold_aka_Actual_Negatives.png")

        plt.figure()
        plt.plot(thresholds, tnr, color="blue", label="TNR")
        plt.plot(thresholds, fpr, color="orange", label="FPR")
        plt.xlabel("Thresholds")
        plt.title("Actual Negatives")
        plt.legend(loc="lower right")

        if self.autosave:
            plt.savefig(FILE_PATH, dpi=300, bbox_inches='tight')
        
        if self.autoshow:
            plt.show()
        
        return FILE_PATH if self.autosave else None
    
    def get_numerical_stats(self):
        return {
            "roc_auc": float(self.roc_auc),
        }

def roc_and_friends(*args, **kwargs):
    # type: (...) -> dict[str, dict[str, int|float|str]]

    r = ROCAndFriendsPlotter(*args, **kwargs)

    return {
        "curves": [
            {
                "type": "ROC",
                "plot_path": r.plot_ROC(),
            },

            {
                "type": "DET",
                "plot_path": r.plot_DET(),
            },

            {
                "type": "Actual Positives",
                "plot_path": r.plot_Actual_Positives(),
            },

            {
                "type": "Actual Negatives",
                "plot_path": r.plot_Actual_Negatives(),
            }
        ],

        "stats": r.get_numerical_stats(),
    }
