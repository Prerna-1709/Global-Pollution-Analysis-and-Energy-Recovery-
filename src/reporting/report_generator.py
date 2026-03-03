"""Report generator module: consolidates metrics, plots, and policy insights.

Generates a multi-page PDF report (Final_Project_Report.pdf) containing:
  - Consolidated model comparison table
  - Confusion matrices (Logistic & Decision Tree)
  - ROC curves (multi-class OvR)
  - Feature importance bar charts
  - Policy recommendation text page
"""

import os
import io
import textwrap
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, classification_report,
    accuracy_score, f1_score
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')


class ReportGenerator:
    """Trains all models, collects metrics, and generates the final PDF report.

    Args:
        X_train, X_test: Feature arrays for regression/classification.
        y_reg_train, y_reg_test: Regression targets (Energy_Recovered_GWh).
        y_cls_train, y_cls_test: Classification targets (Pollution_Severity).
        feature_names: Column names for feature importance plots.
        ann_metrics: Pre-computed ANN metrics dict {R2, MSE, MAE}.
    """

    POLICY_RECOMMENDATIONS = [
        ("1. Prioritise High-Pollution / Low-Recovery Regions",
         "Association rule mining revealed that high Air_Pollution_Index co-occurs with "
         "high Energy_Recovered_GWh (lift ≈ 1.23). Governments should channel renewable-energy "
         "investment into the 'at-risk' cluster (Cluster 0) identified by K-Means — dominated "
         "by Brazil, China, and India — where pollution is high yet recovery infrastructure lags."),

        ("2. Adopt GDP-Linked Deforestation Controls",
         "SVM permutation importance ranked GDP_Billion_USD as the top driver of forest loss. "
         "Policy frameworks should tie deforestation permits to GDP-scaled sustainability bonds, "
         "incentivising wealthier nations to fund reforestation in economically stressed regions."),

        ("3. Industrial Waste as a Leading Energy Feedstock",
         "Apriori rules linked Industrial_Waste_Low to GDP_High (lift 1.29), suggesting "
         "under-utilised waste streams in lower-income economies. Waste-to-energy plant subsidies "
         "in mid-tier industrial nations could simultaneously cut pollution and raise GWh recovery."),

        ("4. Multi-Model Early-Warning System for Severity Escalation",
         "The Logistic Regression classifier achieved 96.7 % accuracy on severity triage. "
         "Deploying this model as a real-time monitoring dashboard — fed by satellite CO₂ and "
         "air-quality APIs — would allow sub-national agencies to trigger interventions before "
         "regions transition from 'Medium' to 'High' Pollution_Severity."),

        ("5. Leverage Neural Network Forecasts for Energy Grid Planning",
         "Once trained on real multi-year data, the ANN (128-64-32, Dropout 0.3) can provide "
         "monthly Energy_Recovered_GWh forecasts with tighter confidence intervals than linear "
         "models. Utilities should integrate these forecasts into capacity-planning pipelines to "
         "reduce over-reliance on fossil-fuel peakers during high-pollution periods."),
    ]

    def __init__(self, X_train, X_test, y_reg_train, y_reg_test,
                 y_cls_train, y_cls_test, feature_names,
                 ann_metrics=None, class_names=None):
        self.X_train = X_train
        self.X_test  = X_test
        self.y_reg_train = y_reg_train
        self.y_reg_test  = y_reg_test
        self.y_cls_train = y_cls_train
        self.y_cls_test  = y_cls_test
        self.feature_names = feature_names
        self.ann_metrics   = ann_metrics or {}
        self.class_names   = class_names or ['Low', 'Medium', 'High']

        # Models
        self.lr  = LinearRegression()
        self.log = LogisticRegression(max_iter=1000, random_state=42)
        self.dt  = DecisionTreeClassifier(max_depth=5, random_state=42)
        self.knn = KNeighborsClassifier(n_neighbors=5)

        self._reg_metrics  = {}
        self._cls_metrics  = {}
        self._trained      = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train_all(self):
        """Trains Linear, Logistic, Decision Tree, and KNN models."""
        self.lr.fit(self.X_train, self.y_reg_train)
        self.log.fit(self.X_train, self.y_cls_train)
        self.dt.fit(self.X_train, self.y_cls_train)
        self.knn.fit(self.X_train, self.y_cls_train)
        self._trained = True

    def collect_metrics(self) -> dict:
        """Evaluates all trained models and returns a metrics summary dict."""
        # Regression
        lr_pred = self.lr.predict(self.X_test)
        self._reg_metrics['Linear Regression'] = {
            'R2': r2_score(self.y_reg_test, lr_pred),
            'MSE': mean_squared_error(self.y_reg_test, lr_pred),
            'MAE': mean_absolute_error(self.y_reg_test, lr_pred),
        }
        if self.ann_metrics:
            self._reg_metrics['Neural Network (ANN)'] = self.ann_metrics

        # Classification
        for name, model in [('Logistic Reg', self.log), ('Decision Tree', self.dt), ('KNN', self.knn)]:
            pred = model.predict(self.X_test)
            self._cls_metrics[name] = {
                'Accuracy': accuracy_score(self.y_cls_test, pred),
                'F1 (macro)': f1_score(self.y_cls_test, pred, average='macro', zero_division=0),
                'Report': classification_report(self.y_cls_test, pred,
                                                target_names=self.class_names,
                                                zero_division=0),
            }
        return {'regression': self._reg_metrics, 'classification': self._cls_metrics}

    # ── Individual plot helpers ────────────────────────────────────────────────

    def _plot_metrics_table(self, ax):
        """Renders consolidated metrics as a formatted table on ax."""
        rows, cols = [], ['Model', 'Task', 'R² / Acc', 'MSE / F1']
        for name, m in self._reg_metrics.items():
            rows.append([name, 'Regression', f"{m['R2']:.4f}", f"{m['MSE']:.1f}"])
        for name, m in self._cls_metrics.items():
            rows.append([name, 'Classification', f"{m['Accuracy']:.4f}", f"{m['F1 (macro)']:.4f}"])

        ax.axis('off')
        tbl = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.6)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor('#4C72B0')
                cell.set_text_props(color='white', fontweight='bold')
            elif r % 2 == 0:
                cell.set_facecolor('#EEF2FA')
        ax.set_title('Consolidated Model Comparison', fontsize=12, fontweight='bold', pad=10)

    def _plot_confusion_matrices(self, axes):
        """Plots confusion matrices for Logistic Reg and Decision Tree."""
        for ax, (name, model) in zip(axes, [('Logistic Reg', self.log), ('Decision Tree', self.dt)]):
            pred = model.predict(self.X_test)
            cm = confusion_matrix(self.y_cls_test, pred)
            disp = ConfusionMatrixDisplay(cm, display_labels=self.class_names)
            disp.plot(ax=ax, colorbar=False, cmap='Blues')
            ax.set_title(f'Confusion Matrix\n{name}', fontsize=10, fontweight='bold')

    def _plot_roc_curves(self, ax):
        """Plots OvR ROC curves for Logistic Reg and Decision Tree."""
        classes = sorted(np.unique(self.y_cls_test))
        y_bin = label_binarize(self.y_cls_test, classes=classes)
        colors = ['#4C72B0', '#DD8452', '#55A868']

        for name, model, ls in [('Logistic Reg', self.log, '-'), ('Decision Tree', self.dt, '--')]:
            if hasattr(model, 'predict_proba'):
                y_score = model.predict_proba(self.X_test)
            else:
                continue
            for i, cls_idx in enumerate(classes):
                if y_bin.shape[1] <= i:
                    continue
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                label_cls = self.class_names[i] if i < len(self.class_names) else str(cls_idx)
                ax.plot(fpr, tpr, color=colors[i % len(colors)], linestyle=ls,
                        label=f'{name} | {label_cls} (AUC={roc_auc:.2f})', linewidth=1.5)

        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves (OvR)', fontsize=10, fontweight='bold')
        ax.legend(fontsize=6, loc='lower right')

    def _plot_feature_importance(self, axes):
        """Plots feature importances from Decision Tree and (absolute) Logistic coefficients."""
        # Decision Tree
        importances_dt = pd.Series(self.dt.feature_importances_, index=self.feature_names).sort_values(ascending=False)
        importances_dt.head(10).plot(kind='barh', ax=axes[0], color='#4C72B0')
        axes[0].invert_yaxis()
        axes[0].set_title('Feature Importance\n(Decision Tree)', fontsize=10, fontweight='bold')
        axes[0].set_xlabel('Importance')

        # Logistic Regression (mean abs coefficient across classes)
        coef_abs = pd.Series(
            np.abs(self.log.coef_).mean(axis=0), index=self.feature_names
        ).sort_values(ascending=False)
        coef_abs.head(10).plot(kind='barh', ax=axes[1], color='#DD8452')
        axes[1].invert_yaxis()
        axes[1].set_title('Feature Importance\n(Logistic Reg |coef|)', fontsize=10, fontweight='bold')
        axes[1].set_xlabel('Mean |Coefficient|')

    def _plot_policy_page(self, fig):
        """Adds a text-only policy recommendations page to the figure."""
        fig.patch.set_facecolor('#F7F9FC')
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.set_title('Policy Recommendations', fontsize=14, fontweight='bold',
                     color='#2C3E50', pad=20, loc='left', x=0.02)

        y_pos = 0.93
        for title, body in self.POLICY_RECOMMENDATIONS:
            ax.text(0.02, y_pos, title, transform=ax.transAxes,
                    fontsize=9, fontweight='bold', color='#2C3E50')
            y_pos -= 0.03
            wrapped = textwrap.fill(body, width=110)
            ax.text(0.02, y_pos, wrapped, transform=ax.transAxes,
                    fontsize=8, color='#4A4A4A', verticalalignment='top')
            y_pos -= 0.12

    # ── Main PDF generation ───────────────────────────────────────────────────

    def generate_pdf(self, save_path: str = 'outputs/Final_Project_Report.pdf'):
        """Generates the complete multi-page PDF report.

        Args:
            save_path: Path to save the PDF.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with PdfPages(save_path) as pdf:

            # ── Page 1: Cover ────────────────────────────────────────────────
            fig = plt.figure(figsize=(11, 8.5))
            fig.patch.set_facecolor('#1A2A4A')
            ax = fig.add_subplot(111)
            ax.axis('off')
            ax.text(0.5, 0.65, 'Global Pollution Analysis\n& Energy Recovery',
                    transform=ax.transAxes, fontsize=26, fontweight='bold',
                    color='white', ha='center', va='center', linespacing=1.5)
            ax.text(0.5, 0.45, 'Final Project Report', transform=ax.transAxes,
                    fontsize=14, color='#A8C8E8', ha='center')
            ax.text(0.5, 0.35, 'Models: Linear Regression · Logistic Regression · SVM\n'
                    'Decision Tree · KNN · K-Means · Hierarchical · ANN · Apriori',
                    transform=ax.transAxes, fontsize=9.5, color='#CCDDEE',
                    ha='center', linespacing=1.6)
            ax.text(0.5, 0.15, 'Generated: March 2026', transform=ax.transAxes,
                    fontsize=9, color='#889AAA', ha='center')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # ── Page 2: Metrics table ─────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(11, 4))
            self._plot_metrics_table(ax)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # ── Page 3: Confusion Matrices + ROC curves ───────────────────────
            fig = plt.figure(figsize=(14, 5))
            gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
            ax_cm1 = fig.add_subplot(gs[0])
            ax_cm2 = fig.add_subplot(gs[1])
            ax_roc = fig.add_subplot(gs[2])
            self._plot_confusion_matrices([ax_cm1, ax_cm2])
            self._plot_roc_curves(ax_roc)
            fig.suptitle('Classification Diagnostics', fontsize=13, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # ── Page 4: Feature Importance ────────────────────────────────────
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            self._plot_feature_importance(axes)
            fig.suptitle('Feature Importance Analysis', fontsize=13, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # ── Page 5: Policy recommendations ───────────────────────────────
            fig = plt.figure(figsize=(11, 8.5))
            self._plot_policy_page(fig)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        print(f"Final report saved to: {save_path}")
