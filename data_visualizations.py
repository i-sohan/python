"""
Complete Data Visualization Guide for Data Scientists
======================================================
Covers Univariate, Bivariate, Multivariate, ML Evaluation, and Dashboard Layouts,
including detailed explanations of what each plot is, when to use it, and key insights.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import confusion_matrix

# Global styling setup
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["font.size"] = 10
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# Synthetic Dataset Generation
# ---------------------------------------------------------
np.random.seed(42)
n_samples = 200

df = pd.DataFrame({
    "Age": np.random.randint(22, 60, n_samples),
    "Experience": np.random.uniform(0, 35, n_samples).round(1),
    "Department": np.random.choice(["Data Science", "Engineering", "Marketing", "Sales"], n_samples),
    "Education": np.random.choice(["Bachelor", "Master", "PhD"], n_samples, p=[0.5, 0.35, 0.15]),
    "PerformanceScore": np.random.normal(75, 10, n_samples).round(1)
})
df["Salary"] = (30000 + df["Experience"] * 3500 + np.random.normal(0, 8000, n_samples)).round(-2)
df.loc[0, "Salary"] = 190000  # Outlier 1
df.loc[1, "Salary"] = 210000  # Outlier 2


# ---------------------------------------------------------
# 1. UNIVARIATE ANALYSIS (Single Variable)
# ---------------------------------------------------------
def plot_univariate():
    """
    Univariate analysis examines the distribution, spread, central tendency,
    outliers, and frequencies of a single feature in isolation.
    """

    # 1.1 Histogram & KDE Plot
    # - WHAT IT IS: Groups continuous data into bins + fits a smooth probability density curve.
    # - WHEN TO USE: Checking feature distribution shape (Normal/Gaussian, Skewed, Bimodal).
    # - WHAT TO LOOK FOR: Skewness direction and multi-modal distribution peaks.
    plt.figure(figsize=(9, 5))
    sns.histplot(df["Salary"], kde=True, color="dodgerblue", bins=20)
    plt.title("1.1 Histogram & KDE: Salary Distribution", fontweight="bold")
    plt.show()

    # 1.2 Box Plot
    # - WHAT IT IS: Displays Q1, Median (Q2), Q3, and points beyond 1.5 * IQR as outliers.
    # - WHEN TO USE: Identifying extreme outliers and assessing dataset skewness.
    # - WHAT TO LOOK FOR: Points plotted beyond the whiskers indicate severe outliers.
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df["Salary"], color="tomato")
    plt.title("1.2 Box Plot: Salary Outlier Detection", fontweight="bold")
    plt.show()

    # 1.3 Violin Plot
    # - WHAT IT IS: Combines a box plot with a rotated KDE density plot on each side.
    # - WHEN TO USE: Comparing distributions across categories with density shapes visible.
    # - WHAT TO LOOK FOR: Bulges showing density concentration peaks in each group.
    plt.figure(figsize=(9, 5))
    sns.violinplot(x="Department", y="Salary", data=df, palette="muted", inner="quartile")
    plt.title("1.3 Violin Plot: Salary Distribution by Department", fontweight="bold")
    plt.show()

    # 1.4 Count Plot
    # - WHAT IT IS: Bar chart showing frequencies of categorical categories.
    # - WHEN TO USE: Checking category frequencies and detecting class imbalance.
    # - WHAT TO LOOK FOR: Significantly underrepresented categories.
    plt.figure(figsize=(8, 4))
    sns.countplot(x="Department", data=df, palette="viridis", order=df["Department"].value_counts().index)
    plt.title("1.4 Count Plot: Employee Count by Department", fontweight="bold")
    plt.show()

    # 1.5 Pie Chart
    # - WHAT IT IS: Displays proportions as slices of a circle totaling 100%.
    # - WHEN TO USE: Showing percentage share for small categorical groups (<= 5).
    # - WHAT TO LOOK FOR: Dominant segment shares.
    edu_counts = df["Education"].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(edu_counts, labels=edu_counts.index, autopct="%1.1f%%", startangle=140, colors=sns.color_palette("pastel")[:3])
    plt.title("1.5 Pie Chart: Education Level Proportion", fontweight="bold")
    plt.show()


# ---------------------------------------------------------
# 2. BIVARIATE ANALYSIS (Two Variables)
# ---------------------------------------------------------
def plot_bivariate():
    """
    Bivariate analysis explores correlations, trends, dependencies, and comparative
    metrics between two features.
    """

    # 2.1 Scatter Plot
    # - WHAT IT IS: Plots data points on a 2D Cartesian plane (X vs Y).
    # - WHEN TO USE: Identifying correlation direction (positive/negative), strength, and clusters.
    # - WHAT TO LOOK FOR: Linear/non-linear slope trends and clustered points.
    plt.figure(figsize=(9, 5))
    sns.scatterplot(x="Experience", y="Salary", hue="Education", size="Age", data=df, palette="deep", alpha=0.85)
    plt.title("2.1 Scatter Plot: Experience vs Salary", fontweight="bold")
    plt.show()

    # 2.2 Line Plot
    # - WHAT IT IS: Connects consecutive data points ordered sequentially (e.g. over time).
    # - WHEN TO USE: Time-series analysis, tracking trends, growth, and training loss over epochs.
    # - WHAT TO LOOK FOR: Upward/downward trends, seasonality, spikes, and drops.
    months = pd.date_range(start="2025-01-01", periods=12, freq="ME")
    sales = np.array([120, 135, 148, 142, 160, 175, 190, 185, 210, 225, 240, 260])
    plt.figure(figsize=(10, 4))
    plt.plot(months, sales, marker="o", color="mediumseagreen", linewidth=2.5)
    plt.title("2.2 Line Plot: Monthly Revenue Trend", fontweight="bold")
    plt.show()

    # 2.3 Grouped Bar Plot
    # - WHAT IT IS: Displays aggregate metric (mean/sum) across one or more categories.
    # - WHEN TO USE: Comparing group averages across sub-categories.
    # - WHAT TO LOOK FOR: Highest and lowest performing category combinations.
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Department", y="Salary", hue="Education", data=df, errorbar=None, palette="Set2")
    plt.title("2.3 Bar Plot: Average Salary by Department & Education", fontweight="bold")
    plt.show()

    # 2.4 Joint Plot
    # - WHAT IT IS: Bivariate scatter/regression plot combined with 1D marginal histograms.
    # - WHEN TO USE: Simultaneous inspection of joint trend and individual variable distributions.
    # - WHAT TO LOOK FOR: Regression line alignment with peak distribution densities.
    g = sns.jointplot(x="Experience", y="Salary", data=df, kind="reg", color="teal", height=6)
    g.fig.suptitle("2.4 Joint Plot: Bivariate Regression with Marginals", y=1.02, fontweight="bold")
    plt.show()


# ---------------------------------------------------------
# 3. MULTIVARIATE & CORRELATION ANALYSIS
# ---------------------------------------------------------
def plot_multivariate():
    """
    Multivariate analysis uncovers complex feature interactions, correlations,
    and groupings across multiple dimensions simultaneously.
    """

    # 3.1 Correlation Heatmap
    # - WHAT IT IS: Matrix displaying Pearson correlation coefficients (-1 to +1) using color.
    # - WHEN TO USE: Feature selection & detecting multicollinearity before linear modeling.
    # - WHAT TO LOOK FOR: High correlation (> 0.7 or < -0.7) indicating redundant features.
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("3.1 Correlation Heatmap", fontweight="bold")
    plt.show()

    # 3.2 Pair Plot
    # - WHAT IT IS: Grid of scatter plots for all numerical variable pairs + diagonal histograms.
    # - WHEN TO USE: Rapid initial EDA to spot feature correlations and class separability.
    # - WHAT TO LOOK FOR: Feature pairs that separate target classes cleanly.
    sns.pairplot(df[["Age", "Experience", "Salary", "PerformanceScore", "Education"]], hue="Education", palette="Set1")
    plt.suptitle("3.2 Pair Plot: All Pairwise Numeric Relationships", y=1.02, fontweight="bold")
    plt.show()

    # 3.3 FacetGrid
    # - WHAT IT IS: Grid of subpanels conditioned on categorical column values.
    # - WHEN TO USE: Comparing complex relationships across subgroups without plot clutter.
    # - WHAT TO LOOK FOR: Differences in relationships (slopes, spreads) across subpanels.
    g = sns.FacetGrid(df, col="Department", hue="Education", height=4, aspect=1)
    g.map(sns.scatterplot, "Experience", "Salary", alpha=0.7)
    g.add_legend()
    g.fig.suptitle("3.3 FacetGrid: Salary vs Experience by Department", y=1.05, fontweight="bold")
    plt.show()


# ---------------------------------------------------------
# 4. ML & MODEL EVALUATION VISUALIZATIONS
# ---------------------------------------------------------
def plot_model_evaluation():
    """
    Model evaluation plots diagnose regression and classification models,
    feature importance, and mathematical modeling assumptions.
    """

    # 4.1 Confusion Matrix Heatmap
    # - WHAT IT IS: Grid mapping True/False Positives/Negatives (TN, FP, FN, TP).
    # - WHEN TO USE: Evaluating classification model performance beyond raw accuracy.
    # - WHAT TO LOOK FOR: High diagonal values (correct predictions) vs off-diagonal errors.
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0]
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Negative (0)", "Positive (1)"], yticklabels=["Negative (0)", "Positive (1)"])
    plt.title("4.1 Confusion Matrix Heatmap", fontweight="bold")
    plt.show()

    # 4.2 Feature Importance Plot
    # - WHAT IT IS: Horizontal bar chart ranking feature contributions in tree models.
    # - WHEN TO USE: Model explainability and removing uninformative features.
    # - WHAT TO LOOK FOR: Top features driving model decisions.
    features = ["Experience", "PerformanceScore", "Age", "Department_Sales", "Education_PhD"]
    importance = [0.45, 0.25, 0.15, 0.10, 0.05]
    plt.figure(figsize=(9, 4))
    plt.barh(features[::-1], importance[::-1], color="darkcyan")
    plt.title("4.2 Feature Importance Plot", fontweight="bold")
    plt.show()

    # 4.3 ROC Curve
    # - WHAT IT IS: Plots True Positive Rate vs False Positive Rate across thresholds.
    # - WHEN TO USE: Evaluating classifier discrimination power independent of threshold.
    # - WHAT TO LOOK FOR: AUC score (closer to 1.0 is better; 0.5 is random chance).
    fpr = np.array([0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0])
    tpr = np.array([0.0, 0.55, 0.75, 0.88, 0.94, 0.98, 1.0])
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="Model (AUC = 0.89)")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Chance (AUC = 0.50)")
    plt.title("4.3 ROC Curve Classification Performance", fontweight="bold")
    plt.legend(loc="lower right")
    plt.show()

    # 4.4 Residual Plot
    # - WHAT IT IS: Plots predicted values vs prediction error residuals (actual - predicted).
    # - WHEN TO USE: Verifying linear regression homoscedasticity (constant error variance).
    # - WHAT TO LOOK FOR: Random scatter around Y=0 without funnels, curves, or trends.
    residuals = df["Salary"] - (30000 + df["Experience"] * 3500)
    plt.figure(figsize=(9, 5))
    sns.residplot(x=df["Salary"], y=residuals, lowess=True, color="purple", scatter_kws={"alpha": 0.6})
    plt.title("4.4 Residual Plot: Fitted Values vs Residuals", fontweight="bold")
    plt.axhline(0, color="black", linestyle="--")
    plt.show()


# ---------------------------------------------------------
# 5. DASHBOARD GRID SUBPLOTS
# ---------------------------------------------------------
def plot_dashboard():
    """
    Dashboard layouts arrange complementary plots into a single multi-panel figure
    grid for executive presentations and reporting.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.histplot(df["Salary"], kde=True, ax=axes[0, 0], color="skyblue")
    axes[0, 0].set_title("1. Salary Distribution")

    sns.scatterplot(x="Experience", y="Salary", hue="Education", data=df, ax=axes[0, 1])
    axes[0, 1].set_title("2. Experience vs Salary")

    sns.boxplot(x="Department", y="Salary", data=df, ax=axes[1, 0], palette="pastel")
    axes[1, 0].set_title("3. Salary by Department")

    sns.countplot(x="Education", data=df, ax=axes[1, 1], palette="magma")
    axes[1, 1].set_title("4. Education Breakdown")

    plt.tight_layout()
    plt.suptitle("Data Science EDA Dashboard", y=1.02, fontsize=16, fontweight="bold")
    plt.show()


if __name__ == "__main__":
    plot_univariate()
    plot_bivariate()
    plot_multivariate()
    plot_model_evaluation()
    plot_dashboard()
