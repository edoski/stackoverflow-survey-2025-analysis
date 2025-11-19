"""
Simple linear regression analysis of work experience vs. compensation.

Performs simple linear regression predicting CompTotal (EUR) from WorkExp
(years of experience) using EUR-currency respondents from Stack Overflow survey.

Analysis includes:
- Train/test split for generalization assessment
- Overall model for EUR subset
- Per-country models for top 5 countries (by sample size)
- Regression diagnostics: residual plots, QQ-plots, Shapiro-Wilk normality test
- Model metrics: R², MSE, RMSE, coefficients
"""
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import shapiro
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from preprocessing import preprocess_eur

# Regression constants
TEST_SIZE = 0.2  # 20% for test set
RANDOM_STATE = 42
PLOT_DPI = 150
SCATTER_ALPHA = 0.3
SCATTER_SIZE = 16
TOP_N_COUNTRIES = 5
IQR_MULTIPLIER = 1.5  # Standard multiplier for IQR outlier detection
FIGSIZE_LARGE = (10, 7)
FIGSIZE_MEDIUM = (9, 6)
FIGSIZE_QQPLOT = (8, 8)


def main():
    """Execute simple linear regression analysis on EUR compensation data."""
    # Caricamento dati
    df = pd.read_csv("survey_results_public.csv", low_memory=False)

    # Pre-processing (NaN, EUR, plausibilità basilare)
    data = preprocess_eur(df)

    plots_reg_dir = Path("plots_regression")
    plots_reg_dir.mkdir(exist_ok=True)

    X = data[["WorkExp"]].values
    Y = data["CompTotal"].values

    # Train/test split for generalization assessment
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Regressione lineare semplice (SLR)
    lr = LinearRegression()
    lr.fit(X_train, y_train)  # fit on training data

    # Predictions
    yhat_train = lr.predict(X_train)
    yhat_test = lr.predict(X_test)

    # Metriche di performance - Training set
    r2_train = r2_score(y_train, yhat_train)
    mse_train = mean_squared_error(y_train, yhat_train)
    rmse_train = float(np.sqrt(mse_train))

    print("=== Training Set Metrics ===")
    print(f"R^2 = {r2_train:.3f}")
    print(f"MSE = {mse_train:.1f}")
    print(f"RMSE = {rmse_train:.1f}")

    # Metriche di performance - Test set
    r2_test = r2_score(y_test, yhat_test)
    mse_test = mean_squared_error(y_test, yhat_test)
    rmse_test = float(np.sqrt(mse_test))

    print("\n=== Test Set Metrics (Generalization) ===")
    print(f"R^2 = {r2_test:.3f}")
    print(f"MSE = {mse_test:.1f}")
    print(f"RMSE = {rmse_test:.1f}")
    print(f"Intercept = {lr.intercept_:.2f}  |  Slope = {lr.coef_[0]:.2f}  (EUR per anno)")

    # Grafico con retta (using full dataset for visualization, model fitted on train)
    yhat_full = lr.predict(X)  # Predict on full X for plotting
    plt.figure(figsize=FIGSIZE_LARGE)
    plt.scatter(X, Y, alpha=SCATTER_ALPHA, s=SCATTER_SIZE, label="Data", color="C0")
    plt.plot(X, yhat_full, color="C1", linewidth=2, label="Fitted line (trained on 80%)")
    plt.xlabel("WorkExp (anni)")
    plt.ylabel("CompTotal (EUR)")
    plt.title(f"SLR (EUR): retta stimata (Test R²={r2_test:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_reg_dir / "scatter_fit_WorkExp_CompTotal.png", dpi=PLOT_DPI)
    plt.close()

    # Regressione per Paese: fit e plot per ciascun Country
    plots_reg_country_dir = Path("plots_regression/by_country")
    plots_reg_country_dir.mkdir(exist_ok=True)

    # Top 5 Paesi per numerosità (post-preprocessing globale)
    top_countries = data["Country"].value_counts().head(TOP_N_COUNTRIES).index.tolist()
    for country in top_countries:
        sub = data[data["Country"] == country]
        # IQR per Paese su CompTotal (rimuove outlier alti specifici del Paese)
        q1, q3 = sub["CompTotal"].quantile([0.25, 0.75])
        high = q3 + IQR_MULTIPLIER * (q3 - q1)
        sub = sub[sub["CompTotal"] <= high]

        Xc = sub[["WorkExp"]].values
        Yc = sub["CompTotal"].values

        # Fit SLR
        lr_c = LinearRegression()
        lr_c.fit(Xc, Yc)
        yhat_c = lr_c.predict(Xc)

        # Metriche di performance
        r2_c = r2_score(Yc, yhat_c)
        mse_c = mean_squared_error(Yc, yhat_c)
        rmse_c = float(np.sqrt(mse_c))

        print(
            f"[{country}] n={len(sub)} R^2={r2_c:.3f} RMSE={rmse_c:.1f} "
            f"Intercept={lr_c.intercept_:.2f} Slope={lr_c.coef_[0]:.2f}"
        )

        # Grafico scatter+retta
        xs_c = np.linspace(Xc.min(), Xc.max(), 200).reshape(-1, 1)
        ys_c = lr_c.predict(xs_c)
        fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)
        ax.scatter(Xc, Yc, alpha=SCATTER_ALPHA, s=SCATTER_SIZE)
        ax.plot(xs_c, ys_c, color="C1", linewidth=2)
        ax.set_xlabel("WorkExp (anni)")
        ax.set_ylabel("CompTotal (EUR)")
        ax.set_title(f"{country} — SLR (EUR)")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(plots_reg_country_dir / f"{country}_slr.png", dpi=PLOT_DPI)
        plt.close(fig)

    # Diagnostica residui (scala EUR) - on training set
    resid = y_train - yhat_train
    fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM)
    ax.scatter(yhat_train, resid, alpha=SCATTER_ALPHA, s=SCATTER_SIZE)
    ax.axhline(0, ls="--", color="black", linewidth=1)
    ax.set_xlabel("Fitted (EUR)")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted (EUR)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(plots_reg_dir / "residuals_vs_fitted_eur.png", dpi=PLOT_DPI)
    plt.close(fig)

    # Test statistico di normalità: Shapiro-Wilk
    # H0: i residui seguono una distribuzione normale
    # Se p-value > 0.05, non rifiutiamo H0 (residui probabilmente normali)
    stat_sw, p_value_sw = shapiro(resid)
    print(f"\n=== Shapiro-Wilk Test for Normality of Residuals ===")
    print(f"Statistic: {stat_sw:.4f}")
    print(f"P-value: {p_value_sw:.4f}")
    if p_value_sw > 0.05:
        print("Conclusion: Residuals appear normally distributed (p > 0.05, fail to reject H0)")
    else:
        print("Conclusion: Residuals may NOT be normally distributed (p ≤ 0.05, reject H0)")
    print("Note: Shapiro-Wilk test is sensitive to sample size; large samples may reject normality for minor deviations.")

    # QQ-plot residui
    fig = sm.ProbPlot(resid).qqplot(line="s")
    fig.set_size_inches(FIGSIZE_QQPLOT[0], FIGSIZE_QQPLOT[1])
    plt.title("QQ-plot residui (EUR)")
    plt.tight_layout()
    fig.savefig(plots_reg_dir / "qq_residuals_eur.png", dpi=PLOT_DPI)
    plt.close(fig)


if __name__ == "__main__":
    main()
