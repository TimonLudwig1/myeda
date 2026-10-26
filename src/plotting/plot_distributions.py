import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

def plot_normal_distribution(
        mu: float, 
        var: float, 
        test_type: str | None = None, 
        critical_value: float | None = None, 
        observed: float | None = None, 
        p_value: float | None = None, 
        show: bool = False
) -> tuple:
    
    fig, ax = plt.subplots(figsize=(7, 3.6))
    sigma = math.sqrt(var)
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    y = stats.norm.pdf(x, mu, sigma)
    ax.plot(x, y, color="#333333", lw=2)
    if test_type and critical_value and observed and p_value:
        ax.set_title(f"N({mu}, {sigma}²), {test_type}")
        ax.axvline(observed, color="#c0392b", lw=2, label=f"observed = {observed:.3g}")
        if test_type == "right":
            ax.axvline(critical_value, color="blue", linestyle="--", label=f"critical value = {critical_value:.3g}")
            ax.fill_between(x,y,where=(x >= observed),color="#c0392b",alpha=0.25,label=f"p-value = {p_value:.3g}")
        elif test_type == "left":
            ax.axvline(critical_value, color="blue", linestyle="--", label=f"critical value = {critical_value:.3g}")
            ax.fill_between(x,y,where=(x <= observed),color="#c0392b",alpha=0.25,label=f"p-value = {p_value:.3g}")
        elif test_type == "two":
            ax.axvline(critical_value, color="blue", linestyle="--", label=f"critical value = ±{critical_value:.3g}")
            ax.axvline(observed*(-1), color="#c0392b", lw=2)
            ax.axvline(critical_value*(-1), color="blue", linestyle="--")
            ax.fill_between(x,y,where=(x >= abs(observed)),color="#c0392b",alpha=0.25, label=f"p-value = {p_value:.3g}")
            ax.fill_between(x,y,where=(x <= abs(observed)*(-1)),color="#c0392b",alpha=0.25)
        ax.set_xlabel("z" if (mu, sigma) == (0, 1) else "x")
        ax.set_ylabel("density")
        ax.set_title(f"N({mu:g}, {sigma:g}²), {test_type} test")
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
    else:
        ax.set_title(f"N({mu:g}, {sigma:g}²)")
    if show:
        plt.show()

    return fig, ax

def plot_t_distribution(
        mu: float,
        var: float,
        dof: float,
        test_type: str | None = None,
        critical_value: float | None = None,
        observed: float | None = None,
        p_value: float | None = None,
        show: bool = False,
) -> tuple:

    fig, ax = plt.subplots(figsize=(7, 3.6))
    sigma = math.sqrt(var)
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    y = stats.t.pdf(x, dof, mu, sigma)
    ax.plot(x, y, color="#333333", lw=2)

    if test_type and critical_value and observed and p_value:
        ax.set_title(f"(t{mu}, {sigma}²), {test_type}")
        ax.axvline(observed, color="#c0392b", lw=2, label=f"observed = {observed:.3g}")
        if test_type == "right":
            ax.axvline(critical_value, color="blue", linestyle="--", label=f"critical value = {critical_value:.3g}")
            ax.fill_between(x,y,where=(x >= observed),color="#c0392b",alpha=0.25,label=f"p-value = {p_value:.3g}")
        elif test_type == "left":
            ax.axvline(critical_value, color="blue", linestyle="--", label=f"critical value = {critical_value:.3g}")
            ax.fill_between(x,y,where=(x <= observed),color="#c0392b",alpha=0.25,label=f"p-value = {p_value:.3g}")
        elif test_type == "two":
            ax.axvline(critical_value, color="blue", linestyle="--", label=f"critical value = ±{critical_value:.3g}")
            ax.axvline(observed*(-1), color="#c0392b", lw=2)
            ax.axvline(critical_value*(-1), color="blue", linestyle="--")
            ax.fill_between(x,y,where=(x >= abs(observed)),color="#c0392b",alpha=0.25, label=f"p-value = {p_value:.3g}")
            ax.fill_between(x,y,where=(x <= abs(observed)*(-1)),color="#c0392b",alpha=0.25)
        ax.set_xlabel("t")
        ax.set_ylabel("density")
        ax.set_title(f"t({mu:g}, {sigma:g}²), {test_type} test")
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
    else:
        ax.set_title(f"N({mu:g}, {sigma:g}²)")
    if show:
        plt.show()

    return fig, ax

def plot_chi2_distribution():
    pass
    
