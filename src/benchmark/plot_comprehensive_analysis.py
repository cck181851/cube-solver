import os
import math
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns

palette = sns.color_palette("colorblind", n_colors=8)

#  Helpers 
def _safe_scalar(x):
    if x is None:
        return None
    try:
        return x.item()
    except Exception:
        return x

def _get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _ensure_numeric(series, eps=1e-12):
    # Return a numeric pandas Series with coercion and eps for log transforms.
    s = pd.to_numeric(series, errors="coerce")
    return s


def _ci95(series):
    series = series.dropna()
    n = series.size
    if n <= 1:
        return 0.0
    se = stats.sem(series)
    # t critical for 95% two-sided
    ci = se * stats.t.ppf(0.975, df=n - 1)
    return ci


# Main class 
class PlotComprehensiveAnalysis:
    def __init__(self, results, output_folder="benchmark_data/plots", solvers=None, verbose=True):
        self.results = results or []
        self.output_folder = output_folder
        self.verbose = verbose
        self.solvers = solvers or ["thistlethwaite", "kociemba"]
        os.makedirs(self.output_folder, exist_ok=True)
        self.df = None
        self.summary = {}

    def _log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _normalize_one(self, row):
        out = {}
        cube_info = _get(row, "cube_info", None)
        # basic cube info fields
        for key in [
            "scramble",
            "cube",
            "manhattan_distance",
            "estimated_solution_length",
            "solution_variance",
            "overall_category",
            "manhattan_category",
            "hamming_category",
            "orientation_category",
            "solution_category",
        ]:
            val = _get(cube_info, key, None)
            out[f"cubeinfo_{key}"] = _safe_scalar(val)

        hamming = _get(cube_info, "hamming_distance", None)
        if hamming and isinstance(hamming, dict):
            out["cubeinfo_total_hamming"] = _safe_scalar(hamming.get("total_hamming"))
            out["cubeinfo_corner_hamming"] = _safe_scalar(hamming.get("corner_hamming"))
            out["cubeinfo_edge_hamming"] = _safe_scalar(hamming.get("edge_hamming"))
        else:
            out["cubeinfo_total_hamming"] = None
            out["cubeinfo_corner_hamming"] = None
            out["cubeinfo_edge_hamming"] = None

        orient = _get(cube_info, "orientation_distance", None)
        if orient and isinstance(orient, dict):
            out["cubeinfo_total_orientation"] = _safe_scalar(orient.get("total_orientation"))
        else:
            out["cubeinfo_total_orientation"] = None

        # solver-specific
        for solver_key in self.solvers:
            s = _get(row, solver_key, None)
            prefix = f"{solver_key}_"
            if s is None:
                out[prefix + "moves"] = None
                out[prefix + "time"] = None
                out[prefix + "time_cpu"] = None
                out[prefix + "memory"] = None
                out[prefix + "nodes_expanded"] = None
                out[prefix + "table_lookups"] = None
                out[prefix + "pruned_nodes"] = None
                out[prefix + "success"] = None
                out[prefix + "solution"] = None
            else:
                out[prefix + "moves"] = _safe_scalar(_get(s, "moves", None))
                out[prefix + "time"] = _safe_scalar(_get(s, "time", None))
                out[prefix + "time_cpu"] = _safe_scalar(_get(s, "time_cpu", None))
                out[prefix + "memory"] = _safe_scalar(_get(s, "memory", None))
                out[prefix + "nodes_expanded"] = _safe_scalar(_get(s, "nodes_expanded", None))
                out[prefix + "table_lookups"] = _safe_scalar(_get(s, "table_lookups", None))
                out[prefix + "pruned_nodes"] = _safe_scalar(_get(s, "pruned_nodes", None))
                out[prefix + "success"] = _safe_scalar(_get(s, "success", None))
                out[prefix + "solution"] = _safe_scalar(_get(s, "solution", None))

        return out

    def build_dataframe(self):
        rows = []
        for idx, r in enumerate(self.results):
            try:
                rows.append(self._normalize_one(r))
            except Exception as e:
                warnings.warn(f"Failed to normalize row {idx}: {e}")
        if not rows:
            raise ValueError("No data to plot (empty or failed normalization).")

        df = pd.DataFrame(rows)

        # convert numeric-like columns
        numeric_cols = []
        for col in df.columns:
            if any(col.endswith(s) for s in ["moves", "time", "time_cpu", "memory", "nodes_expanded", "table_lookups", "pruned_nodes", "total_hamming", "total_orientation"]) or col.startswith("cubeinfo_"):
                numeric_cols.append(col)
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # canonical difficulty column
        if "cubeinfo_overall_category" in df.columns:
            df["overall_category"] = df["cubeinfo_overall_category"].fillna("Unknown")
        else:
            df["overall_category"] = "Unknown"

        self.df = df
        return df

    # Plotting helpers
    def _savefig(self, fig, name, dpi=200):
        path = os.path.join(self.output_folder, name)
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def _paired_counts(self, col):
        # Count how many cubes where solver1 < solver2
        a = self.df[f"{self.solvers[0]}_{col}"]
        b = self.df[f"{self.solvers[1]}_{col}"]
        mask = a.notna() & b.notna()
        diffs = a[mask] - b[mask]
        return (diffs > 0).sum(), (diffs < 0).sum(), (diffs == 0).sum(), mask.sum()

    # Figures 
    def fig_moves(self):
        # Figure A: moves boxplots + paired slope plot
        df = self.df.copy()
        solvers = self.solvers
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1]})
        sns.set_style("whitegrid")

        # left: boxplots 
        moves_cols = [f"{s}_moves" for s in solvers]
        data_for_box = []
        for c in moves_cols:
            data_for_box.append(_ensure_numeric(df[c]).dropna().values)

        ax = axes[0]
        sns.boxplot(data=data_for_box, ax=ax, palette=[palette[0], palette[1]])
        # also overlay the raw points for transparency
        for i, arr in enumerate(data_for_box):
            if len(arr):
                x = np.full_like(arr, i, dtype=float) + (np.random.rand(len(arr)) - 0.5) * 0.12
                ax.scatter(x, arr, alpha=0.6, edgecolor="k", linewidth=0.3, s=30)
        ax.set_xticklabels(solvers)
        ax.set_ylabel("Moves (solution length) — face turns")
        ax.set_title("Distribution of solution lengths by solver")
        # annotate medians numerically above boxes
        for i, arr in enumerate(data_for_box):
            if len(arr):
                med = np.median(arr)
                ax.text(i, med + 0.7, f"{med:.1f}", ha="center", fontsize=10, weight="bold")

        # right: paired slope plot 
        ax2 = axes[1]
        a = _ensure_numeric(df[f"{solvers[0]}_moves"])
        b = _ensure_numeric(df[f"{solvers[1]}_moves"])
        mask = a.notna() & b.notna()
        a_m = a[mask].reset_index(drop=True)
        b_m = b[mask].reset_index(drop=True)
        n_pairs = len(a_m)

        # plot each pair, color by improvement/worse/no change
        for i in range(n_pairs):
            y0 = a_m.iloc[i]
            y1 = b_m.iloc[i]
            if np.isnan(y0) or np.isnan(y1):
                continue
            if y1 < y0:
                col = (0.0, 0.6, 0.0, 0.6)  # green for improvement (shorter moves)
            elif y1 > y0:
                col = (0.8, 0.1, 0.1, 0.6)  # red for worse
            else:
                col = (0.4, 0.4, 0.4, 0.6)  # neutral
            ax2.plot([0, 1], [y0, y1], marker='o', color=col, linewidth=0.9)

        # medians + annotation
        med1 = np.nanmedian(a)
        med2 = np.nanmedian(b)
        ax2.scatter([0, 1], [med1, med2], color='red', s=70, edgecolor='k', zorder=5,
                    label=f"median: {med1:.1f} vs {med2:.1f}")
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(solvers)
        ax2.set_title("Paired moves per cube (lines connect same cube)")
        ax2.set_xlim(-0.4, 1.4)

        # Wilcoxon paired test + paired Cohen's
        if n_pairs >= 2:
            try:
                # two-sided Wilcoxon for paired differences
                stat, pval = stats.wilcoxon(a_m, b_m, alternative='two-sided', zero_method='wilcox')
            except Exception:
                # fallback if all diffs are zero or not allowed
                stat, pval = np.nan, np.nan
            diffs = (b_m - a_m).dropna().values
            if len(diffs) >= 2:
                cohen_d = np.mean(diffs) / np.std(diffs, ddof=1) if np.std(diffs, ddof=1) != 0 else np.nan
            else:
                cohen_d = np.nan
            median_diff = np.median(diffs) if len(diffs) else np.nan
            stats_txt = f"n={n_pairs}, median Δ={median_diff:.1f}\nWilcoxon p={pval:.3g}\npaired d={cohen_d:.2f}"
        else:
            stats_txt = f"n={n_pairs} (not enough pairs for test)"

        ax2.text(0.02, 0.95, stats_txt, transform=ax2.transAxes, fontsize=10,
                va='top', bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.8"))

        ax2.legend(loc='upper right')

        fig.tight_layout()
        return self._savefig(fig, "01_moves_boxplot_and_paired.png")
    

    def fig_avg_metrics_by_difficulty(self):
        """
        Compute average metrics per difficulty group for each solver and plot grouped bars.
        Produces one subplot per metric: moves, time, nodes_expanded, memory.

        Output saved as: '09_avg_metrics_by_difficulty.png'
        """
        df = getattr(self, 'df', None)
        if df is None or df.empty:
            self._log("fig_avg_metrics_by_difficulty: no dataframe available")
            return None

        # canonical difficulty column 
        data = df.copy()
        data['overall_category'] = data.get('overall_category', pd.Series(['Unknown'] * len(data), index=data.index)).fillna('Unknown')

        metrics = ["moves", "time", "nodes_expanded", "memory"]
        groups = sorted(data['overall_category'].unique(), key=lambda x: (str(x).lower() if x is not None else ""))

        if not groups:
            self._log("fig_avg_metrics_by_difficulty: no difficulty groups found")
            return None

        n_groups = len(groups)
        x = np.arange(n_groups)
        width = 0.35

        fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
        if len(metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            ax = axes[i]
            vals0 = []
            vals1 = []
            counts = []
            for g in groups:
                mask = (data['overall_category'] == g)
                counts.append(int(mask.sum()))
                col0 = f"{self.solvers[0]}_{metric}"
                col1 = f"{self.solvers[1]}_{metric}"
                v0 = _ensure_numeric(data.loc[mask, col0]).dropna()
                v1 = _ensure_numeric(data.loc[mask, col1]).dropna()
                vals0.append(float(v0.mean()) if len(v0) else np.nan)
                vals1.append(float(v1.mean()) if len(v1) else np.nan)

            # draw grouped bars
            ax.bar(x - width/2, vals0, width, label=self.solvers[0], color=palette[0], alpha=0.9, edgecolor='k', linewidth=0.4)
            ax.bar(x + width/2, vals1, width, label=self.solvers[1], color=palette[1], alpha=0.9, edgecolor='k', linewidth=0.4)

            # ticks and labels
            ax.set_xticks(x)
            ax.set_xticklabels(groups, rotation=25, ha='right')
            ax.set_title(f"Average {metric} by difficulty")
            if metric in ("time", "nodes_expanded"):
                # these metrics are typically skewed, show log scale for readability
                ax.set_yscale('log')
                ax.set_ylabel(f"{metric} (log scale)")
            else:
                ax.set_ylabel(metric)

            # annotate counts above groups (place at 95% of top visible range)
            # compute y_max robustly ignoring NaNs
            yvals = np.array([v for v in (vals0 + vals1) if not (v is None or np.isnan(v))])
            if yvals.size:
                y_top = np.nanmax(yvals)
                # in log-scale case, place text at a multiplier above max; in linear case use additive offset
                for xi, cnt in enumerate(counts):
                    if metric in ("time", "nodes_expanded"):
                        ypos = y_top * 1.3
                    else:
                        ypos = y_top + (abs(y_top) * 0.08 if y_top != 0 else 1.0)
                    ax.text(xi, ypos, f"n={cnt}", ha='center', va='bottom', fontsize=9, color='black', alpha=0.9)
            else:
                # if all NaN (no data), annotate n=0
                for xi, cnt in enumerate(counts):
                    ax.text(xi, 0.1, f"n={cnt}", ha='center', va='bottom', fontsize=9, color='black', alpha=0.9)

            if i == 0:
                ax.legend()

        fig.suptitle("Average metrics by difficulty (per solver)", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return self._savefig(fig, "09_avg_metrics_by_difficulty.png")

    def fig_time(self):
        # Figure B: time histogram (log) + time vs moves scatter with robust handling of numeric types
        df = self.df.copy()
        solvers = self.solvers
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.set_style("whitegrid")
        eps = 1e-9

        # left: histogram of log10(time) with readable tick labels 
        ax = axes[0]
        min_log = +np.inf
        max_log = -np.inf
        for i, s in enumerate(solvers):
            col = f"{s}_time"
            series = _ensure_numeric(df[col]).dropna()
            if series.empty:
                continue
            # ensure float
            series = series.astype(float)
            logt = np.log10(series + eps)
            min_log = min(min_log, np.nanmin(logt))
            max_log = max(max_log, np.nanmax(logt))
            ax.hist(logt, bins=25, alpha=0.45, label=f"{s} (n={len(series)})",
                    histtype='stepfilled', color=palette[i])

        if min_log == np.inf:
            min_log, max_log = -3, 0
        tick_lo = int(np.floor(min_log))
        tick_hi = int(np.ceil(max_log))
        ticks = np.arange(tick_lo, tick_hi + 1)
        ax.set_xticks(ticks)
        # labels computed with float base to avoid NumPy integer-power issues
        labels = [f"{(10.0 ** float(t)):.3g}s" for t in ticks]
        ax.set_xticklabels(labels)
        ax.set_xlabel("Time (seconds) — log scale ticks")
        ax.set_title("Solve time distributions")
        ax.legend()

        # right: scatter (moves vs time) with regression fitted on log10(time) 
        ax2 = axes[1]
        for i, s in enumerate(solvers):
            common = df[[f"{s}_moves", f"{s}_time"]].dropna()
            if common.empty:
                continue
            x = _ensure_numeric(common[f"{s}_moves"]).astype(float).values
            y = _ensure_numeric(common[f"{s}_time"]).astype(float).values
            ax2.scatter(x, y, alpha=0.75, label=f"{s} (n={len(x)})", s=40,
                        edgecolor='k', linewidth=0.3, marker='o', color=palette[i])

            # regression on log10(y) -> plot back on linear time axis using np.power(10.0, ...)
            try:
                logy = np.log10(y + eps)
                lr = stats.linregress(x, logy)
                xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
                log_ys = (lr.intercept + lr.slope * xs).astype(float)
                # robust inverse-log: use np.power with float base & float exponent
                ys = np.power(10.0, log_ys)
                ax2.plot(xs, ys, linestyle='--', linewidth=1.5, color=palette[i],
                        label=f"{s} trend (R²={lr.rvalue**2:.2f}, p={lr.pvalue:.2g})")
            except Exception as e:
                # print an informative message
                print(f"fig_time: regression plotting failed for solver '{s}': {e}")

        ax2.set_yscale('log')
        ax2.set_xlabel("Moves")
        ax2.set_ylabel("Time (s) — log scale")
        ax2.set_title("Time vs Moves (per solver)")
        ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
        fig.tight_layout()
        return self._savefig(fig, "02_time_hist_and_time_vs_moves.png")

    # Figure C: resources (nodes_expanded + memory)
    def fig_resources(self):
        """
        Figure C: nodes_expanded and memory grouped plots.
        nodes_expanded: show median ± IQR with jittered points; switch to log-scale if strongly skewed.
        memory: mean +- 95% CI with raw points.
        """
        df = self.df.copy()
        solvers = self.solvers
        metrics = ["nodes_expanded", "memory"]
        fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 7))
        sns.set_style("whitegrid")

        for i, m in enumerate(metrics):
            ax = axes[i]
            medians = []
            iqr_half = []
            raw_vals = []
            for s in solvers:
                col = f"{s}_{m}"
                series = _ensure_numeric(df[col]).dropna()
                raw_vals.append(series)
                if series.empty:
                    medians.append(np.nan)
                    iqr_half.append(0.0)
                else:
                    q1 = np.nanpercentile(series, 25)
                    q3 = np.nanpercentile(series, 75)
                    med = np.nanmedian(series)
                    medians.append(med)
                    iqr_half.append((q3 - q1) / 2.0)

            x = np.arange(len(solvers))
            # nodes_expanded: use median+IQR; if skewed, use log scale
            if m == "nodes_expanded":
                # check skewness across pooled data to decide
                pooled = pd.concat([rv for rv in raw_vals if rv is not None and len(rv) > 0], ignore_index=True)
                skewness = pooled.skew() if len(pooled) > 0 else 0.0
                # plot median bars with IQR errorbars
                ax.bar(x, medians, yerr=iqr_half, alpha=0.9, color=[palette[2], palette[3]], capsize=6)
                # overlay jittered raw points
                for xi, rv in enumerate(raw_vals):
                    if rv is None or len(rv) == 0:
                        continue
                    jitter_x = np.random.normal(loc=xi, scale=0.08, size=len(rv))
                    ax.scatter(jitter_x, rv, color='k', alpha=0.6, s=25, edgecolor='white', linewidth=0.3)
                ax.set_ylabel("nodes_expanded (count)")
                ax.set_xticks(x)
                ax.set_xticklabels(solvers)
                ax.set_title("nodes_expanded — median ± IQR (individual points shown)")
                if skewness > 1.0:
                    ax.set_yscale('log')
                    ax.text(0.98, 0.95, f"skew={skewness:.2f} — log scale shown",
                            transform=ax.transAxes, ha="right", va="top",
                            bbox=dict(fc='white', alpha=0.8))
            else:
                # memory: show mean +- 95% CI + jittered points
                means = []
                cis = []
                for rv in raw_vals:
                    if rv is None or len(rv) == 0:
                        means.append(np.nan)
                        cis.append(0.0)
                    else:
                        means.append(np.mean(rv))
                        cis.append(_ci95(rv))
                ax.bar(x, means, yerr=cis, alpha=0.9, color=[palette[4], palette[5]], capsize=6)
                for xi, rv in enumerate(raw_vals):
                    if rv is None or len(rv) == 0:
                        continue
                    jitter_x = np.random.normal(loc=xi, scale=0.08, size=len(rv))
                    ax.scatter(jitter_x, rv, color='k', alpha=0.6, s=25, edgecolor='white', linewidth=0.3)
                ax.set_ylabel("memory (MB)")
                ax.set_xticks(x)
                ax.set_xticklabels(solvers)
                ax.set_title("memory — mean +- 95% CI (individual points shown)")

        fig.tight_layout()
        return self._savefig(fig, "03_resources_nodes_memory.png")

    def fig_difficulty(self):
        df = getattr(self, 'df', None)
        if df is None or df.empty:
            self._log("fig_difficulty: no dataframe available")
            return None

        # make a local copy so we don't mutate original
        df = df.copy()

        # Preferred categorical columns in order of preference
        preferred = ['overall_category', 'solution_category', 'manhattan_category', 'hamming_category']

        # Treat literal 'Unknown' strings as missing 
        for c in preferred:
            if c in df.columns:
                df[c] = df[c].replace('Unknown', np.nan)

        # pick a category column that has >1 non-na values
        cat_col = None
        for c in preferred:
            if c in df.columns and df[c].notna().sum() > 1:
                cat_col = c
                break

        # If there's no meaningful category, create a single 'All' group
        single_group_mode = False
        if cat_col is None:
            df['__all__'] = 'All'
            cat_col = '__all__'
            single_group_mode = True
        else:
            # If the chosen column has <=1 unique non-null value after dropping 'Unknown',
            # collapse to single group to avoid an ugly "Unknown" box
            unique_vals = pd.Series(df[cat_col].dropna().unique())
            if unique_vals.size <= 1:
                df['__all__'] = 'All'
                cat_col = '__all__'
                single_group_mode = True

        # Build long-format DataFrame containing solver, category, moves, time
        rows = []
        for s in self.solvers:
            moves_col = f"{s}_moves"
            time_col = f"{s}_time"
            # include only columns that exist
            cols = [cat_col]
            if moves_col in df.columns:
                cols.append(moves_col)
            if time_col in df.columns:
                cols.append(time_col)
            if len(cols) <= 1:
                continue
            tmp = df[cols].copy()
            tmp = tmp.rename(columns={moves_col: 'moves', time_col: 'time'})
            tmp['solver'] = s
            rows.append(tmp)
        if not rows:
            self._log("fig_difficulty: no solver columns found")
            return None
        long = pd.concat(rows, ignore_index=True)

        # Ensure numeric and drop rows with neither metric
        long['moves'] = pd.to_numeric(long.get('moves'), errors='coerce')
        long['time'] = pd.to_numeric(long.get('time'), errors='coerce')
        long = long.dropna(subset=['moves', 'time'], how='all')

        # compute log-time (safe)
        eps = 1e-9
        long['log_time'] = np.log10(long['time'].fillna(eps) + eps)

        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Decide xlabel: if collapsed to single group, don't show the column name as xlabel
        xlabel = '' if single_group_mode else cat_col

        # Moves boxplot with strip overlay (strip plot will not create a legend to avoid duplicates)
        try:
            sns.boxplot(data=long, x=cat_col, y='moves', hue='solver', ax=axes[0], showfliers=True, palette='muted')
            sns.stripplot(data=long, x=cat_col, y='moves', hue='solver', dodge=True,
                        alpha=0.45, ax=axes[0], jitter=0.12, linewidth=0.35, legend=False)
            axes[0].set_title("Moves by difficulty category (per solver)")
            axes[0].set_xlabel(xlabel)
            axes[0].set_ylabel("Moves (solution length)")
            # remove any duplicate legend at the axes level (we will create one shared legend)
            if axes[0].get_legend() is not None:
                try:
                    axes[0].legend_.remove()
                except Exception:
                    pass
        except Exception as e:
            self._log("fig_difficulty: error plotting moves:", e)

        # log10(time) boxplot with strip overlay
        try:
            sns.boxplot(data=long, x=cat_col, y='log_time', hue='solver', ax=axes[1], showfliers=True, palette='muted')
            sns.stripplot(data=long, x=cat_col, y='log_time', hue='solver', dodge=True,
                        alpha=0.45, ax=axes[1], jitter=0.12, linewidth=0.35, legend=False)
            axes[1].set_title("log10(Time) by difficulty category (per solver)")
            axes[1].set_xlabel(xlabel)
            axes[1].set_ylabel("log10(Time (s))")
            if axes[1].get_legend() is not None:
                try:
                    axes[1].legend_.remove()
                except Exception:
                    pass
        except Exception as e:
            self._log("fig_difficulty: error plotting log_time:", e)

        # remove tick labels when single_group_mode 
        if single_group_mode:
            for ax in axes:
                ax.set_xlabel('')       # ensure axis label is empty
                ax.set_xticks([])       # remove tick positions
                ax.set_xticklabels([])  # and their text (this removes the visible "All")

        # Build a single shared legend (from axes[1] or axes[0] whichever has handles)
        try:
            # get handles/labels from axes[1] first, otherwise axes[0]
            handles, labels = axes[1].get_legend_handles_labels()
            if not handles:
                handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles[:len(self.solvers)], labels[:len(self.solvers)],
                        loc='upper center', ncol=len(self.solvers), frameon=False)
        except Exception:
            pass

        for ax in axes:
            ax.tick_params(axis='x', rotation=25)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        return self._savefig(fig, "04_difficulty_vs_performance_fixed.png")


    def fig_differences_and_success(self):
        """
        Paired differences histograms and success outcomes (robust).
        Computes differences only on paired rows (both solvers present).
        Saves as '05_paired_differences_and_success.png'
        """

        df = getattr(self, 'df', None)
        if df is None or df.empty:
            self._log("fig_differences_and_success: no dataframe available")
            return None

        s0, s1 = (self.solvers[0], self.solvers[1]) if len(self.solvers) >= 2 else (None, None)
        if s0 is None or s1 is None:
            self._log("fig_differences_and_success: need two solvers in self.solvers")
            return None

        # Paired masks (ensure alignment)
        moves_col0, moves_col1 = f"{s0}_moves", f"{s1}_moves"
        time_col0, time_col1 = f"{s0}_time", f"{s1}_time"

        mask_moves = df.get(moves_col0).notna() & df.get(moves_col1).notna() if (moves_col0 in df.columns and moves_col1 in df.columns) else pd.Series([False]*len(df), index=df.index)
        mask_time  = df.get(time_col0).notna()  & df.get(time_col1).notna()  if (time_col0 in df.columns and time_col1 in df.columns) else pd.Series([False]*len(df), index=df.index)

        moves_a = pd.to_numeric(df.loc[mask_moves, moves_col0], errors='coerce')
        moves_b = pd.to_numeric(df.loc[mask_moves, moves_col1], errors='coerce')
        time_a = pd.to_numeric(df.loc[mask_time, time_col0], errors='coerce')
        time_b = pd.to_numeric(df.loc[mask_time, time_col1], errors='coerce')

        # Differences (paired)
        diff_moves = (moves_a - moves_b).dropna()
        eps = 1e-9
        # Use log10 for time difference to reduce skew
        log_time_a = np.log10(time_a + eps)
        log_time_b = np.log10(time_b + eps)
        diff_logtime = (log_time_a - log_time_b).dropna()

        # Prepare figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Histogram: moves differences
        try:
            if not diff_moves.empty:
                axes[0].hist(diff_moves, bins='auto', alpha=0.75)
                mv_mean = diff_moves.mean()
                axes[0].axvline(mv_mean, color='red', linestyle='--', linewidth=1.8, label=f"mean={mv_mean:.2f}")
                axes[0].legend()
            axes[0].set_title(f"Paired difference in moves ({s0} - {s1})")
            axes[0].set_xlabel("Moves difference")
        except Exception as e:
            self._log("fig_differences_and_success: error plotting moves histogram:", e)

        # Histogram: log(time) differences
        try:
            if not diff_logtime.empty:
                axes[1].hist(diff_logtime, bins='auto', alpha=0.75)
                lt_mean = diff_logtime.mean()
                axes[1].axvline(lt_mean, color='red', linestyle='--', linewidth=1.8, label=f"mean={lt_mean:.3f}")
                axes[1].legend()
            axes[1].set_title("Paired difference in log(time)")
            axes[1].set_xlabel("log10(time) difference")
        except Exception as e:
            self._log("fig_differences_and_success: error plotting logtime histogram:", e)

        # Success outcomes: try to detect success columns robustly
        try:
            succ_cols = [c for c in df.columns if 'success' in c.lower()]
            if len(succ_cols) >= 2:
                sc0, sc1 = succ_cols[0], succ_cols[1]
                s0_bool = df[sc0].fillna(False).astype(bool)
                s1_bool = df[sc1].fillna(False).astype(bool)
                both = (s0_bool & s1_bool).sum()
                only0 = (s0_bool & ~s1_bool).sum()
                only1 = (~s0_bool & s1_bool).sum()
                neither = (~s0_bool & ~s1_bool).sum()
                counts = pd.Series({'both': int(both), f'only_{s0}': int(only0), f'only_{s1}': int(only1), 'neither': int(neither)})
            else:
                # fallback: check nested-like columns 'thistlethwaite_success', 'kociemba_success'
                sc0, sc1 = f"{s0}_success", f"{s1}_success"
                if sc0 in df.columns or sc1 in df.columns:
                    s0_bool = df.get(sc0, pd.Series(False, index=df.index)).fillna(False).astype(bool)
                    s1_bool = df.get(sc1, pd.Series(False, index=df.index)).fillna(False).astype(bool)
                    both = (s0_bool & s1_bool).sum()
                    only0 = (s0_bool & ~s1_bool).sum()
                    only1 = (~s0_bool & s1_bool).sum()
                    neither = (~s0_bool & ~s1_bool).sum()
                    counts = pd.Series({'both': int(both), f'only_{s0}': int(only0), f'only_{s1}': int(only1), 'neither': int(neither)})
                else:
                    # no success info available: create zeros
                    counts = pd.Series({'both': 0, f'only_{s0}': 0, f'only_{s1}': 0, 'neither': len(df)})

            order = ['both', f'only_{s0}', f'only_{s1}', 'neither']
            counts = counts.reindex(order).fillna(0).astype(int)
            axes[2].bar(counts.index, counts.values, color='C2')
            axes[2].set_title("Success outcomes")
            axes[2].set_ylabel("Count")
            axes[2].tick_params(axis='x', rotation=25)
        except Exception as e:
            self._log("fig_differences_and_success: error computing success counts:", e)
            axes[2].text(0.5, 0.5, "No success data", ha='center', va='center')

        fig.tight_layout()
        return self._savefig(fig, "05_paired_differences_and_success.png")

    def fig_correlation_heatmap(self):
        # correlation heatmap of numeric metrics
        df = self.df
        # pick relevant numeric columns
        numeric_cols = []
        for col in df.columns:
            if df[col].dtype.kind in 'biufc' and col.count('_')>0:
                numeric_cols.append(col)
        if not numeric_cols:
            return None
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(max(6, len(numeric_cols)*0.5), max(4, len(numeric_cols)*0.5)))
        cax = ax.matshow(corr, vmin=-1, vmax=1)
        fig.colorbar(cax)
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=90)
        ax.set_yticklabels(numeric_cols)
        ax.set_title('Correlation matrix (numeric metrics)')
        return self._savefig(fig, "appendix_correlation_heatmap.png")

    def fig_zscore_heatmap(self):
        df = self.df
        metrics = ['moves', 'time', 'nodes_expanded', 'memory']
        rows = []
        labels = []
        for s in self.solvers:
            row_vals = []
            for m in metrics:
                col = f"{s}_{m}"
                series = _ensure_numeric(df[col]).dropna()
                row_vals.append(series.mean() if not series.empty else np.nan)
            rows.append(row_vals)
            labels.append(s)
        mat = np.array(rows, dtype=float)  # shape: (n_solvers, n_metrics)

        # z-score per metric across solvers
        z = stats.zscore(mat, axis=0, nan_policy='omit')

        # handle case where std==0 producing NaNs
        z = np.where(np.isfinite(z), z, 0.0)

        # center colorbar around zero
        vmax = np.nanmax(np.abs(z)) if z.size else 1.0
        vmin = -vmax

        fig, ax = plt.subplots(figsize=(6, 3))
        cax = ax.matshow(z, aspect='auto', vmin=vmin, vmax=vmax, cmap='RdBu_r')
        fig.colorbar(cax, ax=ax)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_title('Z-scored average metrics (solver x metric)')
        plt.tight_layout()
        return self._savefig(fig, "appendix_zscore_heatmap.png")
    
    #  Summaries & averages
    def save_summary_table(self):
        df = self.df
        rows = []
        for s in self.solvers:
            prefix = f"{s}_"
            row = {"solver": s}
            for metric in ["moves", "time", "nodes_expanded", "memory", "table_lookups", "pruned_nodes"]:
                col = prefix + metric
                ser = _ensure_numeric(df[col])
                row[metric + "__mean"] = float(ser.mean()) if ser.dropna().size else None
                row[metric + "__median"] = float(ser.median()) if ser.dropna().size else None
                row[metric + "__std"] = float(ser.std()) if ser.dropna().size else None
            row['success_rate'] = float(df[prefix + 'success'].fillna(False).astype(bool).mean())
            rows.append(row)
        summary_df = pd.DataFrame(rows)
        csvp = os.path.join(self.output_folder, 'summary_table_by_solver.csv')
        summary_df.to_csv(csvp, index=False)
        jsonp = os.path.join(self.output_folder, 'summary_table_by_solver.json')
        summary_df.to_json(jsonp, orient='records', indent=2)
        return csvp, jsonp

    def compute_group_averages(self):
        """
        Calculate averaged data across different metrics grouped by difficulty level.
        Returns CSV file path and the grouped DataFrame.
        """
        
        df = self.df.copy()
        
        # Ensure having a difficulty column
        if 'overall_category' not in df.columns:
            df['overall_category'] = 'Unknown'
        
        # Handle NaN values in category
        df['overall_category'] = df['overall_category'].fillna('Unknown')
        
        # Get unique categories
        unique_categories = df['overall_category'].unique()
        
        # Define metrics to average
        metrics = ["moves", "time", "nodes_expanded", "memory", "table_lookups", "pruned_nodes"]
        
        # Initialize results storage
        group_results = []
        
        # Group by difficulty category
        for category in unique_categories:
            if pd.isna(category):
                continue
                
            mask = df['overall_category'] == category
            category_data = df[mask]
            n_cubes = len(category_data)
            
            if n_cubes == 0:
                continue
                
            category_result = {
                'difficulty_category': category,
                'n_cubes': n_cubes
            }
            
            # Calculate averages for each solver and metric
            for solver in self.solvers:
                solver_prefix = f"{solver}_"
                
                for metric in metrics:
                    col_name = solver_prefix + metric
                    if col_name not in category_data.columns:
                        continue
                        
                    series = _ensure_numeric(category_data[col_name]).dropna()
                    valid_count = len(series)
                    
                    if valid_count > 0:
                        # Basic statistics
                        category_result[f'{solver}_{metric}_mean'] = float(series.mean())
                        category_result[f'{solver}_{metric}_median'] = float(series.median())
                        category_result[f'{solver}_{metric}_std'] = float(series.std())
                        category_result[f'{solver}_{metric}_min'] = float(series.min())
                        category_result[f'{solver}_{metric}_max'] = float(series.max())
                        category_result[f'{solver}_{metric}_count'] = int(valid_count)
                        
                        # 95% confidence interval
                        if valid_count >= 2:
                            ci_value = _ci95(series)
                            category_result[f'{solver}_{metric}_ci95'] = float(ci_value)
                        else:
                            category_result[f'{solver}_{metric}_ci95'] = 0.0
                    else:
                        # No data available
                        category_result[f'{solver}_{metric}_mean'] = None
                        category_result[f'{solver}_{metric}_median'] = None
                        category_result[f'{solver}_{metric}_std'] = None
                        category_result[f'{solver}_{metric}_min'] = None
                        category_result[f'{solver}_{metric}_max'] = None
                        category_result[f'{solver}_{metric}_count'] = 0
                        category_result[f'{solver}_{metric}_ci95'] = None
                
                # Success rate
                success_col = solver_prefix + 'success'
                if success_col in category_data.columns:
                    success_series = category_data[success_col].fillna(False)
                    success_count = success_series.astype(bool).sum()
                    total_count = len(success_series)
                    if total_count > 0:
                        category_result[f'{solver}_success_rate'] = float(success_count / total_count)
                    else:
                        category_result[f'{solver}_success_rate'] = 0.0
                    category_result[f'{solver}_success_count'] = int(success_count)
                else:
                    category_result[f'{solver}_success_rate'] = None
                    category_result[f'{solver}_success_count'] = None
            
            group_results.append(category_result)
        
        # Create DataFrame from results
        if not group_results:
            # Create empty result with default structure
            group_df = pd.DataFrame(columns=['difficulty_category', 'n_cubes'])
        else:
            group_df = pd.DataFrame(group_results)
        
        # Sort by difficulty category for consistent ordering
        if 'difficulty_category' in group_df.columns and not group_df.empty:
            try:
                # Define logical difficulty order
                difficulty_order = ['Easy', 'Medium', 'Hard', 'Very Hard', 'Extreme', 'All', 'Unknown']
                
                # Get categories present in our data
                present_categories = [cat for cat in difficulty_order if cat in group_df['difficulty_category'].values]
                other_categories = [cat for cat in group_df['difficulty_category'].unique() if cat not in present_categories]
                
                # Combine predefined order with remaining categories
                custom_order = present_categories + sorted(other_categories)
                
                # Convert to categorical for proper sorting
                group_df['difficulty_category'] = pd.Categorical(
                    group_df['difficulty_category'], 
                    categories=custom_order, 
                    ordered=True
                )
                group_df = group_df.sort_values('difficulty_category')
            except Exception as e:
                # Fallback: alphabetical sort
                self._log(f"Custom sorting failed, using alphabetical: {e}")
                group_df = group_df.sort_values('difficulty_category')
        
        # Save to CSV
        csv_path = os.path.join(self.output_folder, 'benchmark_data/averaged_metrics_by_difficulty.csv')
        group_df.to_csv(csv_path, index=False)
        
        # Also save a simplified version for quick viewing
        if not group_df.empty:
            # Create a simplified version with key metrics only
            simple_columns = ['difficulty_category', 'n_cubes']
            for solver in self.solvers:
                simple_columns.extend([
                    f'{solver}_moves_mean', 
                    f'{solver}_time_mean',
                    f'{solver}_success_rate'
                ])
            
            # Only include columns that actually exist
            available_columns = [col for col in simple_columns if col in group_df.columns]
            simple_df = group_df[available_columns]
            simple_csv_path = os.path.join(self.output_folder, 'benchmark_data/simplified_metrics_by_difficulty.csv')
            simple_df.to_csv(simple_csv_path, index=False)
        
        self._log(f"Computed group averages for {len(group_df)} difficulty categories")
        return csv_path, group_df

    def generate_all_plots(self):
        """
        Main entrypoint: build df, create plots, summaries, and return a dict of outputs.
        This robust version wraps each plotting call in try/except so one failure doesn't stop the pipeline.
        """
        outputs = {}
        self._log("Building dataframe...")
        self.build_dataframe()

        # Core plots
        try:
            self._log("Generating Figure: moves")
            outputs['fig_moves'] = self.fig_moves()
        except Exception as e:
            self._log("fig_moves failed:", e)

        try:
            self._log("Generating Figure: time")
            outputs['fig_time'] = self.fig_time()
        except Exception as e:
            self._log("fig_time failed:", e)

        try:
            self._log("Generating Figure: resources")
            outputs['fig_resources'] = self.fig_resources()
        except Exception as e:
            self._log("fig_resources failed:", e)

        # Difficulty plot 
        try:
            self._log("Generating Figure: difficulty")
            outputs['fig_difficulty'] = self.fig_difficulty()
        except Exception as e:
            self._log("fig_difficulty failed:", e)

        try:
            self._log("Generating Figure: differences & success")
            outputs['fig_differences_success'] = self.fig_differences_and_success()
        except Exception as e:
            self._log("fig_differences_and_success failed:", e)

        # Appendices
        try:
            self._log("Generating Appendix: correlation heatmap")
            outputs['fig_corr_heatmap'] = self.fig_correlation_heatmap()
        except Exception as e:
            self._log("fig_correlation_heatmap failed:", e)

        # Additional plots (ECDF, nodes vs time, Bland–Altman)
        for fn_name in ('fig_ecdf', 'fig_nodes_time_scatter', 'fig_bland_altman'):
            fn = getattr(self, fn_name, None)
            if callable(fn):
                try:
                    self._log(f"Generating Figure: {fn_name}")
                    out = fn() if fn_name != 'fig_bland_altman' else fn(metric='time')
                    if out is not None:
                        outputs[fn_name] = out
                except Exception as e:
                    self._log(f"{fn_name} failed:", e)

        # Save summary table
        try:
            self._log("Saving summary table")
            csvp, jsonp = self.save_summary_table()
            outputs['summary_table_csv'] = csvp
            outputs['summary_table_json'] = jsonp
        except Exception as e:
            self._log("save_summary_table failed:", e)

        # averaged metrics by difficulty (grouped bar chart)
        try:
            self._log("Generating Figure: avg_metrics_by_difficulty")
            outputs['fig_avg_metrics_by_difficulty'] = self.fig_avg_metrics_by_difficulty()
        except Exception as e:
            self._log("fig_avg_metrics_by_difficulty failed:", e)            

        # Save brief summary
        try:
            summary_json = os.path.join(self.output_folder, 'brief_summary.json')
            brief = {
                'n_cubes': int(len(self.df)) if hasattr(self, 'df') else None,
                'solvers': self.solvers if hasattr(self, 'solvers') else None,
                'outputs': outputs,
            }
            with open(summary_json, 'w') as f:
                json.dump(brief, f, indent=2)
            outputs['brief_summary_json'] = summary_json
            self.summary.update(brief)
        except Exception as e:
            self._log("Saving brief_summary failed:", e)

        self._log("Done. Files written to:", getattr(self, 'output_folder', 'unknown'))
        return outputs

    def fig_ecdf(self):
        # ECDFs for time and moves per solver (two-panel).
        df = self.df
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        eps = 1e-12

        def _ecdf_plot(ax, data, label):
            # data is 1D numpy/pd series
            x = np.sort(data)
            y = np.arange(1, len(x)+1) / len(x)
            ax.step(x, y, where='post', label=label)
            return

        for s in self.solvers:
            # time ECDF (left)
            colt = _ensure_numeric(df[f"{s}_time"]).dropna()
            if len(colt):
                _ecdf_plot(axes[0], colt.values, s + f" (n={len(colt)})")
            # moves ECDF (right)
            colm = _ensure_numeric(df[f"{s}_moves"]).dropna()
            if len(colm):
                _ecdf_plot(axes[1], colm.values, s + f" (n={len(colm)})")

        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Proportion ≤ x")
        axes[0].set_title("ECDF: Solve Time")
        axes[0].legend()
        axes[1].set_xlabel("Moves (solution length)")
        axes[1].set_ylabel("Proportion ≤ x")
        axes[1].set_title("ECDF: Moves (solution length)")
        axes[1].legend()

        return self._savefig(fig, "06_ecdf_time_and_moves.png")

    def fig_nodes_time_scatter(self, color_by='solver'):
        # Scatter nodes_expanded vs time, colored by solver or difficulty.
        df = self.df
        fig, ax = plt.subplots(figsize=(8, 6))
        eps = 1e-9
        if color_by == 'solver':
            for s in self.solvers:
                common = df[[f"{s}_nodes_expanded", f"{s}_time", f"{s}_moves"]].dropna()
                if common.empty:
                    continue
                x = common[f"{s}_nodes_expanded"]
                y = common[f"{s}_time"]
                sizes = (common[f"{s}_moves"].fillna(common[f"{s}_moves"].median()) + 1) * 6
                ax.scatter(x, y, alpha=0.6, s=sizes, label=s)
                try:
                    rho, p = stats.spearmanr(x, y, nan_policy='omit')
                    ax.annotate(f"{s} ρ={rho:.2f}", xy=(0.02, 0.95 - 0.05*self.solvers.index(s)), 
                                xycoords='axes fraction')
                except Exception:
                    pass
        else:
            # color by difficulty category (overall_category)
            cats = df['overall_category'].fillna('Unknown').unique()
            for cat in cats:
                mask = (df['overall_category'] == cat)
                # combine both solvers' nodes/time for visibility (flatten)
                for s in self.solvers:
                    common = df.loc[mask, [f"{s}_nodes_expanded", f"{s}_time"]].dropna()
                    if common.empty:
                        continue
                    ax.scatter(common[f"{s}_nodes_expanded"], common[f"{s}_time"], alpha=0.6, label=f"{cat}:{s}")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Nodes expanded (log scale)")
        ax.set_ylabel("Time (s) (log scale)")
        ax.set_title("Nodes expanded vs Solve Time (marker size ∝ moves)")
        ax.legend(loc='best', fontsize='small')
        return self._savefig(fig, "07_nodes_vs_time_scatter.png")


    def fig_bland_altman(self, metric='time'):
        """
        Bland-Altman for paired metrics (metric is 'time' or 'moves').
        For time we use log10(time) before difference to reduce skew.
        """
        df = self.df
        s0 = self.solvers[0]
        s1 = self.solvers[1]
        col0 = _ensure_numeric(df[f"{s0}_{metric}"])
        col1 = _ensure_numeric(df[f"{s1}_{metric}"])
        mask = col0.notna() & col1.notna()
        a = col0[mask]
        b = col1[mask]
        if a.empty:
            return None
        if metric == 'time':
            # use log scale for stability
            a = np.log10(a + 1e-9)
            b = np.log10(b + 1e-9)
            ylabel = "log10(time) (s)"
        else:
            ylabel = metric

        mean_ab = (a + b) / 2.0
        diff = a - b
        md = np.nanmean(diff)
        sd = np.nanstd(diff, ddof=1)
        loa_upper = md + 1.96 * sd
        loa_lower = md - 1.96 * sd

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(mean_ab, diff, alpha=0.6)
        ax.axhline(md, color='red', linestyle='--', label=f"mean diff={md:.3f}")
        ax.axhline(loa_upper, color='gray', linestyle='--', label=f"+1.96·SD={loa_upper:.3f}")
        ax.axhline(loa_lower, color='gray', linestyle='--', label=f"-1.96·SD={loa_lower:.3f}")
        ax.set_xlabel(f"Mean of {s0} & {s1} ({ylabel})")
        ax.set_ylabel(f"Difference ({s0} - {s1}) ({ylabel})")
        ax.set_title(f"Bland–Altman: {metric} (paired, n={len(diff)})")
        ax.legend()
        return self._savefig(fig, f"08_bland_altman_{metric}.png")


    def run_stat_tests(self):
        """
        Run paired statistical tests for moves and time. Return a small dict of results.
        - Paired t-test + Wilcoxon (non-parametric) for paired differences.
        - Cohen's for paired samples (mean(diff)/sd(diff)).
        """
        from scipy import stats
        df = self.df
        results = {}
        for metric in ['moves', 'time']:
            s0 = self.solvers[0]
            s1 = self.solvers[1]
            a = _ensure_numeric(df[f"{s0}_{metric}"])
            b = _ensure_numeric(df[f"{s1}_{metric}"])
            mask = a.notna() & b.notna()
            a = a[mask]
            b = b[mask]
            res = {'n': int(mask.sum())}
            if len(a) < 2:
                res.update({'t_test_p': None, 'wilcoxon_p': None, 'cohens_d': None})
                results[metric] = res
                continue
            diff = a - b
            # paired t-test 
            try:
                t_stat, p_t = stats.ttest_rel(a, b, nan_policy='omit')
            except Exception:
                p_t = None
                t_stat = None
            # Wilcoxon signed-rank 
            try:
                w_stat, p_w = stats.wilcoxon(a, b)
            except Exception:
                p_w = None
                w_stat = None
            # Cohen's for paired samples
            md = float(np.nanmean(diff))
            sd = float(np.nanstd(diff, ddof=1))
            cohens_d = md / sd if sd and not math.isnan(sd) else None

            res.update({
                't_stat': float(t_stat) if t_stat is not None else None,
                't_test_p': float(p_t) if p_t is not None else None,
                'wilcoxon_stat': float(w_stat) if w_stat is not None else None,
                'wilcoxon_p': float(p_w) if p_w is not None else None,
                'cohens_d_paired': cohens_d,
                'mean_diff': float(md),
                'sd_diff': float(sd)
            })
            results[metric] = res
        # save JSON
        p = os.path.join(self.output_folder, 'stat_tests_paired.json')
        with open(p, 'w') as f:
            json.dump(results, f, indent=2)
        self.summary['paired_tests'] = p
        return results


if __name__ == '__main__':
    print("This module provides PlotComprehensive Analysis for import.")
