"""
Wage explorer: a Streamlit front end for the Mincer-style log-wage regression.

The fitted statsmodels results object is loaded from a pickle file that sits next
to this script, so no coefficients are hard-coded. Predictions, confidence
intervals and gap tests all come from the model's own parameters and its stored
(cluster-robust) covariance matrix.

Run locally:   streamlit run app.py
Requirements:  streamlit, plotly, numpy, pandas, statsmodels (see requirements.txt)
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =====================================================================
# 1. SETTINGS
# =====================================================================
MODEL_FILE = None   # e.g. "mincer_model.pkl". None = first *.pkl / *.pickle next to this file
WAGE_UNIT = "month"  # display label only; change if your earnings variable is not monthly

REQUIRED = [
    "Intercept", "Category_Dummy", "Sector_Dummy", "Urban_Dummy", "Interaction_Term",
    "Age", "Age_Sq", "Secondary_Plus", "Graduate", "PostGraduate",
]
EDU_LABELS = [
    "Below secondary (up to middle school)",
    "Secondary, higher secondary or diploma",
    "Graduate (bachelor's)",
    "Postgraduate and above",
]
EDU_SHORT = ["Below secondary", "Secondary / diploma", "Graduate", "Postgraduate"]
EDU_PARAM = [None, "Secondary_Plus", "Graduate", "PostGraduate"]  # None = baseline
SECTORS = ["Public sector", "Private sector"]
GROUPS = ["General", "SC / ST / OBC"]
AREAS = ["Rural", "Urban"]
Z95 = 1.96


# =====================================================================
# 2. MODEL WRAPPER
# =====================================================================
class ModelError(Exception):
    """A problem with the model file that the user can fix."""


def pct(b):
    """Log-point coefficient -> % change in wage."""
    return math.expm1(b) * 100


class WageModel:
    def __init__(self, params, cov, nobs, r2, resid_sd, source):
        self.params = params
        self.cov = cov
        self.bse = pd.Series(np.sqrt(np.diag(cov.values)), index=params.index)
        self.nobs, self.r2, self.resid_sd, self.source = nobs, r2, resid_sd, source
        self.extra = [k for k in params.index if k not in REQUIRED]

    @classmethod
    def from_results(cls, obj, source):
        if not (hasattr(obj, "params") and hasattr(obj, "cov_params")):
            raise ModelError(
                f"{source} does not look like a fitted statsmodels results object "
                "(it has no .params / .cov_params())."
            )
        params = pd.Series(obj.params)
        if not all(isinstance(k, str) for k in params.index):
            raise ModelError("The model's parameters have no names. Fit it with a formula so that "
                             "results.params is indexed by variable name.")
        missing = [k for k in REQUIRED if k not in params.index]
        if missing:
            raise ModelError(f"The model is missing these terms: {missing}. "
                             f"It contains: {list(params.index)}.")
        cov = pd.DataFrame(obj.cov_params())
        cov = cov.loc[params.index, params.index]

        def attempt(fn):
            try:
                v = fn()
                return None if v is None or (isinstance(v, float) and math.isnan(v)) else v
            except Exception:
                return None

        nobs = attempt(lambda: int(obj.nobs))
        r2 = attempt(lambda: float(obj.rsquared))
        resid_sd = attempt(lambda: math.sqrt(float(obj.scale)))
        if resid_sd is None:  # back it out of the log-likelihood instead
            resid_sd = attempt(lambda: math.sqrt(math.exp(
                -2 * float(obj.llf) / float(obj.nobs) - math.log(2 * math.pi) - 1)))
        return cls(params, cov, nobs, r2, resid_sd, source)

    # ---- predictions ----
    def vec(self, age, edu, private, urban, reserved):
        x = pd.Series(0.0, index=self.params.index)
        x["Intercept"], x["Age"], x["Age_Sq"] = 1.0, age, age ** 2
        if EDU_PARAM[edu]:
            x[EDU_PARAM[edu]] = 1.0
        x["Sector_Dummy"], x["Urban_Dummy"] = private, urban
        x["Category_Dummy"], x["Interaction_Term"] = reserved, reserved * private
        return x

    def log_wage(self, age, edu, private, urban, reserved):
        return float(self.vec(age, edu, private, urban, reserved).values @ self.params.values)

    def wage(self, age, edu, private, urban, reserved):
        """exp(log wage): the typical (median) wage, not the mean."""
        return math.exp(self.log_wage(age, edu, private, urban, reserved))

    def log_wage_se(self, age, edu, private, urban, reserved):
        x = self.vec(age, edu, private, urban, reserved).values
        return math.sqrt(max(float(x @ self.cov.values @ x), 0.0))

    def contributions(self, age, edu, private, urban, reserved):
        p = self.params
        return {
            "Baseline": p["Intercept"],
            "Age": p["Age"] * age + p["Age_Sq"] * age ** 2,
            "Education": p[EDU_PARAM[edu]] if EDU_PARAM[edu] else 0.0,
            "Sector": p["Sector_Dummy"] * private,
            "Urban": p["Urban_Dummy"] * urban,
            "Social group": p["Category_Dummy"] * reserved + p["Interaction_Term"] * reserved * private,
        }

    # ---- linear combinations of coefficients (with inference) ----
    def effect(self, weights):
        """Wage effect (%) of a linear combination of coefficients, with 95% CI and p-value."""
        w = pd.Series(0.0, index=self.params.index)
        for k, v in weights.items():
            w[k] = v
        est = float(w.values @ self.params.values)
        se = math.sqrt(max(float(w.values @ self.cov.values @ w.values), 0.0))
        p = math.erfc(abs(est / se) / math.sqrt(2)) if se > 0 else float("nan")
        return dict(est=est, se=se, pct=pct(est), lo=pct(est - Z95 * se), hi=pct(est + Z95 * se), p=p)

    def sector_effect(self, reserved):
        return self.effect({"Sector_Dummy": 1, "Interaction_Term": reserved})

    def group_effect(self, private):
        return self.effect({"Category_Dummy": 1, "Interaction_Term": private})

    def edu_effect(self, edu):
        return self.effect({EDU_PARAM[edu]: 1}) if EDU_PARAM[edu] else None

    def peak_age(self):
        return -self.params["Age"] / (2 * self.params["Age_Sq"])


# =====================================================================
# 3. FORMATTING HELPERS
# =====================================================================
def inr(x):
    """Rupees with Indian digit grouping: 123456 -> ₹1,23,456."""
    n = int(round(x))
    s = str(abs(n))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts + [tail])
    return ("-" if n < 0 else "") + "₹" + s


def pct_str(p, digits=1):
    return f"{'+' if p >= 0 else '−'}{abs(p):.{digits}f}%"


def p_str(p):
    return "p &lt; 0.001" if p < 0.001 else f"p = {p:.3f}"


def flat(html):
    """Collapse indented HTML to one line so Markdown never treats it as a code block."""
    return " ".join(line.strip() for line in html.strip().splitlines() if line.strip())


def find_model_file():
    here = Path(__file__).resolve().parent
    if MODEL_FILE:
        path = here / MODEL_FILE
        if not path.exists():
            raise ModelError(f"MODEL_FILE is set to '{MODEL_FILE}' but no such file sits next to app.py.")
        return path
    found = sorted([*here.glob("*.pkl"), *here.glob("*.pickle")])
    if not found:
        raise ModelError("No model file found. Put your pickle (.pkl or .pickle) in the same folder "
                         "as app.py, or set MODEL_FILE at the top of app.py.")
    return found[0]


@st.cache_resource(show_spinner=False)
def load_model(path_str):
    obj = pd.read_pickle(path_str)
    return WageModel.from_results(obj, Path(path_str).name)


# ===== UI =====
NAVY, SAFFRON, BRICK, SLATE = "#12284C", "#E8891D", "#B4442F", "#5B6B82"

st.set_page_config(page_title="Wage model explorer", page_icon="₹", layout="wide")


def get_model():
    try:
        return load_model(str(find_model_file()))
    except ModelError as e:
        st.error(str(e))
    except ModuleNotFoundError as e:
        st.error(f"The pickle needs a package that is not installed: {e.name}. "
                 "Add it to requirements.txt (statsmodels, pinned to the version you fitted with).")
    except Exception as e:  # unpickling errors are usually version mismatches
        st.error(f"Could not load the model file ({type(e).__name__}: {e}). "
                 "This usually means the statsmodels or pandas version here differs from the one "
                 "that saved the pickle. Pin the same versions in requirements.txt.")
    st.stop()


model = get_model()

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=Public+Sans:wght@400;500;600&display=swap');
.stApp { font-family: 'Public Sans', system-ui, sans-serif; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1220px; }
.hero { background:#12284C; color:#fff; border-radius:12px; padding:26px 32px;
        border-left:8px solid #E8891D; margin-bottom:1.3rem; }
.hero h1 { font-family:'Fraunces',Georgia,serif; font-weight:600; font-size:2rem;
           line-height:1.2; margin:0 0 8px 0; padding:0; color:#fff; }
.hero p { margin:0; color:#D5DEEC; font-size:.98rem; max-width:64ch; line-height:1.55; }
.result { background:#fff; border:1px solid #DCE3EC; border-top:6px solid #12284C;
          border-radius:12px; padding:22px 26px; color:#1B2A41; }
.result .label { font-size:.95rem; color:#5B6B82; }
.result .amount { font-family:'Fraunces',Georgia,serif; font-weight:600; font-size:3.5rem;
                  line-height:1.1; color:#12284C; margin:2px 0 8px 0; }
.result .range { font-size:.95rem; color:#1B2A41; }
.result .sub { font-size:.82rem; color:#5B6B82; margin-top:6px; line-height:1.45; }
.chips { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin:14px 0 10px 0; }
.chip { background:#F3F6FA; border:1px solid #DCE3EC; border-radius:10px; padding:12px 14px; color:#1B2A41; }
.chip .k { display:block; font-size:.8rem; color:#5B6B82; }
.chip .v { display:block; font-family:'Fraunces',Georgia,serif; font-size:1.65rem; font-weight:600; color:#12284C; }
.chip .v.down { color:#B4442F; }
.chip .n { display:block; font-size:.78rem; color:#5B6B82; line-height:1.4; }
.note { border-left:4px solid #E8891D; background:#FFF6E9; color:#1B2A41; padding:10px 14px;
        border-radius:0 8px 8px 0; font-size:.9rem; margin:8px 0; line-height:1.5; }
.gapgrid { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:12px; margin:10px 0; }
.gap { background:#fff; border:1px solid #DCE3EC; border-radius:10px; padding:14px 16px; color:#1B2A41; }
.gap .k { font-size:.85rem; color:#5B6B82; }
.gap .v { font-family:'Fraunces',Georgia,serif; font-size:2rem; font-weight:600; color:#B4442F; }
.gap .v.flat { color:#5B6B82; }
.gap .n { font-size:.82rem; color:#5B6B82; line-height:1.45; }
</style>
""",
    unsafe_allow_html=True,
)

n_text = f"{model.nobs:,} " if model.nobs else ""
st.markdown(
    flat(f"""
<div class="hero">
  <h1>What does the model expect a regular worker to earn?</h1>
  <p>A Mincer-style wage regression on PLFS data for {n_text}regular wage and salaried
  workers. Set a profile on the left to see the predicted wage and how sector, social
  group and education move it.</p>
</div>"""),
    unsafe_allow_html=True,
)

if model.extra:
    st.warning(f"The model contains terms this form has no input for ({', '.join(model.extra)}). "
               "Predictions set them to zero, so they will be off. Refit without them or extend app.py.")


# ---------- chart builders ----------
def style(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.12, x=0))
    return fig


def chart_waterfall(parts, total):
    names, vals = list(parts), list(parts.values())
    running, cum = 0.0, []
    for v in vals:
        running += v
        cum.append(running)
    fig = go.Figure(go.Waterfall(
        x=names + ["Predicted log wage"], y=vals + [0],
        measure=["absolute"] + ["relative"] * (len(vals) - 1) + ["total"],
        text=[f"{vals[0]:.2f}"] + [f"{v:+.2f}" for v in vals[1:]] + [f"{total:.2f}"],
        textposition="outside",
        connector=dict(line=dict(color="#9AA8BC", width=1)),
        increasing=dict(marker=dict(color=SAFFRON)),
        decreasing=dict(marker=dict(color=BRICK)),
        totals=dict(marker=dict(color=NAVY)),
        hovertemplate="%{x}: %{text}<extra></extra>",
    ))
    fig.update_yaxes(range=[min(7.0, vals[0] - 0.5), max(cum + [total]) + 0.6],
                     title="Log points (axis is cut at the baseline)")
    return style(fig, 360)


def chart_groups(age, edu, urban):
    fig = go.Figure()
    top = 0
    for s, slabel in enumerate(SECTORS):
        ys = [model.wage(age, edu, s, urban, g) for g in (0, 1)]
        top = max(top, *ys)
        fig.add_bar(name=slabel, x=GROUPS, y=ys, marker_color=[NAVY, SAFFRON][s],
                    text=[inr(y) for y in ys], textposition="outside",
                    hovertemplate="%{x}, " + slabel + ": %{text}<extra></extra>")
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, top * 1.18], title=f"Predicted wage (₹ per {WAGE_UNIT})")
    return style(fig, 380)


def chart_age(edu, private, urban, reserved, age):
    ages = np.arange(15, 66)
    mid = np.array([model.wage(a, edu, private, urban, reserved) for a in ages])
    fig = go.Figure()
    if model.resid_sd:
        band = math.exp(model.resid_sd)
        fig.add_trace(go.Scatter(x=ages, y=mid * band, mode="lines", line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=ages, y=mid / band, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(232,137,29,0.20)",
                                 name="Typical range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ages, y=mid, mode="lines", line=dict(color=NAVY, width=3),
                             name="Predicted wage",
                             hovertemplate="Age %{x}: ₹%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[age], y=[model.wage(age, edu, private, urban, reserved)],
                             mode="markers", name="Selected age",
                             marker=dict(size=13, color=SAFFRON, line=dict(color=NAVY, width=2)),
                             hovertemplate="Age %{x}: ₹%{y:,.0f}<extra></extra>"))
    peak = model.peak_age()
    if 15 <= peak <= 65:
        fig.add_vline(x=peak, line_dash="dot", line_color=SLATE,
                      annotation_text=f"Peak at about {peak:.0f}", annotation_position="top left")
    fig.update_xaxes(title="Age")
    fig.update_yaxes(title=f"Wage (₹ per {WAGE_UNIT})")
    return style(fig, 380)


def chart_education(age, private, urban, reserved, current):
    ys = [model.wage(age, e, private, urban, reserved) for e in range(4)]
    colors = [NAVY if e == current else "#A9B7CC" for e in range(4)]
    fig = go.Figure(go.Bar(x=EDU_SHORT, y=ys, marker_color=colors,
                           text=[inr(y) for y in ys], textposition="outside",
                           hovertemplate="%{x}: %{text}<extra></extra>"))
    fig.update_yaxes(range=[0, max(ys) * 1.18], title=f"Predicted wage (₹ per {WAGE_UNIT})")
    return style(fig, 340)


def coefficient_table():
    labels = {
        "Intercept": "Intercept",
        "Category_Dummy": "SC / ST / OBC (vs General)",
        "Sector_Dummy": "Private sector (vs public)",
        "Urban_Dummy": "Urban (vs rural)",
        "Interaction_Term": "SC / ST / OBC × private sector",
        "Age": "Age (per year)",
        "Age_Sq": "Age squared",
        "Secondary_Plus": "Secondary / higher secondary / diploma",
        "Graduate": "Graduate",
        "PostGraduate": "Postgraduate and above",
    }
    fmt = lambda v: f"{v:.4f}" if abs(v) >= 0.001 else f"{v:.6f}"
    rows = []
    for k in model.params.index:
        b, se = model.params[k], model.bse[k]
        p = math.erfc(abs(b / se) / math.sqrt(2)) if se > 0 else float("nan")
        rows.append({
            "Variable": labels.get(k, k),
            "Coefficient": fmt(b),
            "Std. error": f"{se:.3g}",
            "p-value": "<0.001" if p < 0.001 else f"{p:.3f}",
            "95% CI": f"{fmt(b - Z95 * se)} to {fmt(b + Z95 * se)}",
            "Wage effect": "-" if k in ("Intercept", "Age_Sq") else pct_str(pct(b)),
        })
    return pd.DataFrame(rows)


# ---------- layout: inputs on the left, results on the right ----------
left, right = st.columns([1, 2.2], gap="large")

with left:
    with st.container(border=True):
        st.markdown("### Worker profile")
        age = st.slider("Age (years)", 15, 65, 32)
        edu = st.selectbox("Education", range(4), index=2, format_func=lambda i: EDU_LABELS[i])
        sector = st.selectbox("Employment sector", SECTORS, index=1,
                              help="Public = 0, private = 1 in the model (Sector_Dummy).")
        group = st.selectbox("Social group", GROUPS,
                             help="The model pools SC, ST and OBC into one group (Category_Dummy).")
        area = st.radio("Place of residence", AREAS, index=1, horizontal=True)
        observed = st.number_input(
            f"Observed wage to compare (₹ per {WAGE_UNIT}, optional)",
            min_value=0, value=0, step=1000,
            help="Enter an actual wage to see how far it is from the model's prediction. Leave at 0 to skip.",
        )

private, reserved, urban = SECTORS.index(sector), GROUPS.index(group), AREAS.index(area)
pred_log = model.log_wage(age, edu, private, urban, reserved)
pred = math.exp(pred_log)
se_log = model.log_wage_se(age, edu, private, urban, reserved)
ci_lo, ci_hi = math.exp(pred_log - Z95 * se_log), math.exp(pred_log + Z95 * se_log)

with right:
    tab_pred, tab_gap, tab_curves, tab_model = st.tabs(
        ["Prediction", "Social group gap", "Age and education", "Model details"]
    )

    # ---- Prediction ----
    with tab_pred:
        range_line = ""
        if model.resid_sd:
            low, high = pred * math.exp(-model.resid_sd), pred * math.exp(model.resid_sd)
            range_line = (f'<div class="range">Typical range {inr(low)} to {inr(high)}, which covers '
                          "roughly two-thirds of workers with this profile.</div>")
        st.markdown(
            flat(f"""
<div class="result">
  <div class="label">Predicted wage per {WAGE_UNIT}</div>
  <div class="amount">{inr(pred)}</div>
  {range_line}
  <div class="sub">95% confidence interval for this typical wage: {inr(ci_lo)} to {inr(ci_hi)}
  (uncertainty in the estimates only). Log wage {pred_log:.2f}. The typical (median) wage is lower
  than the average because wages are skewed.</div>
</div>"""),
            unsafe_allow_html=True,
        )

        sec, grp, ed = model.sector_effect(reserved), model.group_effect(private), model.edu_effect(edu)
        ci = lambda e: f"95% CI {pct_str(e['lo'])} to {pct_str(e['hi'])}"
        edu_chip = (
            f'<span class="v">{pct_str(ed["pct"])}</span><span class="n">{EDU_SHORT[edu]}, {ci(ed)}</span>'
            if ed else '<span class="v">Baseline</span><span class="n">The comparison group</span>'
        )
        st.markdown(
            flat(f"""
<div class="chips">
  <div class="chip"><span class="k">Private vs public sector</span>
    <span class="v">{pct_str(sec['pct'])}</span>
    <span class="n">for {group} workers, {ci(sec)}</span></div>
  <div class="chip"><span class="k">SC / ST / OBC vs General</span>
    <span class="v {'down' if grp['p'] < 0.05 and grp['pct'] < 0 else ''}">{pct_str(grp['pct'])}</span>
    <span class="n">in the {sector.lower()}, {ci(grp)}, {p_str(grp['p'])}</span></div>
  <div class="chip"><span class="k">Education vs below secondary</span>{edu_chip}</div>
</div>"""),
            unsafe_allow_html=True,
        )

        if observed > 0:
            diff = math.expm1(math.log(observed) - pred_log) * 100
            r2_text = (f" The model explains about {model.r2:.0%} of the variation in log wages, so"
                       " individual wages often sit well away from the prediction.") if model.r2 else ""
            st.markdown(
                flat(f"""<div class="note">The wage you entered ({inr(observed)}) is
                <b>{abs(diff):.0f}% {'above' if diff >= 0 else 'below'}</b> what the model expects
                for this profile.{r2_text}</div>"""),
                unsafe_allow_html=True,
            )

        st.markdown("**How the prediction builds up**")
        st.plotly_chart(chart_waterfall(model.contributions(age, edu, private, urban, reserved), pred_log),
                        config={"displayModeBar": False}, key="waterfall")
        st.caption("Each step is a coefficient times your input, in log points. Log points add up; "
                   "the wage itself is exp(total).")

    # ---- Social group gap ----
    with tab_gap:
        st.markdown(f"**Same profile ({age} years, {EDU_LABELS[edu].split(' (')[0].lower()}, "
                    f"{area.lower()}), four combinations**")
        st.plotly_chart(chart_groups(age, edu, urban), config={"displayModeBar": False}, key="groups")

        g_pub, g_priv = model.group_effect(0), model.group_effect(1)
        inter = model.effect({"Interaction_Term": 1})
        ci = lambda e: f"95% CI {pct_str(e['lo'])} to {pct_str(e['hi'])}"
        st.markdown(
            flat(f"""
<div class="gapgrid">
  <div class="gap"><div class="k">Gap in the public sector</div>
    <div class="v {'flat' if g_pub['p'] >= 0.05 else ''}">{pct_str(g_pub['pct'])}</div>
    <div class="n">{ci(g_pub)}, {p_str(g_pub['p'])}.
    {'Not distinguishable from zero.' if g_pub['p'] >= 0.05 else 'Statistically significant.'}</div></div>
  <div class="gap"><div class="k">Gap in the private sector</div>
    <div class="v {'flat' if g_priv['p'] >= 0.05 else ''}">{pct_str(g_priv['pct'])}</div>
    <div class="n">{ci(g_priv)}, {p_str(g_priv['p'])}. Group coefficient plus the interaction term.</div></div>
  <div class="gap"><div class="k">Extra gap in private vs public</div>
    <div class="v {'flat' if inter['p'] >= 0.05 else ''}">{pct_str(inter['pct'])}</div>
    <div class="n">The interaction term: {ci(inter)}, {p_str(inter['p'])}.</div></div>
</div>"""),
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="note">Intervals and p-values use the covariance matrix stored in the model, '
            "so they inherit its clustered standard errors.</div>",
            unsafe_allow_html=True,
        )

    # ---- Age and education ----
    with tab_curves:
        st.markdown("**Age-wage profile for this worker**")
        st.plotly_chart(chart_age(edu, private, urban, reserved, age),
                        config={"displayModeBar": False}, key="age")
        band_text = (f" The shaded band is one residual standard deviation ({model.resid_sd:.2f} log "
                     "points) either side of the prediction.") if model.resid_sd else ""
        st.caption(f"Wages rise with age and flatten near {model.peak_age():.0f}.{band_text}")
        st.markdown("**Education ladder at the selected age**")
        st.plotly_chart(chart_education(age, private, urban, reserved, edu),
                        config={"displayModeBar": False}, key="edu")

    # ---- Model details ----
    with tab_model:
        st.markdown("**Regression coefficients** (dependent variable: log wage)")
        st.dataframe(coefficient_table(), hide_index=True)
        stats = [f"N = {model.nobs:,}" if model.nobs else None,
                 f"R² = {model.r2:.3f}" if model.r2 else None,
                 f"loaded from {model.source}"]
        st.caption(". ".join(s for s in stats if s) + ". Wage effect = exp(coefficient) − 1.")
        st.markdown(
            """
**What it is not.** The coefficients are conditional associations, not causal effects. Workers choose
(and are selected into) sectors and education, and SC, ST and OBC are pooled into one group.

**Prediction.** `exp(predicted log wage)` estimates the median wage for the profile.
"""
        )
        with st.expander("Refreshing the model file"):
            st.code(
                "results.summary()          # fills in cached stats first\n"
                "results.remove_data()      # drops the survey rows from the pickle\n"
                "results.save('mincer_model.pkl')",
                language="python",
            )
            st.caption("Commit the new file next to app.py and the app picks it up on the next deploy.")

st.caption("Built from the Mincer wage model notebook. Predictions describe model averages, not individuals.")
