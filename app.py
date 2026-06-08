"""
app.py - Surface Code Decoder Demo (multi-model)
================================================

Interactive Streamlit demo for the surface-decoder-ml project.

Loads pre-trained neural decoders (MLP, CNN, Transformer) saved from the
Phase 3 notebooks, and compares the one you pick against the MWPM baseline
on a freshly simulated quantum error-correction experiment.

Run with:
    streamlit run app.py

Models expected in:  ./models/{mlp_d3,cnn_d5,transformer_d5}.pt
(generate them with the cells in SAVE_SNIPPETS.txt)
"""

from pathlib import Path

import numpy as np
import streamlit as st
import stim
import pymatching
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


st.set_page_config(page_title="Surface Code Decoder Demo", layout="wide")

MODELS_DIR = Path(__file__).parent / "models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---- Model architectures (must match the notebooks) ----
class MLP(nn.Module):
    def __init__(self, n_features, hidden=256, depth=3, dropout=0.2, **_):
        super().__init__()
        layers, in_dim = [], n_features
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CNN(nn.Module):
    def __init__(self, in_channels, hidden=64, **_):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerDec(nn.Module):
    def __init__(self, det_per_round, d_model=64, nhead=4, num_layers=2,
                 dim_ff=128, dropout=0.1, **_):
        super().__init__()
        self.embed = nn.Linear(det_per_round, d_model)
        self.posenc = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        h = self.embed(x)
        h = self.posenc(h)
        h = self.encoder(h)
        h = h.mean(dim=1)
        return self.head(h).squeeze(-1)


# ---- Load saved models ----
@st.cache_resource(show_spinner=False)
def load_model(path_str):
    blob = torch.load(path_str, map_location=DEVICE, weights_only=True)
    kind = blob["kind"]
    hp = blob.get("hparams", {})
    if kind == "mlp":
        model = MLP(n_features=blob["n_features"],
                    **{k: hp[k] for k in ("hidden", "depth", "dropout") if k in hp})
    elif kind == "cnn":
        model = CNN(in_channels=hp.get("in_channels", blob["T"]),
                    hidden=hp.get("hidden", 64))
    elif kind == "transformer":
        model = TransformerDec(
            det_per_round=blob["det_per_round"],
            d_model=hp.get("d_model", 64), nhead=hp.get("nhead", 4),
            num_layers=hp.get("num_layers", 2), dim_ff=hp.get("dim_ff", 128),
            dropout=hp.get("dropout", 0.1),
        )
    else:
        raise ValueError("Unknown model kind: " + str(kind))
    model.load_state_dict(blob["state_dict"])
    model.to(DEVICE).eval()
    return model, blob


def discover_models():
    found = {}
    label_map = {"mlp": "MLP", "cnn": "CNN", "transformer": "Transformer"}
    if MODELS_DIR.exists():
        for p in sorted(MODELS_DIR.glob("*.pt")):
            try:
                blob = torch.load(str(p), map_location="cpu", weights_only=True)
                kind = blob.get("kind", p.stem)
                d, r, pp = blob.get("distance"), blob.get("rounds"), blob.get("noise_p")
                label = "{}  (d={}, T={}, p={:.0e})".format(label_map.get(kind, kind), d, r, pp)
                found[label] = str(p)
            except Exception:
                continue
    return found


# ---- QEC helpers ----
@st.cache_data(show_spinner=False)
def build_circuit(distance, rounds, noise_p):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance, rounds=rounds,
        after_clifford_depolarization=noise_p,
        after_reset_flip_probability=noise_p,
        before_measure_flip_probability=noise_p,
        before_round_data_depolarization=noise_p,
    )


@st.cache_data(show_spinner=False)
def generate_shots(distance, rounds, noise_p, n_shots, seed):
    circuit = build_circuit(distance, rounds, noise_p)
    sampler = circuit.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots=n_shots, separate_observables=True)
    return np.asarray(dets, dtype=bool), np.asarray(obs, dtype=bool).ravel()


def mwpm_decode(distance, rounds, noise_p, X):
    dem = build_circuit(distance, rounds, noise_p).detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    return np.asarray(matching.decode_batch(X)).ravel().astype(bool)


def neural_decode(model, blob, X):
    kind = blob["kind"]
    if kind == "mlp":
        feats = X.astype(np.float32)
    elif kind == "cnn":
        T, H, W = blob["T"], blob["H"], blob["W"]
        dst_t = np.array(blob["dst_t"]); dst_r = np.array(blob["dst_r"]); dst_c = np.array(blob["dst_c"])
        feats = np.zeros((X.shape[0], T, H, W), dtype=np.float32)
        feats[:, dst_t, dst_r, dst_c] = X.astype(np.float32)
    elif kind == "transformer":
        dpr = blob["det_per_round"]
        n_rounds = X.shape[1] // dpr
        feats = X.astype(np.float32).reshape(X.shape[0], n_rounds, dpr)
    else:
        raise ValueError(kind)
    preds = []
    with torch.no_grad():
        for i in range(0, len(feats), 8192):
            xb = torch.from_numpy(feats[i:i + 8192]).to(DEVICE)
            preds.append(torch.sigmoid(model(xb)).cpu().numpy())
    return (np.concatenate(preds) > 0.5).astype(bool)


def ler_with_ci(n_errors, n_total, z=1.96):
    if n_total == 0:
        return 0.0, 0.0, 0.0
    rate = n_errors / n_total
    denom = 1 + z**2 / n_total
    center = (rate + z**2 / (2 * n_total)) / denom
    margin = (z / denom) * np.sqrt(rate * (1 - rate) / n_total + z**2 / (4 * n_total**2))
    return rate, max(0.0, center - margin), center + margin


def plot_syndrome(distance, rounds, noise_p):
    circuit = build_circuit(distance, rounds, noise_p)
    coords = circuit.get_detector_coordinates()
    n_det = circuit.num_detectors
    sampler = circuit.compile_detector_sampler(seed=np.random.randint(0, 10**6))
    dets, _ = sampler.sample(shots=2000, separate_observables=True)
    dets = np.asarray(dets, dtype=bool)
    nz = np.where(dets.any(axis=1))[0]
    shot = dets[nz[0]] if len(nz) else dets[0]
    coord_arr = np.array([coords[i] for i in range(n_det)])
    ts = coord_arr[:, 2].astype(int)
    mask = ts == sorted(set(ts))[0]
    xs, ys, fired = coord_arr[mask, 0], coord_arr[mask, 1], shot[mask]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(xs[~fired], ys[~fired], s=260, facecolors="none",
               edgecolors="#999999", linewidths=1.5, label="quiet stabilizer")
    ax.scatter(xs[fired], ys[fired], s=300, c="#C44E52",
               edgecolors="#7a2f33", linewidths=1.5, label="fired (error detected)")
    ax.set_title("One syndrome on the d=" + str(distance) + " lattice (first round)")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=9, frameon=False)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig, int(fired.sum())


# ---- UI ----
st.title("Surface Code Decoder Demo")
st.markdown(
    "Compare a **neural-network decoder** against the standard "
    "**Minimum-Weight Perfect Matching (MWPM)** algorithm on a simulated "
    "quantum error-correction experiment. Pick which trained decoder to test."
)

models = discover_models()

if not models:
    st.error(
        "No trained models found in `./models/`.\n\n"
        "Run the save cells from **SAVE_SNIPPETS.txt** at the end of each "
        "Phase 3 notebook to create the .pt files, then reload this page."
    )
    st.stop()

with st.sidebar:
    st.header("Choose decoder")
    choice = st.selectbox("Neural decoder to compare vs MWPM", list(models.keys()))
    model, blob = load_model(models[choice])
    distance = blob["distance"]; rounds = blob["rounds"]; noise_p = blob["noise_p"]
    st.divider()
    st.caption("This model was trained for:")
    st.write("- Distance **d = {}**".format(distance))
    st.write("- Rounds **T = {}**".format(rounds))
    st.write("- Physical error rate **p = {:.0e}**".format(noise_p))
    st.caption("The experiment below uses exactly these settings.")
    st.divider()
    n_shots = st.select_slider("Test shots", options=[10_000, 50_000, 100_000, 200_000],
                               value=50_000, format_func=lambda v: "{:,}".format(v))
    run = st.button("Generate & Decode", type="primary", use_container_width=True)

left, right = st.columns([1, 1])
with left:
    st.subheader("What an error looks like")
    fig_syn, n_fired = plot_syndrome(distance, rounds, noise_p)
    st.pyplot(fig_syn)
    st.caption(
        "Each circle is a stabilizer check. Red circles ({} here) detected something "
        "wrong. The decoder reads this pattern and predicts whether the logical qubit "
        "was corrupted.".format(n_fired)
    )
with right:
    st.subheader("How the comparison works")
    st.markdown(
        "- The selected decoder is a **{}**.\n".format(choice.split("  ")[0]) +
        "- **Stim** generates fresh syndromes under circuit-level noise.\n"
        "- **MWPM** decodes them using the known error model (no training).\n"
        "- The neural decoder decodes the same shots.\n"
        "- Both predict whether a **logical error** occurred; lower rate is better.\n\n"
        "At small distances MWPM is near-optimal, so **matching it is a real result**."
    )

st.divider()

if run:
    with st.spinner("Generating fresh quantum error data with Stim..."):
        X_test, y_test = generate_shots(distance, rounds, noise_p, n_shots, seed=7)
    n_test = len(y_test)

    with st.spinner("Decoding with MWPM..."):
        mwpm_pred = mwpm_decode(distance, rounds, noise_p, X_test)
        mwpm_err = int(np.sum(mwpm_pred != y_test))
        mwpm_rate, mwpm_lo, mwpm_hi = ler_with_ci(mwpm_err, n_test)

    nn_name = choice.split("  ")[0]
    with st.spinner("Decoding with the " + nn_name + "..."):
        nn_pred = neural_decode(model, blob, X_test)
        nn_err = int(np.sum(nn_pred != y_test))
        nn_rate, nn_lo, nn_hi = ler_with_ci(nn_err, n_test)

    st.subheader("Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw logical-error events", "{:,}".format(int(y_test.sum())))
    c2.metric("MWPM logical error rate", "{:.4%}".format(mwpm_rate),
              help="{:,} wrong out of {:,}".format(mwpm_err, n_test))
    delta = (nn_rate - mwpm_rate) / mwpm_rate * 100 if mwpm_rate > 0 else 0.0
    c3.metric(nn_name + " logical error rate", "{:.4%}".format(nn_rate),
              delta="{:+.1f}% vs MWPM".format(delta), delta_color="inverse",
              help="{:,} wrong out of {:,}".format(nn_err, n_test))

    overlap = not (mwpm_hi < nn_lo or nn_hi < mwpm_lo)
    if overlap:
        st.success(
            "The 95% confidence intervals overlap: the " + nn_name + " is "
            "**statistically tied** with MWPM at this setting - it has matched a "
            "near-optimal classical decoder."
        )
    else:
        better = nn_name if nn_rate < mwpm_rate else "MWPM"
        st.info("The 95% confidence intervals do not overlap: **" + better + "** is "
                "significantly better at this setting and sample size.")

    fig, ax = plt.subplots(figsize=(7, 4))
    names = ["MWPM", nn_name]
    rates = [mwpm_rate, nn_rate]
    lo = [mwpm_rate - mwpm_lo, nn_rate - nn_lo]
    hi = [mwpm_hi - mwpm_rate, nn_hi - nn_rate]
    bars = ax.bar(names, rates, color=["#C44E52", "#4C72B0"], width=0.55,
                  yerr=[lo, hi], capsize=8, error_kw={"elinewidth": 1.5, "ecolor": "#333"})
    ax.set_ylabel("Logical error rate")
    ax.set_title(nn_name + " vs MWPM at d=" + str(distance) +
                 ", p={:.0e} ({:,} shots, 95% CI)".format(noise_p, n_test))
    for bar, r, hi_val in zip(bars, rates, hi):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            r + hi_val,
            "{:.4%}".format(r),
            ha="center", va="bottom", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.8, alpha=0.92),
        )
    ax.set_ylim(0, max(mwpm_hi, nn_hi, 1e-9) * 1.4)
    plt.tight_layout()
    st.pyplot(fig)

    both_right = int(np.sum((mwpm_pred == y_test) & (nn_pred == y_test)))
    both_wrong = int(np.sum((mwpm_pred != y_test) & (nn_pred != y_test)))
    mwpm_only = int(np.sum((mwpm_pred == y_test) & (nn_pred != y_test)))
    nn_only = int(np.sum((mwpm_pred != y_test) & (nn_pred == y_test)))
    with st.expander("Where do the two decoders agree and disagree?"):
        st.markdown(
            "- Both correct: **{:,}**\n".format(both_right) +
            "- Both wrong (uncorrectable errors): **{:,}**\n".format(both_wrong) +
            "- Only MWPM correct: **{:,}**\n".format(mwpm_only) +
            "- Only {} correct: **{:,}**\n".format(nn_name, nn_only) +
            "- Agreement rate: **{:.4%}**".format((both_right + both_wrong) / n_test)
        )
else:
    st.info("Pick a decoder and shot count in the sidebar, then click **Generate & Decode**.")
