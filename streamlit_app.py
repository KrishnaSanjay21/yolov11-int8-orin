"""YOLOv11s INT8 PTQ — interactive frontend for the host-testable toolkit.

This app runs the *host* side of the project (numpy only): calibration math,
per-class mAP accounting, layer sensitivity, and the fused-DFL plugin reference.
No Jetson / TensorRT / CUDA is required — everything here is what runs and is unit-
tested on a plain CPU. Device-produced numbers (latency/power) are shown only when
you upload the JSON a device run emits; they are never fabricated.

Run locally:   streamlit run streamlit_app.py
"""
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

# Make src/qint importable without an install step (works locally and on Streamlit Cloud).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from qint.calibration.stats import HistogramCollector, MinMaxCollector
from qint.calibration.entropy import entropy_calibrate
from qint.calibration.minmax import minmax_scale, fake_quant
from qint.engine.stub import StubEngine, GTBox, DegradationProfile
from qint.accuracy.metrics import evaluate
from qint.accuracy.diff import diff_precisions
from qint.sensitivity import score_layer, rank_layers, top_k_sensitive, LayerSensitivity
from qint.precision import build_fallback_plan
from qint.plugin.dfl_reference import (
    dfl_fused_reference, dfl_unfused_reference, dfl_torch_reference, TORCH_AVAILABLE, REG_MAX,
)
from qint.report import build_rows, render_benchmarks_table, BenchmarkRow

st.set_page_config(page_title="YOLOv11s INT8 PTQ — Orin NX", page_icon="🟩", layout="wide")

PAGES = [
    "Overview",
    "Calibration explorer",
    "Per-class accuracy",
    "Layer sensitivity",
    "DFL plugin validation",
    "Benchmarks",
]


# ---------------------------------------------------------------------------
def sidebar():
    st.sidebar.title("🟩 YOLOv11s INT8 PTQ")
    st.sidebar.caption("Host-testable toolkit · Jetson Orin NX target")
    page = st.sidebar.radio("Section", PAGES, label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.markdown(
        "**Runs on CPU.** No Orin needed for anything on this site — these are the "
        "numpy modules that are unit-tested on the host. Device latency/power appear "
        "only if you upload a device result JSON."
    )
    st.sidebar.caption(f"PyTorch cross-check available: {'✅' if TORCH_AVAILABLE else '➖ (skipped)'}")
    return page


# ---------------------------------------------------------------------------
def page_overview():
    st.title("YOLOv11s → INT8 on Jetson Orin NX")
    st.markdown(
        "Post-training quantization (FP32 → FP16 → INT8) with **honest per-class "
        "accuracy accounting**, a custom fused TensorRT plugin, calibrator/calibration-"
        "size sweeps, and layer-sensitivity-driven selective FP16 fallback."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("What this frontend does")
        st.markdown(
            "- **Calibration explorer** — entropy (KL) vs min-max on any distribution\n"
            "- **Per-class accuracy** — FP32 vs INT8 mAP per class, >2% flagging\n"
            "- **Layer sensitivity** — SQNR ranking → FP16 fallback plan\n"
            "- **DFL plugin** — fused vs reference op, real max-abs / max-rel error\n"
            "- **Benchmarks** — render a device result JSON into the report table"
        )
    with c2:
        st.subheader("Host vs device")
        st.markdown(
            "Everything here is the **host** side (numpy, CPU, unit-tested). The "
            "**device** side — TensorRT engine builds, INT8 calibration on the GPU, the "
            "CUDA plugin, latency/power — runs on the Orin via committed `# RUN ON "
            "DEVICE` scripts and emits JSON that the Benchmarks page renders. No device "
            "numbers are invented."
        )
    st.divider()
    st.subheader("The custom fused op (kept FP32 on purpose)")
    st.code(
        "input  x : (N, 4*reg_max, A)   raw DFL logits, reg_max=16\n"
        "reshape  -> (N, 4, reg_max, A)\n"
        "softmax over the reg_max axis\n"
        "expectation E[j] = Σ softmax_j · j     ->  (N, 4, A) box distances",
        language="text",
    )


# ---------------------------------------------------------------------------
def _sample_distribution(kind, n, outlier_mag, seed):
    rng = np.random.default_rng(seed)
    if kind == "Gaussian":
        x = rng.normal(0, 1.0, n)
    elif kind == "Heavy-tailed (lognormal)":
        x = rng.lognormal(0, 1.0, n) - np.e ** 0.5
    elif kind == "Bimodal":
        x = np.concatenate([rng.normal(-2, 0.5, n // 2), rng.normal(2, 0.5, n - n // 2)])
    else:  # Post-SiLU-ish (mostly small, positive tail)
        x = rng.normal(0, 1, n)
        x = x * (x > 0) + 0.05 * x * (x <= 0)
    # inject a few outliers to show the entropy-vs-minmax difference
    k = max(1, n // 5000)
    x[rng.integers(0, n, k)] = outlier_mag * np.sign(rng.normal(size=k) + 1e-9)
    return x


def page_calibration():
    st.title("Calibration explorer — entropy (KL) vs min-max")
    st.caption(
        "The mechanism behind the calibrator choice in DECISIONS.md. Min-max never clips "
        "but wastes INT8 levels on outliers; entropy clips the tail to minimize KL "
        "divergence. Watch the entropy threshold sit *below* the absolute max."
    )
    c = st.columns(4)
    kind = c[0].selectbox("Distribution", ["Gaussian", "Heavy-tailed (lognormal)", "Bimodal", "Post-SiLU-ish"])
    n = c[1].select_slider("Samples", [20_000, 50_000, 100_000, 200_000], value=100_000)
    outlier = c[2].slider("Outlier magnitude", 5.0, 100.0, 40.0, 5.0)
    num_bins = c[3].select_slider("Histogram bins", [512, 1024, 2048], value=2048)

    x = _sample_distribution(kind, n, outlier, seed=0)
    hist = HistogramCollector(num_bins=num_bins)
    hist.update(x)
    mm = MinMaxCollector().update(x)

    ent = entropy_calibrate(hist.histogram, hist.upper)
    minmax_s = float(minmax_scale(mm.absmax))

    # quantization error each scale induces on the bulk
    err_entropy = float(np.sqrt(np.mean((x - fake_quant(x, ent.scale)) ** 2)))
    err_minmax = float(np.sqrt(np.mean((x - fake_quant(x, minmax_s)) ** 2)))

    m = st.columns(4)
    m[0].metric("min-max abs-max", f"{mm.absmax:.2f}")
    m[1].metric("entropy clip threshold", f"{ent.threshold_value:.2f}", f"{ent.threshold_value - mm.absmax:.2f} vs abs-max")
    m[2].metric("entropy scale", f"{ent.scale:.4f}", f"{ent.scale - minmax_s:.4f} vs min-max")
    m[3].metric("RMSE (bulk): entropy vs min-max", f"{err_entropy:.4f}", f"{err_entropy - err_minmax:+.4f}",
                delta_color="inverse")

    left, right = st.columns(2)
    with left:
        st.markdown("**|activation| histogram** (clip thresholds marked as scaled positions)")
        edges = hist.bin_edges()[:-1]
        dfh = pd.DataFrame({"|activation|": edges, "count": hist.histogram})
        st.bar_chart(dfh.set_index("|activation|"), height=280)
        st.caption(
            f"min-max keeps the full range up to {mm.absmax:.1f}; entropy clips at "
            f"{ent.threshold_value:.1f} (bin {ent.threshold_bin}/{num_bins})."
        )
    with right:
        st.markdown("**KL divergence vs clip threshold** (entropy picks the minimum)")
        thr_axis = np.arange(len(ent.kl)) + ent.target_bins
        dfk = pd.DataFrame({"threshold bin": thr_axis, "KL divergence": ent.kl})
        dfk = dfk.replace([np.inf, -np.inf], np.nan).dropna()
        st.line_chart(dfk.set_index("threshold bin"), height=280)
        st.caption(f"KL-minimizing threshold at bin {ent.threshold_bin}.")

    st.info(
        "**Takeaway:** on a heavy-tailed tensor, entropy's smaller scale gives lower "
        "quantization error on the bulk of values (where the signal is), at the cost of "
        "clipping rare outliers. On a clean Gaussian the two nearly agree.",
        icon="💡",
    )


# ---------------------------------------------------------------------------
def _grid_gt(n_images, n_classes):
    gts = []
    for img in range(n_images):
        for c in range(n_classes):
            x = 20 + 40 * c
            y = 20 + 30 * (img % 8)
            gts.append(GTBox(image_id=img, class_id=c, box=(x, y, x + 24, y + 24)))
    return gts


def page_accuracy():
    st.title("Per-class accuracy — FP32 vs INT8 (stub engine)")
    st.caption(
        "The 'honest per-class accounting' made interactive. A StubEngine synthesizes "
        "detections from ground truth: FP32 is perfect; the INT8 config degrades classes "
        "you choose. Any class losing >2% absolute AP is flagged 🚩 — aggregate-only "
        "numbers are exactly what this page refuses to show."
    )
    c = st.columns(3)
    n_classes = c[0].slider("Classes", 3, 12, 6)
    n_images = c[1].slider("Images", 10, 200, 60, 10)
    class_names = [f"class_{i}" for i in range(n_classes)]

    st.markdown("**INT8 degradation** — pick classes to hurt and how (simulating quantization damage):")
    d = st.columns(4)
    hurt = d[0].multiselect("Degraded classes", class_names, default=[class_names[-1]])
    drop = d[1].slider("Missed-detection prob", 0.0, 0.9, 0.4, 0.05)
    jitter = d[2].slider("Localization jitter", 0.0, 0.8, 0.35, 0.05)
    bias = d[3].slider("Confidence drop", 0.0, 0.6, 0.25, 0.05)

    hurt_ids = [class_names.index(h) for h in hurt]
    prof = DegradationProfile(
        drop_prob={i: drop for i in hurt_ids},
        loc_jitter={i: jitter for i in hurt_ids},
        score_bias={i: -bias for i in hurt_ids},
    )
    gts = _grid_gt(n_images, n_classes)
    ids = sorted({g.image_id for g in gts})

    fp32 = StubEngine(gts, n_classes, "fp32")
    int8 = StubEngine(gts, n_classes, "int8", profile=prof, seed=1)
    r_fp32 = evaluate(fp32.infer_dataset(None, ids), gts, n_classes, "fp32")
    r_int8 = evaluate(int8.infer_dataset(None, ids), gts, n_classes, "int8")
    dd = diff_precisions(r_fp32, r_int8, class_names=class_names, threshold=0.02)

    m = st.columns(3)
    m[0].metric("mAP@50  FP32→INT8", f"{100*dd.cand_map50:.1f}%", f"{100*dd.d_map50:+.1f}%", delta_color="normal")
    m[1].metric("mAP@50-95  FP32→INT8", f"{100*dd.cand_map5095:.1f}%", f"{100*dd.d_map5095:+.1f}%", delta_color="normal")
    m[2].metric("Classes flagged >2%", f"{len(dd.flagged_classes)}")

    rows = []
    for de in dd.deltas:
        rows.append({
            "class": de.class_name,
            "AP@50 FP32": 100 * de.base_ap50, "AP@50 INT8": 100 * de.cand_ap50,
            "ΔAP@50": 100 * de.d_ap50,
            "AP@50-95 FP32": 100 * de.base_ap5095, "AP@50-95 INT8": 100 * de.cand_ap5095,
            "ΔAP@50-95": 100 * de.d_ap5095,
            "flag": "🚩" if de.flagged(0.02) else "",
        })
    df = pd.DataFrame(rows).set_index("class")
    st.dataframe(
        df.style.format("{:.2f}", subset=[c for c in df.columns if c != "flag"])
        .map(lambda v: "color:#ff6b6b" if isinstance(v, (int, float)) and v < -2 else "",
             subset=["ΔAP@50", "ΔAP@50-95"]),
        use_container_width=True,
    )
    st.markdown("**ΔAP@50-95 per class** (negative = degraded)")
    st.bar_chart(df[["ΔAP@50-95"]], height=260, color="#76b900")


# ---------------------------------------------------------------------------
def page_sensitivity():
    st.title("Layer sensitivity → selective FP16 fallback")
    st.caption(
        "Each synthetic layer's INT8 SQNR (dB) is scored; the lowest-SQNR layers are the "
        "worst-represented and become FP16-fallback candidates. The top-K feed a "
        "PrecisionPlan the device builder applies (`06_build_mixed.py`)."
    )
    c = st.columns(3)
    n_layers = c[0].slider("Layers", 6, 40, 16)
    topk = c[1].slider("FP16 top-K", 1, 8, 5)
    seed = c[2].number_input("Seed", 0, 9999, 0)

    rng = np.random.default_rng(seed)
    layers = []
    for i in range(n_layers):
        # vary the tail heaviness so SQNR spreads realistically
        heavy = rng.uniform(0, 1) ** 2
        a = rng.normal(0, 1, 4000)
        n_out = int(heavy * 30)
        if n_out:
            a = np.concatenate([a, rng.uniform(10, 60, n_out)])
        layers.append(score_layer(f"layer_{i:02d}", a))

    ranked = rank_layers(layers)
    plan = build_fallback_plan(layers, k=topk)

    df = pd.DataFrame([
        {"layer": l.name, "SQNR (dB)": (np.nan if np.isinf(l.sqnr_db) else l.sqnr_db),
         "MSE": l.mse, "scale": l.scale, "FP16": "✅" if l.name in set(plan.fp16_layers) else ""}
        for l in ranked
    ]).set_index("layer")
    left, right = st.columns([2, 1])
    with left:
        st.markdown("**Layers ranked by SQNR (ascending = most sensitive first)**")
        st.dataframe(df.style.format({"SQNR (dB)": "{:.2f}", "MSE": "{:.2e}", "scale": "{:.4f}"}),
                     use_container_width=True, height=380)
    with right:
        st.markdown("**SQNR by layer**")
        st.bar_chart(df[["SQNR (dB)"]], height=380, color="#76b900")

    st.markdown(f"**Generated PrecisionPlan** (top-{topk} → FP16):")
    st.code(plan.to_json(), language="json")


# ---------------------------------------------------------------------------
def page_plugin():
    st.title("Fused DFL plugin — numeric validation")
    st.caption(
        "The custom TensorRT plugin fuses softmax+expectation. Here the fused, kernel-"
        "faithful implementation is validated against the plain reference with a real "
        "max-abs / max-rel error (the same check scripts/validate_plugin.py runs on the "
        "GPU against the CUDA kernel)."
    )
    c = st.columns(4)
    n = c[0].slider("Batch", 1, 8, 2)
    a = c[1].select_slider("Anchors", [50, 100, 500, 1000, 8400], value=100)
    scale = c[2].slider("Logit scale (σ)", 1.0, 12.0, 6.0, 1.0)
    seed = c[3].number_input("Seed", 0, 9999, 1)

    rng = np.random.default_rng(seed)
    x = rng.normal(0, scale, size=(n, 4 * REG_MAX, a)).astype(np.float32)
    fused = dfl_fused_reference(x)
    unfused = dfl_unfused_reference(x)
    max_abs = float(np.max(np.abs(fused - unfused)))
    max_rel = float(np.max(np.abs(fused - unfused) / np.maximum(np.abs(unfused), 1e-8)))

    m = st.columns(3)
    m[0].metric("max abs error", f"{max_abs:.2e}")
    m[1].metric("max rel error", f"{max_rel:.2e}")
    tol = 1e-3
    m[2].metric("tolerance 1e-3", "PASS ✅" if (max_abs < tol and max_rel < tol) else "FAIL ❌")

    if TORCH_AVAILABLE:
        t = dfl_torch_reference(x)
        ta = float(np.max(np.abs(fused - t)))
        st.success(f"PyTorch cross-check available — fused vs torch max abs = {ta:.2e}", icon="🔬")
    else:
        st.info("PyTorch not installed in this environment — torch cross-check skipped "
                "(the numpy fused-vs-reference check above is the guaranteed one).", icon="➖")

    st.markdown("**Sample output** — expected box distances (side 0), first 8 anchors of image 0:")
    st.dataframe(pd.DataFrame({
        "anchor": range(min(8, a)),
        "fused": fused[0, 0, :8], "reference": unfused[0, 0, :8],
    }).set_index("anchor").style.format("{:.5f}"), use_container_width=True)

    with st.expander("CUDA kernel (src/qint/plugin/dfl_plugin/dfl_kernel.cu)"):
        st.code(
            "// one thread per (batch, side, anchor)\n"
            "float m = -INFINITY;\n"
            "for (int j=0;j<regMax;++j) m = fmaxf(m, input[base + j*stride]);   // stable max\n"
            "float esum=0.f, wsum=0.f;\n"
            "for (int j=0;j<regMax;++j){\n"
            "  float e = __expf(input[base + j*stride] - m);\n"
            "  esum += e; wsum += e * j;                                        // one pass\n"
            "}\n"
            "output[outIdx] = wsum / esum;                                      // expectation",
            language="cpp",
        )


# ---------------------------------------------------------------------------
def page_benchmarks():
    st.title("Benchmarks — device results")
    st.caption(
        "Latency / throughput / power / memory come from the Orin only. Upload the JSON "
        "files a device run emits (`results/raw/bench_*.json`, `accuracy_*.json`) and "
        "they render into the report table. Nothing here is synthesized."
    )
    bench_files = st.file_uploader("bench_*.json", type="json", accept_multiple_files=True)
    acc_files = st.file_uploader("accuracy_*.json", type="json", accept_multiple_files=True)

    import json
    bench, acc = {}, {}
    for f in bench_files or []:
        cfg = f.name[len("bench_"):-len(".json")] if f.name.startswith("bench_") else f.name[:-5]
        bench[cfg] = json.load(f)
    for f in acc_files or []:
        cfg = f.name[len("accuracy_"):-len(".json")] if f.name.startswith("accuracy_") else f.name[:-5]
        acc[cfg] = json.load(f)

    if not bench and not acc:
        st.warning("No device JSON uploaded yet — showing the empty template shape.", icon="📄")
        demo = [BenchmarkRow("fp32"), BenchmarkRow("fp16"), BenchmarkRow("int8_entropy_512")]
        st.markdown(render_benchmarks_table(demo))
        st.caption("Run scripts/07_run_accuracy.py + 08_benchmark.py on the Orin to produce real rows.")
        return

    def rank(cfg):
        return (0 if cfg == "fp32" else 1 if cfg == "fp16" else 2, cfg)
    order = sorted(set(bench) | set(acc), key=rank)
    rows = build_rows(bench, acc, order=order)
    st.markdown(render_benchmarks_table(rows))


# ---------------------------------------------------------------------------
PAGE_FUNCS = {
    "Overview": page_overview,
    "Calibration explorer": page_calibration,
    "Per-class accuracy": page_accuracy,
    "Layer sensitivity": page_sensitivity,
    "DFL plugin validation": page_plugin,
    "Benchmarks": page_benchmarks,
}


def main():
    page = sidebar()
    PAGE_FUNCS[page]()


if __name__ == "__main__":
    main()
