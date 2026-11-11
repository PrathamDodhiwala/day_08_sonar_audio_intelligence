
import io
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import spectrogram

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="SONAR | Audio Intelligence Lab",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM UI
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg: #101018;
    --panel: #191925;
    --panel2: #20202E;
    --border: #303044;
    --text: #F2F0FA;
    --muted: #9694AA;
    --lime: #C5F36A;
    --violet: #A18BFF;
    --pink: #F28ACB;
}
.stApp {
    background: var(--bg);
    color: var(--text);
    font-family: 'Space Grotesk', sans-serif;
}
[data-testid="stSidebar"] {
    background: #15151F;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * {
    color: var(--text);
}
.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3, p, label {
    color: var(--text);
}
.eyebrow {
    color: var(--lime);
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
}
.hero {
    font-size: clamp(40px, 5vw, 66px);
    font-weight: 700;
    letter-spacing: -4px;
    line-height: 1.02;
    margin-top: 8px;
}
.subtitle {
    color: var(--muted);
    max-width: 720px;
    font-size: 14px;
    line-height: 1.8;
    margin-top: 12px;
}
.panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 19px;
}
.metric {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 17px;
    min-height: 105px;
}
.metric-label {
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.2px;
    text-transform: uppercase;
}
.metric-value {
    color: var(--lime);
    font-size: 30px;
    font-weight: 700;
    letter-spacing: -1.5px;
    margin-top: 8px;
}
.metric-note {
    color: var(--muted);
    font-size: 11px;
    margin-top: 2px;
}
.section-heading {
    font-size: 21px;
    font-weight: 700;
    letter-spacing: -0.8px;
    margin-top: 16px;
    margin-bottom: 5px;
}
.mono {
    font-family: 'DM Mono', monospace;
    color: var(--muted);
    font-size: 10px;
}
div[data-testid="stAudio"] {
    border: 1px solid var(--border);
    border-radius: 12px;
}
.stButton > button, .stDownloadButton > button {
    background: var(--lime);
    color: #15151F;
    border: 0;
    border-radius: 8px;
    font-weight: 700;
    min-height: 42px;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: #D8FF8A;
    color: #15151F;
    border: 0;
}
[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 10px;
}
hr {
    border-color: var(--border);
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# CONSTANTS AND HELPERS
# =========================================================
PLOT_BG = "#191925"
GRID = "#303044"
LIME = "#C5F36A"
VIOLET = "#A18BFF"
PINK = "#F28ACB"
TEXT = "#F2F0FA"
MUTED = "#9694AA"


def make_demo_audio():
    """Create a synthetic audio sample with several sound events."""
    sr = 22050
    duration = 9
    t = np.arange(int(sr * duration)) / sr
    rng = np.random.default_rng(42)

    audio = (
        0.08 * np.sin(2 * np.pi * 110 * t)
        + 0.025 * np.sin(2 * np.pi * 220 * t)
        + 0.008 * rng.normal(size=len(t))
    )

    # Short noise burst
    burst = (t >= 2.0) & (t <= 2.65)
    audio[burst] += (
        0.35 * rng.normal(size=burst.sum())
        * np.hanning(burst.sum())
    )

    # A rising-frequency tone
    chirp = (t >= 5.0) & (t <= 6.2)
    chirp_t = t[chirp] - 5.0
    audio[chirp] += (
        0.24 * np.sin(
            2 * np.pi * (300 * chirp_t + 450 * chirp_t**2)
        )
    )

    # Another brief impulse
    impulse = (t >= 7.35) & (t <= 7.55)
    audio[impulse] += (
        0.5 * np.sin(2 * np.pi * 900 * t[impulse])
    )

    audio = audio / max(np.max(np.abs(audio)), 1e-9) * 0.92
    return audio.astype(np.float32), sr


def load_audio(file_bytes):
    audio, sr = sf.read(io.BytesIO(file_bytes), dtype="float32")

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    if len(audio) == 0:
        raise ValueError("The uploaded audio file is empty.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("The audio contains invalid numeric values.")

    return audio.astype(np.float32), int(sr)


def make_audio_bytes(audio, sr):
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV")
    return buffer.getvalue()


def extract_features(audio, sr, window_seconds, hop_seconds):
    """Calculate features for overlapping audio windows."""
    window = max(16, int(window_seconds * sr))
    hop = max(1, int(hop_seconds * sr))

    if len(audio) < window:
        audio = np.pad(audio, (0, window - len(audio)))

    rows = []

    for start in range(0, len(audio) - window + 1, hop):
        segment = audio[start:start + window]
        start_time = start / sr
        end_time = (start + window) / sr

        rms = float(np.sqrt(np.mean(segment ** 2)))
        peak = float(np.max(np.abs(segment)))

        signs = np.sign(segment)
        zcr = float(np.mean(signs[:-1] != signs[1:]))

        spectrum = np.abs(np.fft.rfft(segment))
        frequencies = np.fft.rfftfreq(len(segment), d=1 / sr)
        total = spectrum.sum()

        centroid = (
            float(np.sum(frequencies * spectrum) / total)
            if total > 0 else 0.0
        )

        rows.append({
            "Start (s)": start_time,
            "End (s)": end_time,
            "RMS Energy": rms,
            "Peak Amplitude": peak,
            "Zero Crossing Rate": zcr,
            "Spectral Centroid (Hz)": centroid
        })

    return pd.DataFrame(rows)


def add_chart_theme(fig, height=350):
    fig.update_layout(
        height=height,
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT, family="Space Grotesk"),
        margin=dict(l=15, r=15, t=35, b=15),
        xaxis=dict(
            gridcolor=GRID,
            zerolinecolor=GRID,
            title_font=dict(color=MUTED)
        ),
        yaxis=dict(
            gridcolor=GRID,
            zerolinecolor=GRID,
            title_font=dict(color=MUTED)
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT)
        )
    )
    return fig


def metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="metric">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def page_header(kicker, title, description):
    st.markdown(
        f"<div class='eyebrow'>{kicker}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='hero'>{title}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='subtitle'>{description}</div>",
        unsafe_allow_html=True
    )
    st.write("")


# =========================================================
# SIDEBAR — INPUT AND CONTROLS
# =========================================================
st.sidebar.markdown("## ◉ SONAR")
st.sidebar.markdown(
    "<div class='mono'>AUDIO INTELLIGENCE LAB / 008</div>",
    unsafe_allow_html=True
)
st.sidebar.divider()

source = st.sidebar.radio(
    "AUDIO SOURCE",
    ["Synthetic demo", "Upload audio"]
)

if source == "Synthetic demo":
    audio, sample_rate = make_demo_audio()
    audio_name = "sonar_demo_recording.wav"
else:
    uploaded = st.sidebar.file_uploader(
        "Upload audio",
        type=["wav", "flac", "ogg"]
    )

    if uploaded is None:
        page_header(
            "INPUT REQUIRED / 001",
            "Bring sound into focus.",
            "Upload a WAV, FLAC or OGG recording to begin exploring "
            "its signal, frequency content and unusual segments."
        )
        st.info(
            "Choose an audio file from the sidebar. "
            "Or switch to Synthetic demo to explore the app immediately."
        )
        st.stop()

    try:
        audio, sample_rate = load_audio(uploaded.getvalue())
        audio_name = uploaded.name
    except Exception as error:
        st.error(f"Unable to read this audio file: {error}")
        st.stop()

duration = len(audio) / sample_rate

if duration > 300:
    st.error(
        "This demo version supports recordings up to 5 minutes. "
        "Please upload a shorter clip."
    )
    st.stop()

if duration < 0.05:
    st.error("The audio is too short to analyse.")
    st.stop()

st.sidebar.success("Audio loaded")
st.sidebar.caption(f"{audio_name}")
st.sidebar.caption(
    f"{duration:.2f} seconds · {sample_rate:,} Hz"
)

st.sidebar.divider()
st.sidebar.markdown("### ANALYSIS SETTINGS")

window_seconds = st.sidebar.select_slider(
    "Analysis window",
    options=[0.10, 0.20, 0.25, 0.50, 1.0],
    value=0.25,
    format_func=lambda x: f"{x:.2f} sec"
)

hop_seconds = st.sidebar.select_slider(
    "Window step",
    options=[0.05, 0.10, 0.125, 0.20, 0.25],
    value=0.125,
    format_func=lambda x: f"{x:.3f} sec"
)

threshold_multiplier = st.sidebar.slider(
    "Unusual energy threshold",
    min_value=1.0,
    max_value=5.0,
    value=2.5,
    step=0.25
)

# =========================================================
# AUDIO ANALYSIS
# =========================================================
features = extract_features(
    audio,
    sample_rate,
    window_seconds,
    hop_seconds
)

median_rms = float(features["RMS Energy"].median())
mad_rms = float(
    np.median(np.abs(features["RMS Energy"] - median_rms))
)

# MAD is more robust than standard deviation for bursty signals.
if mad_rms > 1e-10:
    energy_limit = median_rms + (
        threshold_multiplier * 1.4826 * mad_rms
    )
else:
    energy_limit = median_rms * (1 + threshold_multiplier)

features["Unusual Segment"] = (
    features["RMS Energy"] > energy_limit
)

# =========================================================
# NAVIGATION
# =========================================================
st.sidebar.divider()
page = st.sidebar.radio(
    "WORKSPACE",
    [
        "Signal Overview",
        "Waveform & Spectrogram",
        "Feature Analysis",
        "Event Timeline",
        "Export Lab"
    ]
)

st.sidebar.divider()
st.sidebar.markdown(
    "<div class='mono'>SONAR · DAY 08 / 60<br>DIGITAL SIGNAL PROCESSING</div>",
    unsafe_allow_html=True
)

# =========================================================
# PAGE 1 — OVERVIEW
# =========================================================
if page == "Signal Overview":
    page_header(
        "SIGNAL OVERVIEW / 001",
        "Listen closer.",
        "A recording is more than sound. SONAR turns audio into "
        "measurable signal characteristics, helping you explore "
        "energy, frequency and unusual changes over time."
    )

    flagged = int(features["Unusual Segment"].sum())
    peak = float(np.max(np.abs(audio)))
    rms_global = float(np.sqrt(np.mean(audio ** 2)))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Duration", f"{duration:.2f}s", "Recording length")
    with c2:
        metric_card("Sample rate", f"{sample_rate:,}", "Samples per second · Hz")
    with c3:
        metric_card("Global RMS", f"{rms_global:.4f}", "Overall signal energy")
    with c4:
        metric_card("Flagged windows", str(flagged), "Unusually high energy")

    st.write("")
    st.markdown("### Playback")
    st.audio(make_audio_bytes(audio, sample_rate), format="audio/wav")

    st.markdown("<div class='section-heading'>Signal snapshot</div>",
                unsafe_allow_html=True)

    time_axis = np.arange(len(audio)) / sample_rate
    display_step = max(1, len(audio) // 12000)

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=time_axis[::display_step],
        y=audio[::display_step],
        mode="lines",
        line=dict(color=LIME, width=1),
        name="Audio amplitude"
    ))
    fig.add_hline(y=0, line_color=GRID)
    add_chart_theme(fig, 300)
    fig.update_layout(
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Amplitude is normalised to the signal's stored numeric range. "
        "It is not a calibrated sound-pressure measurement."
    )

# =========================================================
# PAGE 2 — WAVEFORM AND SPECTROGRAM
# =========================================================
elif page == "Waveform & Spectrogram":
    page_header(
        "SIGNAL VISUALISATION / 002",
        "See the sound.",
        "Explore how the waveform changes over time and how its "
        "frequency content evolves throughout the recording."
    )

    time_axis = np.arange(len(audio)) / sample_rate
    display_step = max(1, len(audio) // 15000)

    waveform = go.Figure()
    waveform.add_trace(go.Scattergl(
        x=time_axis[::display_step],
        y=audio[::display_step],
        mode="lines",
        line=dict(color=LIME, width=1),
        name="Waveform"
    ))
    add_chart_theme(waveform, 280)
    waveform.update_layout(
        title="TIME-DOMAIN WAVEFORM",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        showlegend=False
    )
    st.plotly_chart(waveform, use_container_width=True)

    frequencies, times, power = spectrogram(
        audio,
        fs=sample_rate,
        nperseg=min(1024, len(audio)),
        noverlap=min(768, max(0, min(1024, len(audio)) - 1))
    )

    power_db = 10 * np.log10(power + 1e-12)

    spec = go.Figure(data=go.Heatmap(
        x=times,
        y=frequencies,
        z=power_db,
        colorscale=[
            [0, "#171522"],
            [0.35, "#4A367A"],
            [0.7, "#A18BFF"],
            [1, "#C5F36A"]
        ],
        colorbar=dict(title="dB")
    ))
    add_chart_theme(spec, 430)
    spec.update_layout(
        title="FREQUENCY-DOMAIN SPECTROGRAM",
        xaxis_title="Time (seconds)",
        yaxis_title="Frequency (Hz)"
    )
    st.plotly_chart(spec, use_container_width=True)

    st.caption(
        "The spectrogram shows the distribution of signal power "
        "across time and frequency. Colours represent relative power."
    )

# =========================================================
# PAGE 3 — FEATURE ANALYSIS
# =========================================================
elif page == "Feature Analysis":
    page_header(
        "FEATURE EXTRACTION / 003",
        "Measure the signal.",
        "Compare short-time audio characteristics to understand "
        "how the recording behaves across different moments."
    )

    selected_feature = st.selectbox(
        "Choose a feature",
        [
            "RMS Energy",
            "Peak Amplitude",
            "Zero Crossing Rate",
            "Spectral Centroid (Hz)"
        ]
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=features["Start (s)"],
        y=features[selected_feature],
        mode="lines+markers",
        line=dict(color=VIOLET, width=2),
        marker=dict(size=4, color=LIME),
        name=selected_feature
    ))
    add_chart_theme(fig, 370)
    fig.update_layout(
        title=selected_feature.upper(),
        xaxis_title="Time (seconds)",
        yaxis_title=selected_feature
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Feature guide")
    descriptions = {
        "RMS Energy": "Root Mean Square energy estimates the average magnitude of a signal window.",
        "Peak Amplitude": "The largest absolute sample value inside a window.",
        "Zero Crossing Rate": "The proportion of adjacent samples that change sign.",
        "Spectral Centroid (Hz)": "The weighted centre of the frequency spectrum; often used as a measure of spectral brightness."
    }
    st.info(descriptions[selected_feature])

    summary = features[
        [
            "RMS Energy",
            "Peak Amplitude",
            "Zero Crossing Rate",
            "Spectral Centroid (Hz)"
        ]
    ].describe().T
    st.markdown("### Statistical summary")
    st.dataframe(summary.round(5), use_container_width=True)

# =========================================================
# PAGE 4 — EVENT TIMELINE
# =========================================================
elif page == "Event Timeline":
    page_header(
        "SEGMENT DETECTION / 004",
        "Find unusual moments.",
        "SONAR flags windows whose RMS energy is unusually high "
        "relative to the recording's typical energy level. "
        "This is a signal-based screening method, not a sound classifier."
    )

    flagged = features[features["Unusual Segment"]]

    a, b, c = st.columns(3)
    with a:
        metric_card("Flagged windows", str(len(flagged)), "Above selected threshold")
    with b:
        metric_card("Threshold", f"{energy_limit:.4f}", "RMS energy limit")
    with c:
        pct = 100 * len(flagged) / max(1, len(features))
        metric_card("Share flagged", f"{pct:.1f}%", "Of analysed windows")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=features["Start (s)"],
        y=features["RMS Energy"],
        mode="lines",
        line=dict(color=VIOLET, width=2),
        name="RMS energy"
    ))
    fig.add_hline(
        y=energy_limit,
        line_dash="dash",
        line_color=PINK,
        annotation_text="Detection threshold"
    )

    if not flagged.empty:
        fig.add_trace(go.Scatter(
            x=flagged["Start (s)"],
            y=flagged["RMS Energy"],
            mode="markers",
            marker=dict(color=LIME, size=9, symbol="diamond"),
            name="Flagged windows"
        ))

    add_chart_theme(fig, 380)
    fig.update_layout(
        title="ENERGY EVENT TIMELINE",
        xaxis_title="Time (seconds)",
        yaxis_title="RMS energy"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Flagged segments")
    if flagged.empty:
        st.success(
            "No windows crossed the current energy threshold. "
            "Try lowering the threshold in the sidebar."
        )
    else:
        st.dataframe(
            flagged[
                [
                    "Start (s)",
                    "End (s)",
                    "RMS Energy",
                    "Peak Amplitude",
                    "Spectral Centroid (Hz)"
                ]
            ].round(5),
            hide_index=True,
            use_container_width=True
        )

    st.caption(
        "High energy does not necessarily mean a defect or anomaly. "
        "A loud but normal sound can also be flagged."
    )

# =========================================================
# PAGE 5 — EXPORT LAB
# =========================================================
elif page == "Export Lab":
    page_header(
        "DATA OUTPUT / 005",
        "Take the analysis further.",
        "Export the extracted features and segment flags for "
        "additional analysis, reporting or model development."
    )

    st.markdown("### Feature table")
    st.dataframe(
        features.head(20).round(5),
        hide_index=True,
        use_container_width=True
    )

    st.markdown("### Export files")
    csv_bytes = features.to_csv(index=False).encode("utf-8")

    report = pd.DataFrame([{
        "File": audio_name,
        "Duration (seconds)": duration,
        "Sample Rate (Hz)": sample_rate,
        "Sample Count": len(audio),
        "Global RMS": float(np.sqrt(np.mean(audio ** 2))),
        "Peak Amplitude": float(np.max(np.abs(audio))),
        "Analysis Window (seconds)": window_seconds,
        "Window Step (seconds)": hop_seconds,
        "Energy Threshold": energy_limit,
        "Flagged Windows": int(features["Unusual Segment"].sum())
    }])
    report_bytes = report.to_csv(index=False).encode("utf-8")

    x, y = st.columns(2)
    with x:
        st.download_button(
            "Download feature data",
            data=csv_bytes,
            file_name="sonar_audio_features.csv",
            mime="text/csv",
            use_container_width=True
        )
    with y:
        st.download_button(
            "Download recording summary",
            data=report_bytes,
            file_name="sonar_recording_summary.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.info(
        "The exported event flags are based on RMS energy only. "
        "They are not verified labels of real-world sound events."
    )

# =========================================================
# FOOTER
# =========================================================
st.divider()
st.markdown(
    "<div class='mono'>SONAR AUDIO INTELLIGENCE LAB · "
    "DAY 08 OF 60 · BUILT WITH PYTHON</div>",
    unsafe_allow_html=True
)
