
# SONAR — Audio Intelligence Lab

Day 8 of the 60-Day End-of-Year Project Marathon.

SONAR is an interactive audio analysis application built with Python.
It transforms audio recordings into visualisations and measurable
signal features using Digital Signal Processing techniques.

## Features

- Synthetic audio demo.
- WAV, FLAC and OGG upload.
- Audio playback.
- Waveform visualisation.
- Spectrogram visualisation.
- RMS energy extraction.
- Peak amplitude calculation.
- Zero-crossing rate.
- Spectral centroid.
- Configurable window-based analysis.
- Unusual high-energy segment detection.
- CSV feature export.
- Recording summary export.

## Technology Stack

- Python
- Streamlit
- NumPy
- Pandas
- SciPy
- SoundFile
- Plotly

## Installation — Windows / PowerShell

Open the project folder in VS Code.

Create a virtual environment:

    py -3.10 -m venv .venv

Activate it:

    .\.venv\Scripts\Activate.ps1

If PowerShell blocks activation:

    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\.venv\Scripts\Activate.ps1

Install dependencies:

    python -m pip install --upgrade pip --timeout 600 --retries 10

    python -m pip install -r requirements.txt --no-cache-dir --timeout 600 --retries 10

Run the application:

    streamlit run app.py

## Getting Started

1. Open the application.
2. Select Synthetic demo in the sidebar.
3. Play the generated recording.
4. Explore the waveform and spectrogram.
5. Review the extracted audio features.
6. Explore unusually high-energy segments.
7. Export the results as CSV.

## Upload Format

Supported formats:

- WAV
- FLAC
- OGG

The current version supports recordings up to five minutes.

## Analytical Methods

- RMS Energy: measures signal energy over short windows.
- Peak Amplitude: measures the maximum absolute sample value.
- Zero Crossing Rate: measures sign changes between adjacent samples.
- Spectral Centroid: estimates the weighted centre of the frequency spectrum.
- Spectrogram: visualises signal power across time and frequency.
- Energy Screening: flags windows above a robust median-based threshold.

## Limitations

The energy detector identifies unusually high-energy windows.
It does not identify the semantic meaning of a sound or prove
that a recording contains a defect.

This is a signal-processing project, not a production-certified
audio classification or fault-detection system.
