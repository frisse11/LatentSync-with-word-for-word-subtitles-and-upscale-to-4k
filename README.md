# LatentSync: Enhanced Web UI & Word-by-Word Subtitles

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is a forked and extended version of the original [LatentSync project](https://github.com/Yingqing-Pei/LatentSync). This version introduces a powerful Gradio Web UI for easy operation, along with automatic, word-by-word colored subtitle generation.

**Original Project:** [LatentSync](https://github.com/Yingqing-Pei/LatentSync)
**Author of this Fork:** Marc Fabry

---

### Before & After Demo

**Original Video:**
![Original Video Demo](demo/before.gif)

**Processed Video with Subtitles:**
![Processed Video Demo](demo/after.gif)

*(Note: These GIFs are for demonstration. The original MP4 files are in the `demo/` directory.)*

---

## Table of Contents

*   [Key Features](#key-features)
*   [Prerequisites](#prerequisites)
*   [Installation Guide](#installation-guide)
    *   [Step 1: Clone the Repository](#step-1-clone-the-repository)
    *   [Step 2: Create the Conda Environment](#step-2-create-the-conda-environment)
    *   [Step 3: Install Python Dependencies](#step-3-install-python-dependencies)
    *   [Step 4: Download the Checkpoints](#step-4-download-the-checkpoints)
    *   [Step 5: Install FFmpeg](#step-5-install-ffmpeg)
*   [Launch the Application](#launch-the-application)
*   [How to Use the Web UI](#how-to-use-the-web-ui)
*   [Troubleshooting](#troubleshooting)
*   [Future Improvements](#future-improvements)
*   [License](#license)

---

## Key Features

*   **Intuitive Gradio Web UI:** No command-line needed! A user-friendly web interface guides you through the process.
*   **Word-by-Word Colored Subtitles:** Automatically generate and burn colorful, animated subtitles directly onto your video.
*   **Full Transcription Control:** Edit the generated transcription (text and timestamps) in a simple table before creating the final video.
*   **Simplified Workflow:** The UI is organized into clear tabs: Upload, Transcribe, LatentSync, and Finalize.
*   **Customizable Output:** Easily adjust LatentSync parameters, font size, subtitle position, and even force a 4K vertical output.
*   **Automatic Cleanup:** Temporary files are automatically deleted after processing.

---

## Prerequisites

Before you begin, ensure you have the following software installed:

1.  **Git:** For cloning the repository.
2.  **Conda (Anaconda or Miniconda):** For managing the Python environment and dependencies. We recommend **Miniconda** for a lightweight installation.
    *   [Download Anaconda/Miniconda](https://docs.conda.io/en/latest/miniconda.html)
3.  **NVIDIA GPU with CUDA:** For the best performance, an NVIDIA GPU with CUDA support is required. Ensure you have the latest NVIDIA drivers installed.
4.  **FFmpeg:** A crucial tool for video and audio processing.

---

## Installation Guide

Follow these steps carefully to set up the application correctly.

### Step 1: Clone the Repository

Open a terminal (Linux/macOS) or Anaconda Prompt (Windows) and run the following commands:

```bash
git clone https://github.com/frisse11/LatentSync-with-word-for-word-subtitles-and-upscale-to-4k.git
cd LatentSync-with-word-for-word-subtitles-and-upscale-to-4k
```

### Step 2: Create the Conda Environment

We will use the `setup_env.sh` script to create a Conda environment named `latentsync` with all the necessary base packages.

```bash
# Create the environment from the file
conda env create -f setup_env.sh

# Activate the newly created environment
conda activate latentsync
```

> **Note:** If the command above fails, you can create the environment manually:
> ```bash
> conda create --name latentsync python=3.9 -y
> conda activate latentsync
> # You would then need to install packages from the .sh file manually.
> # It's better to make the script work if possible.
> ```

### Step 3: Install Python Dependencies

Once the environment is active, install the Python packages required for the web UI and other features using `pip`.

```bash
pip install -r requirements.txt
```

### Step 4: Download the Checkpoints

The core functionality of LatentSync requires pre-trained model files (checkpoints).

1.  Download the checkpoints as described in the **[original LatentSync README](https://github.com/Yingqing-Pei/LatentSync#installation)**.
2.  Place the downloaded files into the `checkpoints/` folder in the project directory.

**This is a crucial step! The application will not work without the checkpoints.**

### Step 5: Install FFmpeg

FFmpeg is required for processing video and audio files.

**For Linux (Debian/Ubuntu):**
```bash
sudo apt update && sudo apt install ffmpeg
```

**For Windows:**
The easiest way is to use **winget** or **Chocolatey**.

*   **With Winget (built into Windows 10/11):**
    Open PowerShell and run:
    ```powershell
    winget install -e --id Gyan.FFmpeg
    ```
*   **With Chocolatey:**
    ```powershell
    choco install ffmpeg
    ```
*   **Manually:**
    1.  Download the FFmpeg binaries from [ffmpeg.org](https://ffmpeg.org/download.html).
    2.  Extract the `.zip` file to a location like `C:fmpeg`.
    3.  Add the `bin` folder (e.g., `C:fmpegin`) to your Windows `Path` environment variable.

---

## Launch the Application

After the installation is complete, you can launch the web UI.

1.  Make sure your `latentsync` Conda environment is active:
    ```bash
    conda activate latentsync
    ```
2.  Start the Gradio app:
    ```bash
    python gradio_app.py
    ```
3.  Open your web browser and navigate to the local URL shown in the terminal (usually `http://127.0.0.1:7860`).

---

## How to Use the Web UI

1.  **Upload Media:** Go to the first tab and upload your video and audio files.
2.  **Transcribe:** Click "Transcribe Audio". Once finished, you can edit the words and timestamps in the table.
    > **Whisper Model Note:** The first time you transcribe, the application will automatically download the `openai/whisper-small` model (a few hundred MB). This requires an internet connection and may take a few minutes. Subsequent transcriptions will be much faster as the model will be cached on your system.
3.  **Run LatentSync:** Adjust the parameters and click "Run LatentSync".
4.  **Finalize Video:** Customize the subtitle appearance and click "Apply Subtitles & Finalize Video". Your final video will appear here and be saved to the `processed_videos` folder.

> **Font Note:** The subtitle feature uses the `LuckiestGuy-Regular.ttf` font. If you don't have it, please install it or update the `FONT_PATH` variable in `gradio_app.py` to a font file on your system.

---

## Troubleshooting

*   **`CUDA out of memory`:** Your GPU does not have enough memory. Close other programs that are using the GPU.
*   **`ModuleNotFoundError`:** A Python package is missing. Activate the correct environment (`conda activate latentsync`) and run `pip install -r requirements.txt` again.
*   **`ffmpeg: command not found`:** FFmpeg is not installed correctly or not added to your system's PATH. Re-follow [Step 5](#step-5-install-ffmpeg).
*   **Font not found:** Make sure the font file exists at the path specified in `gradio_app.py`, or change the path to a font you have.

---

## Future Improvements

*   [ ] **Automatic 4K Orientation:** Automatically detect video orientation (portrait/landscape) for smart 4K scaling.
*   [ ] **More subtitle animations:** Add more options for subtitle animations.

---

## License

This project is licensed under the [MIT License](LICENSE).