# 🎤 LatentSync Enhanced – AI Lipsync with Optional Word-by-Word Subtitles, 4K, and Metadata Spoofing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-blue.svg)
![Python](https://img.shields.io/badge/python-3.10-blue)
![GPU Recommended](https://img.shields.io/badge/GPU-NVIDIA%20CUDA%20Recommended-green)

> **Completely rebuilt version of [LatentSync](https://github.com/Yingqing-Pei/LatentSync)** with a powerful, no-code Web UI. This enhanced fork adds dynamic subtitles, optional 4K upscaling, audio padding, metadata injection, and full local control – no Stable Diffusion or ComfyUI needed.  
> **This is THE FINAL VERSION! No more work from myself will be done to this project unless unexpected errors show up.**

---

## ✨ Key Highlights

- ✅ **Full Gradio Web UI** – no command-line knowledge required  
- ✅ **Advanced AI Lipsync** for any voice and face  
- ⚙️ **Optional Dynamic Word-by-Word Subtitles**  
- ⚙️ **Optional 4K Upscaling** (Portrait / Landscape / Square)  
- ⚙️ **Optional Audio Padding** (extend short audio to video length)  
- ✅ **Realistic Camera/GPS Metadata Injection**  
- ✅ **Automatic Cleanup** of temporary files  
- ✅ **100% Offline / Local-Only Processing**  
- ⚙️ **Customizable Parameters** (subtitle font, offset, sync quality)  

---

## 🎬 Before & After

| Original Video | AI-Synced & Subtitled |
|----------------|------------------------|
| ![](demo/before.gif) | ![](demo/after.gif) |

> 🎥 Full MP4s available in the [`demo/`](demo/) folder.

---

## ⚡ Quick Start (Cross-Platform)

```bash
# 1. Clone
git clone https://github.com/frisse11/LatentSync-with-word-for-word-subtitles-and-upscale-to-4k.git
cd LatentSync-with-word-for-word-subtitles-and-upscale-to-4k

# 2. Create Conda env
conda create -n latentsync python=3.10 -y
conda activate latentsync

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install FFmpeg + ExifTool (see OS section below)

# 5. Download LatentSync model checkpoints
# → https://github.com/Yingqing-Pei/LatentSync#installation

# 6. Launch the Web UI
python gradio_app.py
```

---

## 🖥️ OS-Specific Installation

### 🐧 Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install ffmpeg exiftool
```

### 🪟 Windows

- **Winget (Recommended):**
```powershell
winget install -e --id Gyan.FFmpeg
winget install -e --id PhilHarvey.ExifTool
```

- **Chocolatey (Alternative):**
```powershell
choco install ffmpeg
choco install exiftool
```

- **Manual Installation:**
  - [FFmpeg](https://ffmpeg.org/download.html) – add `bin/` to PATH
  - [ExifTool](https://exiftool.org/install.html#windows) – rename `exiftool(-k).exe` and place in a folder in PATH

### 🍎 macOS (with Homebrew)
```bash
brew install ffmpeg exiftool
```

---

## 🧠 Feature Overview

### 🔁 **Lipsync Engine (LatentSync, Rebuilt)**
- AI-based frame-level mouth synchronization
- Designed for **realism**, **emotion**, and **natural motion**
- Adjustable: *Guidance Scale*, *Inference Steps*, *Seed*

### ⚙️ **Optional Word-by-Word Subtitles**
- Dynamic word-by-word rendering
- Burned directly into the video (if enabled)
- Customizable font size, vertical offset
- Preview & edit transcript before final render

### ⚙️ **Optional 4K Video Upscaling**
- Smart aspect-ratio detection (portrait, landscape, square)
- High-quality upscale to:
  - `3840x2160` (landscape)
  - `2160x3840` (portrait)
  - `2160x2160` (square)

### ⚙️ **Optional Audio Padding**
- Automatically adds silence if audio is shorter than video
- Prevents abrupt cutoffs and sync issues

### ✅ **Realistic Metadata Injection**
- Fake EXIF (camera model, lens info)
- Fake GPS (geotagging for TikTok, Reels)
- Helps bypass “low-quality” platform filters

---

## 🧪 How to Use the Web UI

1. **Upload Media & Configure**
   - Upload your **video** and **audio**
   - Enable optional features:
     - ✅ Dynamic Subtitles
     - ✅ 4K Upscaling
     - ✅ Audio Padding
   - Set model parameters

2. **Processing**
   - Live terminal view of progress
   - If subtitles are enabled: edit transcript in table → confirm

3. **Finalization**
   - Preview processed video
   - File saved to: `./processed_videos/`

---

## 🛠️ Troubleshooting

| Problem | Solution |
|--------|----------|
| `CUDA out of memory` | Reduce video resolution or close other GPU apps |
| `ModuleNotFoundError` | Make sure your Conda env is active and run `pip install -r requirements.txt` |
| `ffmpeg/exiftool: not found` | Not installed or not in PATH – revisit [Step 4](#step-4-install-ffmpeg--exiftool) |
| Subtitle font error | Ensure `LuckiestGuy-Regular.ttf` is installed or substitute your own |
| Timestamp error: `‘<=’ not supported` | Make sure subtitle timestamps are numeric and clean |

---

## 📄 License

MIT © [Marc Fabry](https://github.com/frisse11)  
Forked from the original [LatentSync by Yingqing Pei](https://github.com/Yingqing-Pei/LatentSync)

---

## ⚠️ Disclaimer

This project is provided **as-is**, without any warranties, guarantees, or liability of any kind.  
You use this software **entirely at your own risk**.

The author explicitly **disclaims any responsibility or obligation** for:

- Software failures  
- Operating system or hardware conflicts  
- Data corruption or loss  
- Incompatibilities with future dependencies  
- Any kind of personal, commercial, or production damage  

Although care was taken during development, **no guarantees** are provided regarding functionality, security, or suitability.  
This is a **final release** – no further updates or support will be provided unless critical bugs are discovered.

