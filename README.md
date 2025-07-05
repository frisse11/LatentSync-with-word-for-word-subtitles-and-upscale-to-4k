# LatentSync with Enhanced Web UI and Word-by-Word Subtitles (by Marc Fabry)

This is a forked and extended version of the original [LatentSync project](https://github.com/Yingqing-Pei/LatentSync) with significant improvements focused on ease of use and additional functionality via an intuitive Gradio Web User Interface.

**Original Project:** [LatentSync](https://github.com/Yingqing-Pei/LatentSync)
**Author of this Fork:** Marc Fabry

## New Features

This fork introduces the following key functionalities:

1.  **Gradio Web User Interface:**
    *   A user-friendly, web-based application (`gradio_app.py`, located in the root directory of the project) for executing LatentSync operations without requiring command-line interaction.
    *   The workflow is structured into clear tabs:
        *   **Upload Media:** Easily upload video and audio files.
        *   **Transcribe:** Perform Automatic Speech Recognition (ASR) on the uploaded audio (via OpenAI Whisper model from Hugging Face Transformers). The transcription is displayed in an editable table, allowing users to correct errors or adjust word timings.
        *   **Run LatentSync:** Adjust parameters such as `Guidance Scale` and `Inference Steps` and then initiate the LatentSync generation process.
        *   **Finalize Video:** Configure subtitle settings (font size, vertical offset) and an optional 4K vertical resolution conversion. The final processed video is displayed here.
2.  **Word-by-Word Colored Subtitles:**
    *   The application automatically generates colorful, word-by-word animated subtitles that are burned directly onto the video (hardcoded subtitles).
    *   The transcription table allows for precise editing of subtitle text and timing, ensuring accuracy before video finalization.
    *   **Note on Font:** This feature utilizes a specific font. Ensure the `LuckiestGuy-Regular.ttf` font is available on your system (e.g., in `/usr/share/fonts/truetype/luckiestguy/`), or update the `FONT_PATH` variable in `gradio_app.py` to point to a suitable alternative font file.
3.  **Improved User Experience:**
    *   Clear status updates are provided during each step of the process.
    *   Automatic navigation between tabs guides the user through the workflow.
    *   Intelligent activation/deactivation of buttons ensures only relevant actions are available.
    *   Automated cleanup of temporary files after the process completes.
    *   Configurable video display sizes in the UI for better presentation and fit.

## Demo: Before & After

Here are visual demonstrations of the original video compared to the processed output with colored subtitles:

### Original Video (Before)
<video src="demo/before.mp4" controls muted loop autoplay width="600"></video>

### Processed Video with Subtitles (After)
<video src="demo/after.mp4" controls muted loop autoplay width="600"></video>

*(Note: These are raw .mp4 video files embedded directly. Playback performance may vary depending on your browser and network connection. You can find these files in the `demo/` directory of this repository.)*

## Installation and Usage

Follow these steps to set up and run the Gradio Web UI:

1.  **Clone This Fork:**
    ```bash
    git clone https://github.com/frisse11/LatentSync-with-word-for-word-subtitles-and-upscale-to-4k.git
    cd LatentSync-with-word-for-word-subtitles-and-upscale-to-4k
    ```
2.  **Create or Activate the Conda Environment:**
    This project is built upon the Conda environment of the original LatentSync project, with additional dependencies.
    ```bash
    # If you haven't set up the original LatentSync environment yet:
    # Refer to the original LatentSync README for base environment setup.
    # Typically: conda env create -f setup_env.sh
    
    # Activate your LatentSync Conda environment:
    conda activate latentsync
    ```
3.  **Install Required Python Dependencies:**
    The new dependencies for the Gradio UI have been added to `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: It is recommended to create a dedicated Conda or virtual environment for this project to manage dependencies effectively and avoid conflicts with other Python projects.*

4.  **Download Necessary LatentSync Checkpoints:**
    Ensure you download and place the required checkpoints for LatentSync into the `checkpoints/` folder, as described in the [original LatentSync README](https://github.com/Yingqing-Pei/LatentSync#installation). This step is crucial for the application's core functionality.

5.  **Start the Gradio Web UI:**
    Once all dependencies are installed and checkpoints are in place, you can launch the web interface:
    ```bash
    python gradio_app.py
    ```
    The application will then be accessible locally in your web browser, typically at `http://127.0.0.1:7860`.

## Todo / Future Improvements

*   **Automatic 4K Orientation Detection:** Implement automatic detection of source video orientation (portrait vs. landscape) to intelligently apply either 2160x3840 (vertical) or 3840x2160 (horizontal) scaling for 4K output, enhancing user convenience.
*   **Demo Videos:** Add "Before & After" demonstration videos/GIFs directly to the README to visually showcase the project's capabilities. *(Note: The raw .mp4 files are in `demo/`, but consider converting them to GIFs or compressed MP4s for better web viewing.)*

## Contributing

Contributions to this project are welcome! If you encounter bugs, have suggestions for improvements, or wish to add new features, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). Please refer to the original LatentSync repository for its specific licensing details.
