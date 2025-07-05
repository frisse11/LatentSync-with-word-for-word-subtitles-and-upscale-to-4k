import gradio as gr
from pathlib import Path
from scripts.inference import main  # Ensure this import path is correct for your LatentSync project
from omegaconf import OmegaConf
import argparse
from datetime import datetime
import subprocess
import os
import shutil
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import pipeline
import librosa
import random
import pandas as pd

# --- General Configuration ---
CONFIG_PATH = Path("configs/unet/stage2.yaml")
CHECKPOINT_PATH = Path("checkpoints/latentsync_unet.pt")
FOLDER_FOR_PROCESSED_VIDEOS = Path("./processed_videos")
FOLDER_FOR_PROCESSED_VIDEOS.mkdir(parents=True, exist_ok=True)
TEMP_DIR = Path("./temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# --- Subtitling Configuration ---
FONT_PATH = Path("/usr/share/fonts/truetype/luckiestguy/LuckiestGuy-Regular.ttf")
# Check if font exists and set fallback
if not FONT_PATH.exists():
    print(f"WARNING: Font not found at {FONT_PATH}. Subtitles might use a default font or fail.")
    if Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf").exists():
        FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        print(f"INFO: Falling back to DejaVuSans-Bold.ttf at {FONT_PATH}")
    elif os.name == 'nt' and Path("C:/Windows/Fonts/arial.ttf").exists():
        FONT_PATH = Path("C:/Windows/Fonts/arial.ttf")
        print(f"INFO: Falling back to Arial.ttf at {FONT_PATH}")
    else:
        FONT_PATH = None # Indicates PIL should use its default
        print("WARNING: No suitable specific font found, using PIL default font.")

# --- Utility Functions ---
def get_media_duration(file_path: Path) -> float:
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(file_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        raise gr.Error(f"Error getting duration for {file_path} with ffprobe: {e.stderr.decode(errors='ignore')}")
    except ValueError:
        raise gr.Error(f"Could not parse duration for {file_path}.")

def pad_audio_with_silence(original_audio_path: Path, target_duration: float, output_dir: Path) -> Path:
    audio_duration = get_media_duration(original_audio_path)
    if audio_duration >= target_duration:
        return original_audio_path

    silence_to_add = target_duration - audio_duration
    padded_audio_filename = f"{original_audio_path.stem}_padded_{datetime.now().strftime('%H%M%S')}.m4a"
    padded_audio_path = output_dir / padded_audio_filename

    cmd = [
        "ffmpeg",
        "-i", str(original_audio_path),
        "-af", f"apad=pad_dur={silence_to_add}",
        "-map", "0:a:0",
        "-c:a", "aac",
        "-b:a", "128k",
        "-y",
        str(padded_audio_path),
    ]

    try:
        subprocess.run(cmd, check=True)
        return padded_audio_path
    except subprocess.CalledProcessError as e:
        raise gr.Error(f"Error padding audio: {e.stderr.decode(errors='ignore')}")

# --- Subtitling Functions ---
def create_colored_text_image(text, font_path, font_size=120, outline_width=3):
    try:
        if font_path: # Check if font_path is not None (due to fallback logic)
            font = ImageFont.truetype(str(font_path), font_size)
        else:
            font = ImageFont.load_default() # Use default if no specific font is found
    except IOError:
        font = ImageFont.load_default() # Fallback if error loading font

    bbox = font.getbbox(text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    extra_padding = outline_width * 3
    img_width = text_width + 2 * extra_padding
    img_height = text_height + 2 * extra_padding

    img = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    text_offset_x = extra_padding - bbox[0]
    text_offset_y = extra_padding - bbox[1]

    r, g, b = random.randint(150, 255), random.randint(150, 255), random.randint(150, 255)
    text_color = (r, g, b, 255)

    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if abs(dx) + abs(dy) <= outline_width + 1:
                draw.text((text_offset_x + dx, text_offset_y + dy), text, font=font, fill=(0, 0, 0, 255))

    draw.text((text_offset_x, text_offset_y), text, font=font, fill=text_color)
    return img

def process_video_with_colored_words_from_data(
    input_video_path: Path,
    words_with_timestamps: list,
    font_size: int,
    vertical_offset_percent: int,
) -> Path:
    video_basename = input_video_path.stem
    temp_subtitled_video_path = TEMP_DIR / f"{video_basename}_subtitled_{datetime.now().strftime('%H%M%S')}.mp4"
    temp_images_dir = TEMP_DIR / f"temp_text_images_{video_basename}_{datetime.now().strftime('%H%M%S')}"
    temp_images_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize ffmpeg_cmd to an empty list to ensure it's always defined
        ffmpeg_cmd = [] 

        if not words_with_timestamps:
            shutil.copy(str(input_video_path), str(temp_subtitled_video_path))
            return temp_subtitled_video_path

        ffmpeg_filters = []
        image_inputs = []
        current_video_stream = "base_video"
        ffmpeg_filters.append(f"[0:v]format=yuv420p[base_video]")

        x_pos = "(W - w)/2"
        y_pos = f"H - h - H*{vertical_offset_percent/100:.3f}"

        for i, word_data in enumerate(words_with_timestamps):
            word_text = word_data['word']
            start_sec = word_data['start']
            end_sec = word_data['end']

            if not word_text or end_sec <= start_sec:
                continue

            img = create_colored_text_image(word_text, FONT_PATH, font_size=font_size, outline_width=3)
            img_path = temp_images_dir / f"text_{i:04d}.png"
            img.save(img_path)

            image_inputs.extend(["-i", str(img_path)])
            filter_str = f"[{current_video_stream}][{i+1}:v]overlay={x_pos}:{y_pos}:enable='between(t,{start_sec},{end_sec})'[v{i+1}]"
            ffmpeg_filters.append(filter_str)
            current_video_stream = f"v{i+1}"

        ffmpeg_inputs = ["-i", str(input_video_path)] + image_inputs
        full_filter_complex = ";".join(ffmpeg_filters)
        video_output_stream_to_map = f"[{current_video_stream}]" if ffmpeg_filters else "[base_video]"

        ffmpeg_cmd = [
            "ffmpeg",
            *ffmpeg_inputs,
            "-filter_complex", full_filter_complex,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-map", video_output_stream_to_map,
            "-map", "0:a?",
            "-y",
            str(temp_subtitled_video_path)
        ]

        subprocess.run(ffmpeg_cmd, check=True)
        return temp_subtitled_video_path
    except subprocess.CalledProcessError as e:
        raise gr.Error(f"Fout bij video overlay: {e.stderr.decode(errors='ignore')}")
    except Exception as e:
        # Pass ffmpeg_cmd to error message for better debugging
        # Ensure ffmpeg_cmd is defined even if an earlier error occurred
        final_cmd_str = ' '.join(ffmpeg_cmd) if ffmpeg_cmd else "Command not fully assembled"
        raise gr.Error(f"Subtitling error during image overlay process: {str(e)}. Attempted FFmpeg command: {final_cmd_str}")
    finally:
        # Corrected: Use ignore_errors=True for shutil.rmtree for robustness
        shutil.rmtree(temp_images_dir, ignore_errors=True) # Use ignore_errors parameter

# --- Gradio Workflow Functions (modified to use the refined helper functions) ---

def handle_upload_change(video_path_str, audio_path_str):
    """Updates states, status and enables/disables transcribe button."""
    video_present = bool(video_path_str)
    audio_present = bool(audio_path_str)

    if video_present and audio_present:
        status_msg = "Video and Audio uploaded. Ready to transcribe."
        transcribe_active = True
    elif video_present:
        status_msg = "Video uploaded. Waiting for audio..."
        transcribe_active = False
    elif audio_present:
        status_msg = "Audio uploaded. Waiting for video..."
        transcribe_active = False
    else:
        status_msg = "Please upload both video and audio files."
        transcribe_active = False

    # Return values for video_state, audio_state, status_upload, transcribe_btn
    return (
        video_path_str, # video_state
        audio_path_str, # audio_state
        gr.update(value=status_msg, visible=True), # status_upload
        gr.update(interactive=transcribe_active) # transcribe_btn
    )

def transcribe_audio_get_data_for_df(input_audio_path):
    """
    Transcribes audio using Whisper pipeline and returns data for DataFrame.
    Returns: list of dicts [{'word': '...', 'start': float, 'end': float}]
    """
    words_with_timestamps = []
    
    try:
        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            return_timestamps="word", # This returns 'chunks' with word segments
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device=device,
            chunk_length_s=30,
            stride_length_s=(5, 5)
        )
        audio_input, sr = librosa.load(str(input_audio_path), sr=16000, mono=True)
        result = pipe(audio_input)

        if 'chunks' not in result or not result['chunks']:
            print("Warning: No 'chunks' found in transcription result.")
            return []

        for segment in result['chunks']: 
            word_text = segment.get('text', '').strip()
            timestamp_tuple = segment.get('timestamp') # This will be (start, end) or None
            
            if timestamp_tuple is not None and len(timestamp_tuple) == 2:
                start_sec_raw, end_sec_raw = timestamp_tuple
                
                # Corrected: Explicitly convert to float and handle potential None in tuple elements
                try:
                    # Check if elements are actually convertible, if not, set to None
                    start_sec = float(start_sec_raw) if (start_sec_raw is not None and isinstance(start_sec_raw, (int, float))) else None
                    end_sec = float(end_sec_raw) if (end_sec_raw is not None and isinstance(end_sec_raw, (int, float))) else None
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert timestamp elements to float for segment: {segment}. Skipping.")
                    continue # Skip this segment if conversion fails

                # Corrected: Handle the last word's missing end_sec more gracefully.
                # If end_sec is None, set it to be slightly after start_sec.
                # This ensures the last word is always included.
                if start_sec is not None and word_text: # Ensure start_sec is valid and text exists
                    final_end_sec = end_sec if (end_sec is not None and end_sec > start_sec) else (start_sec + 0.1 if start_sec is not None else 0.1) # Assign small duration if end_sec is invalid/None
                    words_with_timestamps.append({
                        'word': word_text,
                        'start': start_sec,
                        'end': final_end_sec
                    })
                elif start_sec is None:
                    print(f"Warning: Start timestamp is None for segment: {segment}. Skipping.")
                elif not word_text:
                    print(f"Warning: Empty word text for segment: {segment}. Skipping.")
            else:
                print(f"Warning: Missing or malformed timestamp (not a 2-element tuple) for segment: {segment}. Skipping.")
        
        return words_with_timestamps
    except Exception as e:
        raise gr.Error(f"Fout bij transcriptie: {str(e)}")

def transcribe_audio_only(
    current_video_path_state: str,
    audio_path_str: str,
    # Outputs to keep track of, passed as inputs to the function as Gradio does
    current_latentsync_state: str,
    current_padded_audio_state: str
):
    """Transcribes audio, updates DataFrame, and enables LatentSync button."""
    # This check acts as a final safeguard; interactivity should prevent this call without inputs.
    if not current_video_path_state or not audio_path_str:
        # ALL 14 OUTPUTS MUST BE YIELDED here
        yield (
            pd.DataFrame(columns=["Word", "Start (s)", "End (s)"]), # 1: transcription_df
            None, # 2: transcription_state
            "Error: Both video and audio must be uploaded.", # 3: status_transcribe
            gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), # 4-6: btn updates
            None, None, None, None, None, # 7-11: state resets (video_state, audio_state, transcription_state, latentsync_state, padded_audio_state)
            gr.update(value=None), gr.update(value=None), # 12-13: video_input, audio_input visual resets
            gr.update(selected=0) # 14: tab navigation (gr.update(selected=index) is correct here)
        )
        raise gr.Error("Both video and audio must be uploaded before transcription.")

    audio_file_path = Path(audio_path_str)
    
    # Initial yield for status update and button/tab control
    # Yield all 14 outputs, even if their values don't change in this specific yield
    yield (
        pd.DataFrame(columns=["Word", "Start (s)", "End (s)"]), # 1: transcription_df (empty)
        audio_path_str, # 2: transcription_state (temp audio path)
        f"Transcribing audio from {audio_file_path.name}...", # 3: status_transcribe
        gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), # 4-6: btn updates (disable all)
        current_video_path_state, audio_path_str, None, current_latentsync_state, current_padded_audio_state, # 7-11: states (keep relevant, clear transcription_state temporarily)
        gr.update(value=current_video_path_state), gr.update(value=audio_path_str), # 12-13: video_input, audio_input visual (keep)
        gr.update(selected=1) # 14: stay on transcribe tab
    )

    try:
        words_data = transcribe_audio_get_data_for_df(audio_file_path)
        transcription_for_df = pd.DataFrame(words_data)
        transcription_for_df.rename(columns={"word": "Word", "start": "Start (s)", "end": "End (s)"}, inplace=True)
        
        # Final yield for successful transcription
        yield (
            transcription_for_df, # 1: transcription_df
            audio_path_str, # 2: transcription_state (set to transcribed audio path)
            f"Transcription complete for {audio_file_path.name}. You can now edit the table.", # 3: status_transcribe
            gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False), # 4-6: btn updates (enable LatentSync)
            current_video_path_state, audio_path_str, audio_path_str, current_latentsync_state, current_padded_audio_state, # 7-11: states (keep video, audio, transcription states, other states pass through)
            gr.update(value=current_video_path_state), gr.update(value=audio_path_str), # 12-13: video_input, audio_input visual (keep)
            gr.update(selected=2) # 14: navigate to LatentSync tab
        )

    except Exception as e:
        error_msg = f"Transcription error: {str(e)}"
        # All outputs must be yielded in case of error too
        yield (
            pd.DataFrame(columns=["Word", "Start (s)", "End (s)"]), # 1: transcription_df (empty)
            None, # 2: transcription_state (clear on error)
            error_msg, # 3: status_transcribe
            gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False), # 4-6: btn updates (enable transcribe for retry)
            current_video_path_state, None, None, current_latentsync_state, current_padded_audio_state, # 7-11: states (clear audio_state, transcription_state)
            gr.update(value=current_video_path_state), gr.update(value=None), # 12-13: video_input visual (keep), audio_input visual (clear)
            gr.update(selected=1) # 14: stay on transcribe tab
        )
        raise gr.Error(error_msg) # Re-raise for Gradio to show error pop-up

def run_latentsync_process(
    video_path_str: str, # Current uploaded video path (from video_state)
    transcribed_audio_path_str: str, # Audio path from transcription step (from transcription_state)
    guidance_scale: float,
    inference_steps: int,
    seed: int,
    enable_audio_padding: bool,
    # Outputs to keep track of, passed as inputs to the function as Gradio does
    current_video_state: str,
    current_audio_state: str,
    current_transcription_state: str,
    current_latentsync_state: str,
    current_padded_audio_state: str
):
    """Runs LatentSync, updates video output, and enables Finalize button."""
    # Safeguard check
    if not video_path_str or not transcribed_audio_path_str:
        yield (
            None, # 1: latentsync_state
            None, # 2: padded_audio_state
            "Error: Missing video or audio. Please ensure media is uploaded and transcribed.", # 3: status_latentsync
            gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False), # 4-6: btn updates
            current_video_state, current_audio_state, current_transcription_state, current_latentsync_state, current_padded_audio_state, # 7-11: states
            gr.update(value=current_video_state), gr.update(value=current_audio_state), # 12-13: video_input, audio_input visual
            gr.update(selected=1) # 14: Back to transcribe tab
        )
        raise gr.Error("Missing video or audio. Please ensure media is uploaded and transcribed.")

    video_file_path = Path(video_path_str)
    audio_for_latentsync_path = Path(transcribed_audio_path_str)
    temp_padded_audio_path = None
    
    # Initial yield for status update and button/tab control
    yield (
        None, # 1: latentsync_state (clear)
        None, # 2: padded_audio_state (clear)
        f"Running LatentSync on {video_file_path.name} with audio {audio_for_latentsync_path.name}...", # 3: status_latentsync
        gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), # 4-6: btn updates (disable all)
        current_video_state, current_audio_state, current_transcription_state, None, None, # 7-11: states (clear latentsync, padded_audio state)
        gr.update(value=current_video_state), gr.update(value=current_audio_state), # 12-13: video_input, audio_input visual (keep)
        gr.update(selected=2) # 14: stay on latentsync tab
    )

    try:
        if enable_audio_padding:
            video_duration = get_media_duration(video_file_path)
            audio_duration = get_media_duration(audio_for_latentsync_path)
            if audio_duration < video_duration:
                # Yield for padding status
                yield (
                    None, None, # latentsync_state, padded_audio_state
                    f"Padding audio by {video_duration - audio_duration:.2f} seconds...", # status_latentsync
                    gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), # btn updates (disable all)
                    current_video_state, current_audio_state, current_transcription_state, None, None, # states (keep current, clear latentsync, padded_audio state)
                    gr.update(value=current_video_path_state), gr.update(value=current_audio_path_state), # 12-13: video_input, audio_input visual (keep)
                    gr.update(selected=2) # 14: stay on latentsync tab
                )
                temp_padded_audio_path = pad_audio_with_silence(audio_for_latentsync_path, video_duration, TEMP_DIR)
                audio_input_for_main = temp_padded_audio_path
            else:
                audio_input_for_main = audio_for_latentsync_path
        else:
            audio_input_for_main = audio_for_latentsync_path

        latentsync_output_filename = f"{video_file_path.stem}_latentsync_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        latentsync_output_path = TEMP_DIR / latentsync_output_filename

        config = OmegaConf.load(CONFIG_PATH)
        config["run"].update({
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps,
        })

        # Corrected: Explicitly cast inference_steps and seed to int before passing to create_args
        args = create_args(
            video_file_path,
            audio_input_for_main,
            latentsync_output_path,
            int(inference_steps), # Ensure this is an integer
            guidance_scale,
            int(seed)             # Ensure this is an integer
        )

        # Added try-except SystemExit block around main() call to prevent server crash from argparse errors
        try:
            main(config=config, args=args) # Call the LatentSync main function
        except SystemExit as se:
            if se.code == 2: # argparse.ArgumentParser.error will call sys.exit(2)
                error_msg = f"LatentSync setup error: Invalid arguments provided. Please check console for details. (e.g., Inference Steps/Seed must be whole numbers)"
            else:
                error_msg = f"LatentSync process exited unexpectedly (code {se.code})."
            # Yield error state for all 14 outputs
            yield (
                None, None, # latentsync_state, padded_audio_state
                error_msg, # status_latentsync
                gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False), # btn updates (enable LatentSync for retry)
                current_video_state, current_audio_state, current_transcription_state, None, None, # states (clear latentsync_state, padded_audio_state)
                gr.update(value=current_video_state), gr.update(value=current_audio_state), # video_input, audio_input visual (keep)
                gr.update(selected=2) # Stay on LatentSync tab
            )
            raise gr.Error(error_msg) # Re-raise for Gradio to show error pop-up
        
        # Final yield for successful LatentSync
        yield (
            str(latentsync_output_path), # 1: latentsync_state
            str(temp_padded_audio_path) if temp_padded_audio_path else None, # 2: padded_audio_state
            f"LatentSync complete. Video saved to {latentsync_output_path.name}. Proceed to Finalize tab.", # 3: status_latentsync
            gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True), # 4-6: btn updates (enable finalize)
            current_video_state, current_audio_state, current_transcription_state, str(latentsync_output_path), str(temp_padded_audio_path) if temp_padded_audio_path else None, # 7-11: states
            gr.update(value=current_video_state), gr.update(value=current_audio_state), # 12-13: video_input, audio_input visual (keep)
            gr.update(selected=3) # 14: navigate to Finalize tab
        )

    except Exception as e:
        error_msg = f"LatentSync error: {str(e)}"
        # Yield all outputs in case of error
        yield (
            None, None, # latentsync_state, padded_audio_state
            error_msg, # status_latentsync
            gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False), # btn updates (enable LatentSync for retry)
            current_video_state, current_audio_state, current_transcription_state, None, None, # states (clear latentsync_state, padded_audio_state)
            gr.update(value=current_video_state), gr.update(value=current_audio_state), # 12-13: video_input, audio_input visual (keep)
            gr.update(selected=2) # 14: Stay on LatentSync tab
        )
        raise gr.Error(error_msg) # Re-raise for Gradio to show error pop-up

def apply_subtitles_and_finalize_video(
    latentsync_video_path_str: str,
    edited_transcription_dataframe: pd.DataFrame,
    font_size: int,
    vertical_offset_percent: int,
    force_4k_vertical: bool,
    # Current state values passed for cleanup/resetting
    current_video_path_state: str,
    current_audio_path_state: str,
    current_transcription_audio_path_state: str,
    current_latentsync_video_path_state: str,
    current_padded_audio_path_state: str
):
    """Applies subtitles, finalizes video, and resets UI for a new run."""
    # Safeguard check
    if not latentsync_video_path_str:
        yield (
            None, # 1: output_video
            "Error: No LatentSync video provided. Please run LatentSync first.", # 2: status_finalize
            gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True), # 3-5: btn updates
            pd.DataFrame(columns=["Word", "Start (s)", "End (s)"]), # 6: transcription_df (clear for new run)
            None, None, None, None, None, # 7-11: state resets
            gr.update(value=None), gr.update(value=None), # 12-13: video_input, audio_input visual resets
            gr.update(selected=0) # 14: go to upload tab for new run
        )
        raise gr.Error("No LatentSync video provided. Please run LatentSync first.")

    # Initial yield for status update and button/tab control
    yield (
        None, # 1: output_video (clear)
        "Applying subtitles and finalizing video...", # 2: status_finalize
        gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), # 3-5: btn updates (disable all)
        edited_transcription_dataframe, # 6: transcription_df (keep current during processing)
        current_video_path_state, current_audio_path_state, current_transcription_audio_path_state, current_latentsync_video_path_state, current_padded_audio_path_state, # 7-11: states (keep all current)
        gr.update(value=current_video_path_state), gr.update(value=current_audio_path_state), # 12-13: video_input, audio_input visual (keep)
        gr.update(selected=3) # 14: stay on finalize tab
    )

    final_output_video_path = None
    temp_subtitled_video_path_created = None # Track this for cleanup
    try:
        edited_words_with_timestamps = []
        if not edited_transcription_dataframe.empty and len(edited_transcription_dataframe.columns) == 3:
            for _, row in edited_transcription_dataframe.iterrows():
                try:
                    word = str(row["Word"]).strip()
                    start_time = float(row["Start (s)"])
                    end_time = float(row["End (s)"])
                    if word and end_time > start_time:
                        edited_words_with_timestamps.append({
                            'word': word,
                            'start': start_time,
                            'end': end_time
                        })
                except Exception as e:
                    # Corrected: Use a standard print for debug/warning.
                    print(f"Warning: Skipping malformed row in transcription DataFrame: {row.to_dict()} - Error: {e}")
                    continue
        else:
            # If no transcription data, log warning and proceed without subtitles
            print("Warning: No valid transcription data found, skipping subtitling.")
            # If we need to pass a video without subtitles, we copy the latent sync output to a temp file
            temp_subtitled_video_path_created = TEMP_DIR / f"{Path(latentsync_video_path_str).stem}_no_subtitles.mp4"
            shutil.copy(str(latentsync_video_path_str), str(temp_subtitled_video_path_created))


        if not edited_words_with_timestamps and temp_subtitled_video_path_created:
            # Case where no subtitles were created (e.g., empty transcription), use the copied latent sync video
            temp_subtitled_video_path_used = temp_subtitled_video_path_created
        else:
            # Case where subtitles are created
            temp_subtitled_video_path_used = process_video_with_colored_words_from_data(
                Path(latentsync_video_path_str),
                edited_words_with_timestamps,
                font_size,
                vertical_offset_percent,
            )
            # Ensure temp_subtitled_video_path_created points to the actual subtitled file for cleanup
            if not temp_subtitled_video_path_created: # Only if it wasn't already set by skipping subtitles
                temp_subtitled_video_path_created = temp_subtitled_video_path_used

        status_msg_detail = ""
        if force_4k_vertical:
            final_output_filename = f"{Path(latentsync_video_path_str).stem}_4K_vertical_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            final_output_video_path = FOLDER_FOR_PROCESSED_VIDEOS / final_output_filename

            ffmpeg_4k_cmd = [
                "ffmpeg",
                "-i", str(temp_subtitled_video_path_used),
                "-vf", "scale=2160:3840:force_original_aspect_ratio=increase,crop=2160:3840",
                "-c:v", "libx264", # Changed back to libx264 as it's very common and efficient
                "-preset", "medium",
                "-crf", "23",
                "-c:a", "copy",
                "-y",
                str(final_output_video_path)
            ]
            subprocess.run(ffmpeg_4k_cmd, check=True)
            status_msg_detail = f"Video resized to 4K vertical and saved to {final_output_video_path.name}"
        else:
            final_output_video_path = FOLDER_FOR_PROCESSED_VIDEOS / f"{Path(latentsync_video_path_str).stem}_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            shutil.move(str(temp_subtitled_video_path_used), str(final_output_video_path))
            status_msg_detail = f"Final video saved to {final_output_video_path.name}"

        # --- File Cleanup ---
        if Path(latentsync_video_path_str).exists():
            os.remove(latentsync_video_path_str) # LatentSync output
        if temp_subtitled_video_path_created and Path(temp_subtitled_video_path_created).exists() and Path(temp_subtitled_video_path_created) != final_output_video_path:
            # temp_subtitled_video_path_created might be moved or not created, so check if it still exists and wasn't the final output
            os.remove(temp_subtitled_video_path_created)
        if current_padded_audio_path_state and Path(current_padded_audio_path_state).exists():
            os.remove(current_padded_audio_path_state) # Padded audio

        # Final yield for successful finalization and UI reset
        yield (
            str(final_output_video_path), # 1: output_video
            f"Processing complete! {status_msg_detail}", # 2: status_finalize
            gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False), # 3-5: btn resets
            pd.DataFrame(columns=["Word", "Start (s)", "End (s)"]), # 6: transcription_df reset
            None, None, None, None, None, # 7-11: state resets (video_state, audio_state, transcription_state, latentsync_state, padded_audio_state)
            gr.update(value=None), gr.update(value=None), # 12-13: video_input, audio_input visual resets
            gr.update(selected=0) # 14: Go back to Upload Media tab for a new run
        )

    except Exception as e:
        error_msg = f"Finalization error: {str(e)}"
        # Yield all outputs in case of error
        yield (
            None, # output_video
            error_msg, # status_finalize
            gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True), # btn updates (finalize active)
            edited_transcription_dataframe, # transcription_df (keep for debugging/editing)
            current_video_path_state, current_audio_path_state, current_transcription_audio_path_state, current_latentsync_video_path_state, current_padded_audio_path_state, # states (keep all current)
            gr.update(value=current_video_path_state), gr.update(value=current_audio_path_state), # 12-13: video_input, audio_input visual (keep)
            gr.update(selected=3) # 14: Stay on finalize tab
        )
        raise gr.Error(error_msg) # Re-raise for Gradio to show error pop-up

def create_args(video_path, audio_path, output_path, inference_steps, guidance_scale, seed):
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=1247)

    return parser.parse_args([
        "--inference_ckpt_path", str(CHECKPOINT_PATH),
        "--video_path", str(video_path),
        "--audio_path", str(audio_path),
        "--video_out_path", str(output_path),
        "--inference_steps", str(inference_steps), # Passed as int already
        "--guidance_scale", str(guidance_scale),
        "--seed", str(seed), # Passed as int already
    ])

# --- Gradio Interface ---
with gr.Blocks(title="LatentSync with Colored Subtitles") as demo:
    gr.Markdown("# LatentSync Video Editor with Colored Subtitles")
    gr.Markdown("---")

    # State Variables (defined at the top level of Blocks)
    video_state = gr.State(None) # Holds path of uploaded video
    audio_state = gr.State(None) # Holds path of uploaded audio (original)
    transcription_state = gr.State(None) # Holds path of the audio that was actually transcribed (could be original or padded)
    latentsync_state = gr.State(None) # Holds path of the video generated by LatentSync
    padded_audio_state = gr.State(None) # (Optional) Holds path of padded audio for potential cleanup (if created)

    with gr.Tabs() as tabs: # The actual tabs component instance
        with gr.TabItem("1. Upload Media", id="tab_upload"):
            gr.Markdown("### Upload your video and audio files.")
            # Corrected: Added height and width for smaller video display
            video_input = gr.Video(label="Input Video", height=300, width=400)
            audio_input = gr.Audio(label="Input Audio", type="filepath")
            status_upload = gr.Textbox(label="Upload Status", interactive=False, lines=1,
                                       value="Please upload both video and audio files.")

        with gr.TabItem("2. Transcribe", id="tab_transcribe"):
            gr.Markdown("### Transcribe the audio and review/edit the transcription.")
            transcription_df = gr.DataFrame(
                headers=["Word", "Start (s)", "End (s)"],
                datatype=["str", "number", "number"],
                row_count=(1, "dynamic"),
                col_count=(3, "fixed"),
                label="Transcription (Edit as needed)",
                interactive=True
            )
            transcribe_btn = gr.Button("Transcribe Audio", interactive=False)
            status_transcribe = gr.Textbox(label="Transcription Status", interactive=False, lines=1)


        with gr.TabItem("3. Run LatentSync", id="tab_latentsync"):
            gr.Markdown("### Adjust LatentSync parameters and run the process.")
            guidance_scale = gr.Slider(1.0, 3.0, 1.5, label="Guidance Scale", interactive=True)
            inference_steps = gr.Slider(10, 50, 20, label="Inference Steps", interactive=True)
            seed = gr.Number(value=1247, label="Seed", interactive=True)
            enable_padding = gr.Checkbox(label="Pad Audio to Video Length (if shorter)", value=True, interactive=True)
            latentsync_btn = gr.Button("Run LatentSync", interactive=False)
            status_latentsync = gr.Textbox(label="LatentSync Status", interactive=False, lines=1)


        with gr.TabItem("4. Finalize Video", id="tab_finalize"):
            gr.Markdown("### Apply subtitles and finalize the output video.")
            font_size = gr.Slider(40, 200, 120, label="Font Size", interactive=True)
            vertical_offset = gr.Slider(0, 100, 24, label="Vertical Offset (%) (0=Bottom, 100=Top)", interactive=True)
            force_4k = gr.Checkbox(label="Force 4K Vertical (2160x3840) Output", value=False, interactive=True)
            finalize_btn = gr.Button("Apply Subtitles & Finalize Video", interactive=False)
            # Corrected: Added height, width, and autoplay for smaller video display
            output_video = gr.Video(label="Output Video", interactive=False, height=300, width=400, autoplay=True)
            status_finalize = gr.Textbox(label="Finalization Status", interactive=False, lines=3)

    gr.Markdown("---")
    gr.Markdown("### Instructions:")
    gr.Markdown("1.  Go to '1. Upload Media' tab, **upload both Video and Audio**. The 'Transcribe Audio' button will become active.")
    gr.Markdown("2.  Go to '2. Transcribe' tab, click 'Transcribe Audio'. Review the transcription.")
    gr.Markdown("3.  Go to '3. Run LatentSync' tab, adjust parameters, click 'Run LatentSync'.")
    gr.Markdown("4.  Go to '4. Finalize Video' tab, adjust subtitle options, click 'Apply Subtitles & Finalize Video'.")
    gr.Markdown("5.  The final video will appear in the 'Finalize Video' tab and be saved to `./processed_videos`.")


    # --- Event Listeners ---

    # Handler for initial media uploads. Updates states, status, and transcribe button.
    video_input.change(
        handle_upload_change,
        inputs=[video_input, audio_input],
        outputs=[video_state, audio_state, status_upload, transcribe_btn]
    )
    audio_input.change(
        handle_upload_change,
        inputs=[video_input, audio_input],
        outputs=[video_state, audio_state, status_upload, transcribe_btn]
    )

    # Transcribe Audio workflow
    transcribe_btn.click(
        transcribe_audio_only,
        inputs=[video_state, audio_state, latentsync_state, padded_audio_state], # Inputs to the function (4)
        outputs=[transcription_df, transcription_state, status_transcribe, # Function's explicit returns for UI (3)
                 transcribe_btn, latentsync_btn, finalize_btn, # Button interactivity updates (3)
                 video_state, audio_state, transcription_state, latentsync_state, padded_audio_state, # State variable updates (5)
                 video_input, audio_input, # Visual clearing/setting of input components (2)
                 tabs] # Tab navigation (1) -> Total 14
    )

    # Run LatentSync workflow
    latentsync_btn.click(
        run_latentsync_process,
        inputs=[video_state, transcription_state, guidance_scale, inference_steps, seed, enable_padding, # Function specific inputs (6)
                video_state, audio_state, transcription_state, latentsync_state, padded_audio_state], # Current states as inputs (5)
        outputs=[latentsync_state, padded_audio_state, status_latentsync, # Function's explicit returns for UI (3)
                 transcribe_btn, latentsync_btn, finalize_btn, # Button interactivity updates (3)
                 video_state, audio_state, transcription_state, latentsync_state, padded_audio_state, # State variable updates (5)
                 video_input, audio_input, # Visual clearing/setting of input components (2)
                 tabs] # Tab navigation (1) -> Total 14
    )

    # Apply Subtitles and Finalize workflow
    finalize_btn.click(
        apply_subtitles_and_finalize_video,
        inputs=[latentsync_state, transcription_df, font_size, vertical_offset, force_4k, # Function-specific inputs (5)
                video_state, audio_state, transcription_state, latentsync_state, padded_audio_state], # Pass-through of all states for resetting/cleanup (5)
        outputs=[output_video, status_finalize, # Function's explicit returns for UI (2)
                 transcribe_btn, latentsync_btn, finalize_btn, # Button interactivity updates (3)
                 transcription_df, # Dataframe reset (1)
                 video_state, audio_state, transcription_state, latentsync_state, padded_audio_state, # State variable updates (5)
                 video_input, audio_input, # Visual clearing/setting of input components (2)
                 tabs] # Tab navigation (1) -> Total 14
    )


if __name__ == "__main__":
    # Ensure necessary directories exist
    FOLDER_FOR_PROCESSED_VIDEOS.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Processed videos will be saved to: {FOLDER_FOR_PROCESSED_VIDEOS.resolve()}")
    print(f"Temporary files will be stored in: {TEMP_DIR.resolve()}")

    # Corrected: Set share=False to prevent public link generation
    demo.launch(inbrowser=True, share=False)