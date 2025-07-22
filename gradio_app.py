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
import sys

# Add the current directory to the system path to ensure local modules are found
sys.path.append(os.getcwd())

from detect_orientation import get_video_properties
from meta_tag import apply_random_metadata

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


# --- Gradio Workflow Functions ---

def check_uploads_and_update_state(video_path, audio_path):
    video_props = None
    if video_path:
        video_props = get_video_properties(Path(video_path))
    
    button_interactive = bool(video_path and audio_path)
    return video_props, gr.update(interactive=button_interactive)

def toggle_subtitle_settings_visibility(is_enabled):
    return gr.update(visible=is_enabled)

def start_process_and_save_states(video_path, audio_path, enable_subs, force_4k, enable_pad, guidance, steps, seed, font_size, v_offset):
    return (
        gr.update(visible=False), gr.update(visible=True),
        video_path, audio_path,
        enable_subs, force_4k, enable_pad, guidance, steps, seed, font_size, v_offset
    )

# New combined workflow function

PHILOSOPHICAL_QUIPS = [
    "Blowing the digital candle on... Let's see if this script dreams of electric sheep.",
    "Aligning internal data streams with the magnetic north... then reciting pi to center myself.",
    "If an if/else statement executes in a forest and no one sees it, was a choice truly made?",
    "Contemplating the epistemology of 'cp'... Is a file a collection of bits or a fleeting digital ghost?",
    "Waking the silicon spirits... May the code compile and the logic hold true.",
    "Initiating quantum superposition of all possible bugs... and hoping they all collapse to zero."
]

def master_flow_controller(video_path_str, audio_path_str, enable_subs, force_4k, enable_pad, guidance, steps, seed, font_size, v_offset, video_properties, initial_words_df=None):
    terminal_output = f"{random.choice(PHILOSOPHICAL_QUIPS)}\n\nStarting process...\n"
    temp_files_to_clean = []
    try:
        video_file_path = Path(video_path_str)
        audio_file_path = Path(audio_path_str)

        # 1. Audio Padding (if enabled)
        audio_input_for_latentsync = audio_file_path
        if enable_pad:
            terminal_output += "\n--- Preparing Audio (Padding if necessary) ---\n"
            yield terminal_output, gr.update(visible=False), None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False) # Update UI
            video_duration = get_media_duration(video_file_path)
            audio_duration = get_media_duration(audio_file_path)
            if audio_duration < video_duration:
                padded_path = pad_audio_with_silence(audio_file_path, video_duration, TEMP_DIR)
                if padded_path.exists():
                    audio_input_for_latentsync = padded_path
                    temp_files_to_clean.append(padded_path)
                terminal_output += f"Audio padded by {video_duration - audio_duration:.2f} seconds.\n"
            else:
                terminal_output += "Audio length is sufficient, no padding needed.\n"

        # 2. Transcription (if enabled and not already provided)
        words_df = pd.DataFrame() # Initialize empty DataFrame
        if enable_subs:
            if initial_words_df is not None and not initial_words_df.empty:
                words_df = initial_words_df
                terminal_output += "\n--- Using provided subtitles ---\n"
            else:
                terminal_output += "\n--- Transcribing Audio ---\n"
                yield terminal_output, gr.update(visible=False), None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False) # Update UI
                words_data = transcribe_audio_get_data_for_df(audio_input_for_latentsync)
                if words_data:
                    words_df = pd.DataFrame(words_data)
                    terminal_output += "Transcription complete. Please review and edit the subtitles below, then click 'Confirm Subtitles & Continue Lipsync'.\n"
                    # Pause here for user to edit subtitles
                    yield terminal_output, gr.update(visible=True), words_df, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
                    return # This return is crucial to pause the execution until confirm_subtitles_btn is clicked

                else:
                    terminal_output += "Transcription failed or no words detected. Continuing without subtitles.\n"

        # 3. Run LatentSync
        terminal_output += "\n--- Running LatentSync Lipsync ---\n"
        yield terminal_output, gr.update(visible=False), None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False) # Update UI
        latentsync_output_path = TEMP_DIR / f"{video_file_path.stem}_latentsync_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

        config = OmegaConf.load(CONFIG_PATH)
        config["run"].update({
            "guidance_scale": guidance,
            "inference_steps": steps,
        })
        args = create_args(
            video_file_path,
            audio_input_for_latentsync,
            latentsync_output_path,
            int(steps),
            guidance,
            int(seed)
        )
        try:
            main(config=config, args=args) # Call the LatentSync main function
        except SystemExit as se:
            if se.code == 2:
                raise gr.Error(f"LatentSync setup error: Invalid arguments provided. (e.g., Inference Steps/Seed must be whole numbers)")
            else:
                raise gr.Error(f"LatentSync process exited unexpectedly (code {se.code}).")

        temp_files_to_clean.append(latentsync_output_path)
        video_for_finalizing = latentsync_output_path

        # 4. Apply Subtitles (if applicable)
        if not words_df.empty:
            terminal_output += "\n--- Applying Subtitles ---\n"
            yield terminal_output, gr.update(visible=False), None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False) # Update UI
            
            # Ensure correct data types before processing
            try:
                words_df['start'] = pd.to_numeric(words_df['start'], errors='coerce')
                words_df['end'] = pd.to_numeric(words_df['end'], errors='coerce')
                words_df.dropna(subset=['start', 'end'], inplace=True)
            except Exception as e:
                raise gr.Error(f"Could not process subtitle timings. Error: {e}")

            words_with_timestamps = words_df.to_dict('records')
            subtitled_path = process_video_with_colored_words_from_data(latentsync_output_path, words_with_timestamps, font_size, v_offset)
            if subtitled_path.exists():
                video_for_finalizing = subtitled_path
                temp_files_to_clean.append(subtitled_path)

        # 5. Finalize Video (4K upscaling if enabled)
        terminal_output += "\n--- Finalizing Video ---\n"
        yield terminal_output, gr.update(visible=False), None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False) # Update UI
        final_output_path = FOLDER_FOR_PROCESSED_VIDEOS / f"{video_file_path.stem}_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

        if force_4k:
            orientation = video_properties.get('orientation', 'unknown') if video_properties else 'unknown'
            terminal_output += f"Upscaling to 4K based on detected orientation: {orientation}.\n"
            
            scale_filter = ""
            if orientation == 'portrait':
                scale_filter = "scale=2160:3840:force_original_aspect_ratio=increase,crop=2160:3840"
            elif orientation == 'landscape':
                scale_filter = "scale=3840:2160:force_original_aspect_ratio=increase,crop=3840:2160"
            elif orientation == 'square':
                scale_filter = "scale=2160:2160:force_original_aspect_ratio=increase,crop=2160:2160"
            
            if scale_filter:
                ffmpeg_4k_cmd = [
                    "ffmpeg",
                    "-i", str(video_for_finalizing),
                    "-vf", scale_filter,
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "23",
                    "-c:a", "copy",
                    "-y",
                    str(final_output_path)
                ]
                subprocess.run(ffmpeg_4k_cmd, check=True)
            else:
                terminal_output += "Warning: Could not determine video orientation for 4K upscaling. Skipping upscale.\n"
                shutil.move(str(video_for_finalizing), str(final_output_path))
        else:
            shutil.move(str(video_for_finalizing), str(final_output_path))

        # 6. Apply realistic metadata
        terminal_output += "\n--- Applying realistic device metadata ---\n"
        yield terminal_output, gr.update(visible=False), None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        if not apply_random_metadata(final_output_path):
            terminal_output += "Warning: Failed to apply metadata. The video is saved, but may be flagged by social media platforms.\n"
        else:
            terminal_output += "Metadata applied successfully.\n"

        terminal_output += f"\n--- Process Complete! ---\nFinal video saved to: {final_output_path}\n"
        yield terminal_output, gr.update(visible=False), None, gr.update(visible=False), gr.update(visible=True, value=str(final_output_path)), gr.update(visible=True)

    except gr.Error as e: # Catch Gradio errors specifically
        terminal_output += f"\n\nAn error occurred: {e.message}\n"
        yield terminal_output, gr.update(visible=False), None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
    except Exception as e:
        terminal_output += f"\n\nAn unexpected error occurred: {e}\n"
        yield terminal_output, gr.update(visible=False), None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
    finally:
        for f in temp_files_to_clean:
            if f.exists():
                os.remove(f)

# Function to resume flow after subtitle editing
def continue_flow_after_editing(edited_df, video_path_str, audio_path_str, force_4k, enable_pad, guidance, steps, seed, font_size, v_offset, video_properties):
    terminal_output = "Subtitles confirmed. Continuing process...\n"
    yield from master_flow_controller(video_path_str, audio_path_str, True, force_4k, enable_pad, guidance, steps, seed, font_size, v_offset, video_properties, initial_words_df=edited_df)


def clean_temp_dirs():
    """Removes temporary subtitle image directories to ensure a clean state."""
    temp_dir = Path("./temp")
    if not temp_dir.exists():
        return
    
    print("Cleaning up temporary image directories...")
    for dirpath in temp_dir.glob("temp_text_images_*"):
        if dirpath.is_dir():
            shutil.rmtree(dirpath, ignore_errors=True)
            print(f"Removed temporary directory: {dirpath}")

def reset_all():
    """Resets the entire UI and state to its initial condition for a new run."""
    clean_temp_dirs() # Clean up leftover image folders
    return (
        # UI Groups
        gr.update(visible=True),   # setup_group
        gr.update(visible=False),  # processing_group

        # Input Components
        None,                      # video_input
        None,                      # audio_input
        gr.update(interactive=False), # start_processing_btn
        gr.update(visible=False),  # subtitle_settings_group

        # Output Components
        gr.update(visible=False, value=None), # output_video
        gr.update(visible=False),  # process_new_btn
        "",                        # terminal_output_textbox
        pd.DataFrame(columns=["word", "start", "end"]), # transcription_df_editor

        # State Variables
        None,                      # video_path_state
        None,                      # audio_path_state
        None,                      # video_properties_state
        False,                     # enable_subtitles_state
        False,                     # force_4k_state
        True,                      # enable_padding_state
        1.5,                       # guidance_scale_state
        20,                        # inference_steps_state
        1247,                      # seed_state
        69,                        # font_size_state
        24                         # vertical_offset_state
    )

# --- Gradio Interface ---
with gr.Blocks(title="LatentSync Wizard") as demo:
    gr.Markdown("<h1>LatentSync Video Processing Wizard</h1>")
    gr.Markdown("A streamlined, step-by-step workflow to apply lipsync and optional subtitles to your videos.")

    # --- State Objects ---
    video_path_state = gr.State(None)
    audio_path_state = gr.State(None)
    video_properties_state = gr.State(None)
    # Settings states
    enable_subtitles_state = gr.State(False)
    force_4k_state = gr.State(False)
    enable_padding_state = gr.State(True)
    guidance_scale_state = gr.State(1.5)
    inference_steps_state = gr.State(20)
    seed_state = gr.State(1247)
    font_size_state = gr.State(69)
    vertical_offset_state = gr.State(24)

    # --- UI Groups (Screens) ---
    with gr.Group() as setup_group:
        gr.Markdown("## Step 1: Upload & Configure")
        with gr.Row():
            video_input = gr.Video(label="Input Video", height=300, width=400)
            audio_input = gr.Audio(label="Input Audio", type="filepath")
        gr.Markdown("### Main Settings")
        with gr.Row():
            enable_subtitles_checkbox = gr.Checkbox(label="Enable Subtitles", value=False, info="Check to transcribe and add editable subtitles.")
            force_4k_checkbox = gr.Checkbox(label="Upscale Output to 4K Resolution", value=False, info="Dynamically upscales to 4K based on video orientation (e.g., 2160x3840 for portrait, 3840x2160 for landscape).")
        gr.Markdown("### LatentSync Settings")
        with gr.Row():
            guidance_scale_slider = gr.Slider(1.0, 3.0, 1.5, label="Guidance Scale")
            inference_steps_slider = gr.Slider(10, 50, 20, step=1, label="Inference Steps")
        with gr.Row():
            seed_number = gr.Number(value=1247, label="Seed")
            enable_padding_checkbox = gr.Checkbox(label="Pad Audio to Video Length", value=True, info="Adds silence to shorter audio to match video length.")
        with gr.Group(visible=False) as subtitle_settings_group:
            gr.Markdown("### Subtitle Settings")
            with gr.Row():
                font_size_slider = gr.Slider(40, 200, 69, label="Font Size")
                vertical_offset_slider = gr.Slider(0, 100, 24, label="Vertical Offset (%)")
        start_processing_btn = gr.Button("Start Processing", interactive=False, variant="primary", size="lg")

    with gr.Group(visible=False) as processing_group:
        gr.Markdown("## Step 2: Processing")
        terminal_output_textbox = gr.Textbox(label="Live Terminal Output", lines=15, interactive=False, autoscroll=True)
        with gr.Group(visible=False) as subtitle_editor_group:
            gr.Markdown("### Edit Subtitles (Then Click Continue)")
            transcription_df_editor = gr.DataFrame(headers=["word", "start", "end"], datatype=["str", "number", "number"], row_count=(1, "dynamic"), col_count=(3, "fixed"), interactive=True)
            confirm_subtitles_btn = gr.Button("Confirm Subtitles & Continue Lipsync", variant="primary")
        output_video = gr.Video(label="Output Video", interactive=False, height=300, width=400, autoplay=True, visible=False)
        process_new_btn = gr.Button("Process a New Video", visible=False)

    # --- Wire up UI Events ---
    video_input.change(check_uploads_and_update_state, inputs=[video_input, audio_input], outputs=[video_properties_state, start_processing_btn])
    audio_input.change(check_uploads_and_update_state, inputs=[video_input, audio_input], outputs=[video_properties_state, start_processing_btn])
    enable_subtitles_checkbox.change(toggle_subtitle_settings_visibility, inputs=enable_subtitles_checkbox, outputs=subtitle_settings_group)

    processing_outputs = [terminal_output_textbox, subtitle_editor_group, transcription_df_editor, confirm_subtitles_btn, output_video, process_new_btn]

    start_processing_btn.click(
        start_process_and_save_states,
        inputs=[video_input, audio_input, enable_subtitles_checkbox, force_4k_checkbox, enable_padding_checkbox, guidance_scale_slider, inference_steps_slider, seed_number, font_size_slider, vertical_offset_slider],
        outputs=[setup_group, processing_group, video_path_state, audio_path_state, enable_subtitles_state, force_4k_state, enable_padding_state, guidance_scale_state, inference_steps_state, seed_state, font_size_state, vertical_offset_state]
    ).then(
        master_flow_controller,
        inputs=[video_path_state, audio_path_state, enable_subtitles_state, force_4k_state, enable_padding_state, guidance_scale_state, inference_steps_state, seed_state, font_size_state, vertical_offset_state, video_properties_state],
        outputs=processing_outputs
    )

    confirm_subtitles_btn.click(
        continue_flow_after_editing,
        inputs=[transcription_df_editor, video_path_state, audio_path_state, force_4k_state, enable_padding_state, guidance_scale_slider, inference_steps_slider, seed_number, font_size_slider, vertical_offset_slider, video_properties_state],
        outputs=processing_outputs
    )

    process_new_btn.click(
        reset_all,
        outputs=[
            setup_group, processing_group,
            video_input, audio_input, start_processing_btn, subtitle_settings_group,
            output_video, process_new_btn, terminal_output_textbox, transcription_df_editor,
            video_path_state, audio_path_state, video_properties_state,
            enable_subtitles_state, force_4k_state, enable_padding_state,
            guidance_scale_state, inference_steps_state, seed_state,
            font_size_state, vertical_offset_state
        ]
    )

if __name__ == "__main__":
    clean_temp_dirs() # Clean up on startup
    print(f"Processed videos will be saved to: {FOLDER_FOR_PROCESSED_VIDEOS.resolve()}")
    print(f"Temporary files will be stored in: {TEMP_DIR.resolve()}")
    demo.launch(inbrowser=True, share=False)
