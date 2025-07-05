import gradio as gr
import cv2
import numpy as np
import tempfile
import os
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

def make_video_from_image_audio(image_path, audio_path):
    try:
        # Debug: Print de paden
        print(f"Image path: {image_path}")
        print(f"Audio path: {audio_path}")

        # Laad afbeelding
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Kan afbeelding niet laden")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # moviepy verwacht RGB

        # Debug: Print afbeelding shape
        print(f"Image shape: {image.shape}")

        # Laad audio en check duur
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration
        print(f"Audio duration: {duration} seconds")

        # Maak een video-clip van de afbeelding met dezelfde duur als audio
        image_clip = ImageClip(image).set_duration(duration)

        # Voeg audio toe aan de videoclip
        video = image_clip.set_audio(audio_clip)

        # Save output tijdelijk
        tmpdir = tempfile.gettempdir()
        output_path = os.path.join(tmpdir, "output_video.mp4")
        video.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac", verbose=False, logger=None)

        print(f"Video saved to: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error: {str(e)}")
        return str(e)

with gr.Blocks() as demo:
    gr.Markdown("# Maak video van afbeelding + audio")
    with gr.Row():
        img_input = gr.Image(label="Upload Afbeelding", type="filepath")  # Zorg voor type="filepath"
        audio_input = gr.Audio(label="Upload Audio", type="filepath")
    out_video = gr.Video(label="Output Video")

    btn = gr.Button("Maak Video")
    btn.click(fn=make_video_from_image_audio, inputs=[img_input, audio_input], outputs=out_video)

if __name__ == "__main__":
    demo.launch()
