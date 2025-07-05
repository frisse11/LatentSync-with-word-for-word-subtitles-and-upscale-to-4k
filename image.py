# This Gradio app allows the user to upload a video and displays it in a smaller, scaled-down box.
import gradio as gr

# Define a function that takes a video and returns it.
def video_display(video):
    return video

# Create a Gradio interface that takes a video input and displays it in a video output with a smaller size.
# The video output is set to a smaller height and width to make the video box smaller.
demo = gr.Interface(
    fn=video_display, 
    inputs=gr.Video(height=300, width=400), 
    outputs=gr.Video(height=300, width=400)  # Set the height and width to make the video smaller
)

# Launch the interface.
demo.launch(show_error=True)
