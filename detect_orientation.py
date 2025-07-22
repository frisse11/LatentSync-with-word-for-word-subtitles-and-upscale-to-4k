
import subprocess
import json
from pathlib import Path

def get_video_properties(video_path: Path) -> dict:
    """
    Analyzes a video file to determine its orientation, width, and height.

    Args:
        video_path: The path to the video file.

    Returns:
        A dictionary containing the orientation ('portrait', 'landscape', 'square', or 'unknown'),
        width, and height.
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        streams = json.loads(result.stdout).get("streams", [])
        
        video_stream = None
        for stream in streams:
            if stream.get("codec_type") == "video":
                video_stream = stream
                break
        
        if not video_stream:
            raise ValueError("No video stream found in the file.")

        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))

        if width == 0 or height == 0:
             raise ValueError("Could not determine video dimensions.")

        if height > width:
            orientation = 'portrait'
        elif width > height:
            orientation = 'landscape'
        else:
            orientation = 'square'
            
        return {'orientation': orientation, 'width': width, 'height': height}

    except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError, FileNotFoundError) as e:
        print(f"Warning: Could not get video properties for {video_path}. Error: {e}")
        return {'orientation': 'unknown', 'width': 0, 'height': 0}

