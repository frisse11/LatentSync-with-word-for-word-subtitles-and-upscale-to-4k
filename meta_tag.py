

import subprocess
import random
import datetime
from pathlib import Path
import sys

# A list of realistic device profiles to choose from.
DEVICE_PROFILES = [
    {
        "Make": "Apple",
        "Model": "iPhone 15 Pro Max",
        "Software": "17.5.1"
    },
    {
        "Make": "Apple",
        "Model": "iPhone 14 Pro",
        "Software": "17.4"
    },
    {
        "Make": "Samsung",
        "Model": "SM-S928B", # Galaxy S24 Ultra
        "Software": "Android 14"
    },
    {
        "Make": "Google",
        "Model": "Pixel 8 Pro",
        "Software": "Android 14"
    }
]

# A geographical bounding box to generate plausible GPS coordinates.
# This one covers roughly the Benelux area.
GPS_BOUNDING_BOX = {
    'lat_min': 49.5,
    'lat_max': 53.5,
    'lon_min': 2.5,
    'lon_max': 7.2
}

def apply_random_metadata(video_path: Path) -> bool:
    """
    Applies randomized, realistic metadata to a video file using exiftool.

    Args:
        video_path: The path to the video file to be modified.

    Returns:
        True if the metadata was applied successfully, False otherwise.
    """
    if not video_path.exists():
        print(f"Error: Video file not found at {video_path}")
        return False

    try:
        # 1. Select a random device profile
        profile = random.choice(DEVICE_PROFILES)

        # 2. Generate a random, recent date and time
        now = datetime.datetime.now(datetime.timezone.utc)
        random_days_ago = random.randint(1, 45) # Sometime in the last ~6 weeks
        random_seconds_offset = random.randint(0, 86400)
        random_date = now - datetime.timedelta(days=random_days_ago, seconds=random_seconds_offset)
        formatted_date = random_date.strftime('%Y:%m:%d %H:%M:%S')

        # 3. Generate random GPS coordinates within the bounding box
        latitude = random.uniform(GPS_BOUNDING_BOX['lat_min'], GPS_BOUNDING_BOX['lat_max'])
        longitude = random.uniform(GPS_BOUNDING_BOX['lon_min'], GPS_BOUNDING_BOX['lon_max'])

        # 4. Build the exiftool command
        command = [
            "exiftool",
            "-overwrite_original",
            f"-Make={profile['Make']}",
            f"-Model={profile['Model']}",
            f"-Software={profile['Software']}",
            f"-GPSLatitude={latitude:.4f}",
            "-GPSLatitudeRef=N",
            f"-GPSLongitude={longitude:.4f}",
            "-GPSLongitudeRef=E",
            # Apply the same timestamp to all relevant date fields
            f"-CreationDate={formatted_date}",
            f"-ModifyDate={formatted_date}",
            f"-TrackCreateDate={formatted_date}",
            f"-TrackModifyDate={formatted_date}",
            f"-MediaCreateDate={formatted_date}",
            f"-MediaModifyDate={formatted_date}",
            str(video_path)
        ]

        print(f"Applying metadata to {video_path.name}...")
        print(f"  - Profile: {profile['Make']} {profile['Model']}")
        print(f"  - Timestamp: {formatted_date}")

        # 5. Execute the command
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("Successfully applied metadata.")
        # print(result.stdout) # Uncomment for detailed exiftool output
        return True

    except FileNotFoundError:
        print("\n--- METADATA ERROR ---")
        print("Error: 'exiftool' is not installed or not found in your system's PATH.")
        print("Please install it to enable metadata features.")
        print("You can download it from: https://exiftool.org/")
        print("----------------------\n")
        return False
    except subprocess.CalledProcessError as e:
        print(f"\n--- METADATA ERROR ---")
        print(f"Error: exiftool failed while processing {video_path.name}.")
        print(f"Stderr: {e.stderr}")
        print("----------------------\n")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during metadata application: {e}")
        return False

if __name__ == '__main__':
    # This part allows the script to be tested from the command line.
    # Usage: python meta-tag.py /path/to/your/video.mp4
    if len(sys.argv) > 1:
        video_file = Path(sys.argv[1])
        apply_random_metadata(video_file)
    else:
        print("Usage: python meta-tag.py /path/to/your/video.mp4")

