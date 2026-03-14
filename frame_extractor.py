import cv2
import os

def extract_frames(video_path: str, output_folder: str = "frames", interval_seconds: int = 2):
    """
    Extracts one frame every `interval_seconds` from a video.
    Saves frames as JPG images in output_folder.
    Returns list of frame info dicts with path and timestamp.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Clear old frames
    for f in os.listdir(output_folder):
        os.remove(os.path.join(output_folder, f))

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.1f} seconds")

    frame_interval = int(fps * interval_seconds)
    frame_count = 0
    saved_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            timestamp_seconds = frame_count / fps
            minutes = int(timestamp_seconds // 60)
            seconds = int(timestamp_seconds % 60)
            timestamp_str = f"{minutes:02d}:{seconds:02d}"

            frame_filename = f"frame_{timestamp_str.replace(':', '-')}_{frame_count}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)

            cv2.imwrite(frame_path, frame)

            saved_frames.append({
                "frame_path": frame_path,
                "timestamp": timestamp_str,
                "timestamp_seconds": timestamp_seconds,
                "frame_number": frame_count
            })

            print(f"Saved frame at {timestamp_str} -> {frame_filename}")

        frame_count += 1

    cap.release()
    print(f"\nTotal frames extracted: {len(saved_frames)}")
    return saved_frames