import os
from pathlib import Path

OUTPUT_DIR = "generated_videos"
Path(OUTPUT_DIR).mkdir(exist_ok=True)


def generate_video(story_id: int, image_paths: list[str], story_title: str) -> str:
    """
    Create a slideshow MP4 video from comic images.
    Compatible with MoviePy v2.x (moviepy.editor was removed in v2).
    """
    # MoviePy v2 imports (no more moviepy.editor)
    from moviepy import ImageClip, concatenate_videoclips

    SECONDS_PER_IMAGE = 3
    FPS = 24

    clips = []
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"[video_gen] Warning: image not found — {img_path}, skipping.")
            continue

        # ImageClip in v2 uses duration parameter
        img_clip = ImageClip(img_path).with_duration(SECONDS_PER_IMAGE)
        clips.append(img_clip)
        print(f"[video_gen] Added clip: {img_path}")

    if not clips:
        raise ValueError("No valid images found to generate video.")

    # Concatenate all panels
    final_video = concatenate_videoclips(clips, method="compose")

    output_path = f"{OUTPUT_DIR}/video_{story_id}.mp4"
    final_video.write_videofile(output_path, fps=FPS, logger=None)
    print(f"[video_gen] Video saved: {output_path}")

    final_video.close()
    for clip in clips:
        clip.close()

    return output_path
