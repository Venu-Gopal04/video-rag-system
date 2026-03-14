import os
import base64
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def describe_frame(frame_path: str, timestamp: str) -> str:
    """
    Takes a frame image and returns a text description using Groq's vision model.
    """
    with open(frame_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    prompt = """You are a workplace safety analyst reviewing CCTV footage.
Describe what you see in this video frame in detail. Focus on:
- People present and what they are doing
- Any equipment, vehicles, or machinery visible (especially forklifts)
- Any safety gear (helmets, vests, gloves)
- Any potential hazards or unsafe behaviors
- The location/area type (warehouse, construction site, etc.)
Be specific and factual. Mention if anything looks unsafe."""

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        max_tokens=300
    )

    description = response.choices[0].message.content
    print(f"  Frame {timestamp}: {description[:80]}...")
    return description


def describe_all_frames(frames: list) -> list:
    """
    Describes all extracted frames and returns enriched list with descriptions.
    """
    print(f"\nDescribing {len(frames)} frames with AI vision...")
    enriched = []

    for i, frame_info in enumerate(frames):
        print(f"Processing frame {i+1}/{len(frames)} at {frame_info['timestamp']}...")
        try:
            description = describe_frame(
                frame_info["frame_path"],
                frame_info["timestamp"]
            )
            enriched.append({
                **frame_info,
                "description": description
            })
        except Exception as e:
            print(f"  Error describing frame {frame_info['timestamp']}: {e}")
            enriched.append({
                **frame_info,
                "description": f"Frame at {frame_info['timestamp']} - could not be described."
            })

    print(f"Done describing all frames!")
    return enriched