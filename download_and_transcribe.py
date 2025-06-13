import whisper
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import sys
import yt_dlp

def download_audio(youtube_url: str, output_path: str = "audio.mp4") -> str:
    """
    Download the best audio from the given YouTube URL using yt_dlp,
    saving to output_path.
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'noplaylist': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return output_path

def transcribe_audio(video_path: str, model_name: str = "base", transcript_path: str = "transcript.txt"):
    model = whisper.load_model(model_name)
    result = model.transcribe(video_path)
    with open(transcript_path, "w") as f:
        for seg in result["segments"]:
            f.write(f"{seg['start']:.1f}-{seg['end']:.1f}: {seg['text']}\n")
    print(f"Transcript saved to {transcript_path}")

if __name__ == "__main__":
    # Get URL from command-line argument or prompt the user
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("Enter YouTube video URL: ").strip()
    # Download and transcribe
    mp4_path = download_audio(url)
    transcribe_audio(mp4_path)

