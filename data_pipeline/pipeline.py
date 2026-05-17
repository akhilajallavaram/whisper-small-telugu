import subprocess
import sys

def run_script(script_name):
    print(f"\n========================================================")
    print(f"========== Running {script_name} ==========")
    print(f"========================================================\n")
    
    # Use python3 to ensure it runs correctly on Mac
    result = subprocess.run([sys.executable, script_name])
    
    if result.returncode != 0:
        print(f"\n[ERROR] Pipeline failed at step: {script_name}")
        sys.exit(result.returncode)
        
    print(f"\n========== Completed {script_name} ==========\n")

if __name__ == "__main__":
    print("Starting Telugu Whisper Data Pipeline...")
    
    scripts = [
        "extract_telugu_data.py",   # Step 1: Extract metadata and select videos
        "download_audio.py",        # Step 2: Download audio from YouTube
        "segment_audio.py",         # Step 3: Segment and filter the audio clips
        "generate_transcripts.py"   # Step 4: Transcribe the segments using Whisper
    ]
    
    for script in scripts:
        run_script(script)
    
    print("\n========================================================")
    print("========== PIPELINE COMPLETED SUCCESSFULLY! ==========")
    print("========================================================")
