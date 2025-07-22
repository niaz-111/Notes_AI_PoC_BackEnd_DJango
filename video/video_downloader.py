import yt_dlp
from tqdm import tqdm
import os
import re

class MyLogger:
    def debug(self, msg): pass
    def warning(self, msg): print(msg)
    def error(self, msg): print(msg)

class ProgressBar:
    def __init__(self):
        self.pbar = None

    def hook(self, d):
        if d['status'] == 'downloading':
            if not self.pbar:
                total = d.get('total_bytes') or d.get('total_bytes_estimate')
                self.pbar = tqdm(total=total, unit='B', unit_scale=True, desc='Downloading')
            downloaded = d.get('downloaded_bytes', 0)
            self.pbar.n = downloaded
            self.pbar.refresh()
        elif d['status'] == 'finished':
            if self.pbar:
                self.pbar.n = self.pbar.total
                self.pbar.refresh()
                self.pbar.close()
                print("\nDownload completed.")

def sanitize_filename(title):
    # Replace special characters and spaces with underscores
    return re.sub(r'[^\w\-_.]', '_', title)

def download_with_ytdlp(url, output_path='temp_uploads'):
    pbar = ProgressBar()
    final_path = None

    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
        info = ydl.extract_info(url, download=False)  # Only get metadata

    clean_title = sanitize_filename(info['title'])
    filename_template = f'{output_path}/{clean_title}.%(ext)s'

    ydl_opts = {
        'outtmpl': filename_template,
        'progress_hooks': [pbar.hook],
        'logger': MyLogger(),
        'quiet': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded_filename = ydl.prepare_filename(info)
        final_path = os.path.abspath(downloaded_filename)

    return final_path

if __name__ == "__main__":
    video_url = input("Enter YouTube video URL: ").strip()
    file_uri = download_with_ytdlp(video_url)
    print(f"\n✔️ Sanitized video path: {file_uri}")
