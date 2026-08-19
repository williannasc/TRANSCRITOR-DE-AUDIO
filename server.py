import os
import shutil
import tempfile
from pathlib import Path

import whisper
from flask import Flask, jsonify, request
from mutagen import File as MutagenFile
from yt_dlp import YoutubeDL

app = Flask(__name__)


def _first_metadata_value(tags, keys):
    for key in keys:
        value = tags.get(key)
        if value is None:
            continue

        if hasattr(value, 'text') and value.text:
            return value.text[0]
        if isinstance(value, (list, tuple)) and value:
            return str(value[0])
        return str(value)

    return None


def _extract_file_metadata(file_path):
    title = None
    artist = None

    try:
        tags = MutagenFile(file_path)
        if tags is None:
            return title, artist

        tag_map = getattr(tags, 'tags', {}) or {}
        title = _first_metadata_value(tag_map, ['TIT2', '©nam', 'title'])
        artist = _first_metadata_value(tag_map, ['TPE1', '©ART', 'artist'])
    except Exception:
        return title, artist

    return title, artist


def _extract_youtube_metadata(video_url):
    options = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
        'extract_flat': False,
        'noplaylist': True,
    }

    try:
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(video_url, download=False)
    except Exception:
        return None, None

    if not info:
        return None, None

    title = info.get('title') or None
    artist = info.get('uploader') or info.get('channel') or info.get('artist') or None
    return title, artist


def _extract_transcript(audio_path, pause_seconds):
    model_name = request.form.get('model', 'small')
    model = whisper.load_model(model_name)
    result = model.transcribe(
        audio_path,
        language='pt',
        temperature=0,
        condition_on_previous_text=False,
    )

    lines = []
    last_end = 0.0

    for segment in result.get('segments', []):
        text = segment.get('text', '').strip()
        if not text:
            continue

        start = float(segment.get('start', 0.0))
        if start - last_end > pause_seconds:
            lines.append('')

        lines.append(text)
        last_end = float(segment.get('end', start))

    return '\n'.join(lines)


def _download_youtube_audio(video_url, temp_dir):
    output_template = os.path.join(temp_dir, 'youtube_audio.%(ext)s')
    option_sets = [
        {
            'format': 'bestaudio/best',
            'noplaylist': True,
            'outtmpl': output_template,
            'quiet': True,
            'no_warnings': True,
            'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
        },
        {
            'format': 'bestaudio/best',
            'noplaylist': True,
            'outtmpl': output_template,
            'quiet': True,
            'no_warnings': True,
            'extractor_args': {'youtube': {'player_client': ['web', 'android']}},
        },
        {
            'format': 'bestaudio/best',
            'noplaylist': True,
            'outtmpl': output_template,
            'quiet': True,
            'no_warnings': True,
            'extractor_args': {'youtube': {'player_client': ['android']}},
        },
    ]

    last_error = None

    for options in option_sets:
        try:
            with YoutubeDL(options) as ydl:
                ydl.download([video_url])

            downloaded_files = [
                os.path.join(temp_dir, name)
                for name in os.listdir(temp_dir)
                if os.path.isfile(os.path.join(temp_dir, name))
            ]

            audio_path = next(
                (
                    item
                    for item in downloaded_files
                    if Path(item).suffix.lower() in {'.mp3', '.wav', '.m4a', '.webm', '.aac', '.opus', '.mp4'}
                ),
                None,
            )

            if audio_path is not None:
                return audio_path

            last_error = FileNotFoundError('Não foi possível localizar o áudio baixado do YouTube.')
        except Exception as exc:  # pragma: no cover
            last_error = exc

    if last_error is not None:
        raise last_error

    raise FileNotFoundError('Não foi possível baixar o áudio do YouTube.')


@app.get('/api/health')
def health():
    return jsonify({'status': 'ok'})


@app.post('/api/transcribe')
def transcribe():
    audio = request.files.get('audio')
    youtube_url = request.form.get('youtubeUrl', '').strip()
    pause_seconds = float(request.form.get('pause', '2.5'))

    if not audio and not youtube_url:
        return jsonify({'error': 'Arquivo de áudio ou link do YouTube é obrigatório.'}), 400

    temp_dir = tempfile.mkdtemp(prefix='transcritor_')
    temp_path = None
    source_title = None
    source_artist = None

    try:
        if youtube_url:
            source_title, source_artist = _extract_youtube_metadata(youtube_url)
            temp_path = _download_youtube_audio(youtube_url, temp_dir)
        else:
            suffix = Path(audio.filename).suffix or '.mp3'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=temp_dir) as temp_file:
                audio.save(temp_file.name)
                temp_path = temp_file.name

            source_title, source_artist = _extract_file_metadata(temp_path)

        transcript = _extract_transcript(temp_path, pause_seconds)
        return jsonify({
            'transcript': transcript,
            'title': source_title or '',
            'artist': source_artist or '',
        })
    except Exception as exc:  # pragma: no cover
        return jsonify({'error': f'Falha na transcrição: {exc}'}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
