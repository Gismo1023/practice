# day37_ex.py
# WAV 파일을 이용한 STT > 응답 생성 > TTS 파이프라인

import asyncio # 비동기 TTS 함수를 실행
import hashlib # 원본 음성의 SHA-256 해시값 계산
import json # 실행 기록을 JSONL 형식으로 저장
import os # DATABASE_URL 환경변수 조회
import tempfile # STT 입력을 위한 임시 WAV 파일 생성
import time # 처리 단계별 실행 시간 측정
import uuid # 요청별 고유 ID 생성
import wave # WAV 파일 헤더와 PCM 형식 검사
from datetime import datetime, timezone # UTC 시각 생성
from pathlib import Path # 파일 경로 처리
import streamlit as st # 웹 기반 사용자 인터페이스
from dotenv import load_dotenv

load_dotenv() # .env 로딩

# 입력 파일 최대 크기를 10Mbytes로 제한
MAX_BYTES = 10 * 1024 * 1024

# 실행이력을 저장할 jsonl파일 경로
LOG_PATH = Path(__file__).with_name('voice_runs.jsonl')

# 사용할 Whisper모델 크기
MODEL_SIZE = 'base'

# Edge TTS에서 사용하는 한국어 여성 음성 모델 ID
VOICE_ID = 'ko-KR-SunHiNeural'

# 실제 WAV 파일인지 확인하고 음성 파일의 기본 정보를 추출하는 함수
# audio: 업로드되거나 녹음된 WAV파일의 바이너리 데이터 (bytes)
def validate_wav(audio):

    import io # WAV 데이터를 메모리 파일처럼 읽기 위한 모듈

    try:

        # 바이너리를 메모리 스트림으로 변환하여 WAV 헤더 읽기
        with wave.open(io.BytesIO(audio), 'rb') as wav:
            channels = wav.getnchannels() # 채널 수
            sample_rate = wav.getframerate() # 샘플링레이트
            sample_width = wav.getsampwidth() # 샘플당 바이트 수
            frames = wav.getnframes() # 전체 오디오 샘플 수
            compression = wav.getcomptype() # 압축 형식 (NONE = 비압축 PCM)

    except (wave.Error, EOFError, OSError) as e:
        raise ValueError('유효한 비압축 PCM WAV 파일을 입력해 주세요.') from e

    if compression != 'NONE' or channels not in (1, 2) or sample_rate < 0 or frames <= 0:
        raise ValueError('모노/스테레오 비압축 PCM WAV 파일이 필요합니다.')

    return {
        'channels': channels,
        'sample_rate': sample_rate,
        'bit_depth': sample_width * 8,
        'duration_sec': round(frames / sample_rate, 2)
    }

# faster-whisper 음성 인식 모델을 CPU에 로딩하는 함수
@st.cache_resource(show_spinner='Wisper 모델을 로딩하고 있습니다...')
def load_stt_model(model_size):
    from faster_whisper import WhisperModel
    return WhisperModel(model_size, device='CPU', compute_type='int8')

# WAV 데이터를 임시 파일로 저장한 후 실제 한국어 음성 인식 수행하는 함수
def transcribe_wav(audio, model_size):

    model = load_stt_model(model_size)

    # faster-whisper가 처리할 수 있도록 임시 WAV 파일 생성
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        f.write(audio) # 업로드된 원본 WAV 바이너리 기록
        wav_path = Path(f.name) # 생성된 임시 파일 경로

    try:

        # 전사 후 segment들과 부가정보 획득
        segments, info = model.transcribe(
            str(wav_path),
            language='ko', # 언어를 한국어로 인식
            vad_filter=True # 음성 활동 탐지로 무음 구간 처리
        )

        # 구간별 텍스트를 공백으로 연결하여 하나의 문장으로 생성
        transcript = ''.join(segment.text.strip() for segment in segments).strip()

        return transcript

    finally:
        # 인식 성공/실패 여부와 관계없이 임시 WAV 파일 제거
        wav_path.unlink(missing_ok=True)

# 인식한 문장을 정리하고 간단한 규칙 기반 응답 문장 생성하는 함수
def build_reply(transcript):
    # split/join으로 연속된 공백과 줄바꿈을 하나의 공백으로 통일
    clean = ''.join(transcript.split())
    if not clean:
        return '음성 인식을 하지 못했습니다. 다시 말씀해 주세요.'
    return f'말씀하신 내용은 {clean}입니다.'

# Edge TTS로 한국어 음성을 합성하여 MP3 bytes로 반환하는 함수
async def generate_tts_mp3(text, voice):
    import edge_tts
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
        mp3_path = Path(f.name)
    try:
        # 문장과 목소리ID를 전달한 후 MP3로 저장
        await edge_tts.Communicate(text=text, voice=voice).save(str(mp3_path))
        return mp3_path.read_bytes()
    finally:
        mp3_path.unlink(missing_ok=True)

# 비동기 Edge TTS 함수를 동기코드에서 실행하는 함수
def synthesize_speech(text, voice_id):
    mp3_bytes = asyncio.run(generate_tts_mp3(text, voice_id))
    return mp3_bytes, 'audio/mpeg' # MP3파일의 bytes와 MIME 반환

# 실행 정보를 JSONL 파일에 추가하고 선택적으로 PostgreSQL에 저장
def store_meta(meta):

    # JSONL은 JSON객체를 한 줄에 하나씩 기록하는 텍스트 파일
    with LOG_PATH.open('a', encoding='utf-8') as f:
        # json.dumps(딕셔너리) : 딕셔너리를 JSON객체 문자열로 변환
        # ensure_ascii=False : 한글로 저장 = ASCII로 변환해서 저장하지 않음
        f.write(json.dumps(meta, ensure_ascii=False) + '\n')

    # 환경변수의 DATABASE_URL 변수의 값
    database_url = os.getenv('DATABASE_URL', '').strip()
    if not database_url:
        return

    try:

        import psycopg

        with psycopg.connect(database_url, connect_timeout=5) as conn:

            with conn.cursor() as cur:

                cur.execute('''
                    CREATE TABLE IF NOT EXISTS voice_runs (
                        request_id TEXT PRIMARY KEY, -- 요청ID
                        created_at TIMESTAMPTZ NOT NULL, -- 생성일시 (타임존)
                        status TEXT NOT NULL, -- 상태문자열
                        stt_ms DOUBLE PRECISION NOT NULL, -- STT 소요시간
                        reply_ms DOUBLE PRECISION NOT NULL, -- 응답 소요시간
                        tts_ms DOUBLE PRECISION NOT NULL, -- TTS 소요시간
                        total_ms DOUBLE PRECISION NOT NULL, -- 소요시간 합계
                        model_version TEXT NOT NULL, -- 모델 버젼 문자열
                        audio_sha256 TEXT NOT NULL -- 오디오 해시값
                    )
                ''')

                cur.execute('''
                    INSERT INTO voice_runs
                    (request_id, created_at, status, stt_ms, reply_ms,
                     tts_ms, total_ms, model_version, audio_sha256)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''', (
                    meta['request_id'], meta['created_at'], meta['status'],
                    meta['stt_ms'], meta['reply_ms'], meta['tts_ms'],
                    meta['total_ms'], meta['model_version'], meta['audio_sha256']
                ))

    except Exception as e:
        st.warning(f'DB 실행 이력 저장 실패: {type(e).__name__}: {e}')

# WAV 검증 > STT > 응답 생성 > TTS > 로그 저장 과정을 연결해서 실행하는 파이프라인 함수
def run_pipeline(audio, model_size, voice_id):

    # 요청을 구분할 고유 요청 ID
    request_id = str(uuid.uuid4())

    # 음성 원문을 기록하지 않고 해시값만 보관
    digest = hashlib.sha256(audio).hexdigest() # SHA256 16진수 해시값 생성

    # 단계별 실행시간 초기화 (ms)
    marks = {'stt_ms': 0.0, 'reply_ms': 0.0, 'tts_ms': 0.0}

    # 전체 처리 시작 시각
    started = time.perf_counter()

    result = None # 성공한 경우 처리 결과 저장
    error = None # 실패한 경우 예외 저장

    try:

        # STT 단계 실행 후 소요시간 측정
        t0 = time.perf_counter()
        transcript = transcribe_wav(audio, model_size)
        marks['stt_ms'] = (time.perf_counter() - t0) * 1000

        # 규칙 기반 응답 생성 단계 실행 후 소요시간 측정
        t0 = time.perf_counter()
        reply = build_reply(transcript)
        marks['reply_ms'] = (time.perf_counter() - t0) * 1000

        # 실제 TTS 음성 합성 단계 실행 후 소요시간 측정
        t0 = time.perf_counter()
        output, mime = synthesize_speech(reply, voice_id)
        marks['tts_ms'] = (time.perf_counter() - t0) * 1000

        # UI에서 사용할 성공 결과를 튜플로 저장
        result = (transcript, reply, output, mime)

    except Exception as e:
        error = e

    # 전체 처리시간에는 최초 모델 로딩 시간도 포함
    elapsed = (time.perf_counter() - started) * 1000

    meta = {
        'request_id': request_id,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'status': 'error' if error else 'success',
        **{key: round(value, 2) for key, value in marks.items()},
        'total_ms': round(elapsed, 2),
        'model_version': f'whisper-{model_size}/edge-tts/{voice_id}',
        'audio_sha256': digest
    }

    store_meta(meta)

    return result, marks, meta, error

# Streamlit 웹 화면 구성, WAV 입력/결과/출력 표시
def main():

    st.set_page_config(
        page_title = '실제 WAV 음성 AI 파이프라인',
        layout = 'centered'
    )

    st.title('실제 WAV 음성 AI 파이프라인')
    st.caption('WAV파일 > faster-whisper STT > 규칙 기반 응답 > edge-tts 음성 합성')

    with st.form('voice_form'):

        source = st.radio('음성 입력 방식', ['WAV파일 업로드', '마이크 녹음'])

        if source == 'WAV파일 업로드':
            uploaded = st.file_uploader('실제 WAV파일을 선택하세요', type=['wav'])
        else:
            uploaded = st.audio_input('마이크로 한국어 음성을 녹음하세요')

        model_size = st.selectbox(
            'STT 모델',
            ['tiny', 'base', 'small', 'medium'],
            index=1
        )

        voice_id = st.selectbox(
            'TTS 목소리',
            ['ko-KR-SunHiNeural', 'ko-KR-InJoonNeural']
        )

        submitted = st.form_submit_button('음성 파이프라인 실행')

    if not submitted:
        return # 실행하지 않았다면 화면 구성만 수행

    if uploaded is None:
        st.error('먼저 WAV파일을 선택하거나 음성을 녹음해 주세요.')
        return

    audio = uploaded.getvalue() # 업로드된 WAV파일의 bytes 읽기

    if not audio or len(audio) > MAX_BYTES:
        st.error(f'WAV 파일은 0바이트보다 크고 {MAX_BYTES}바이트 이하여야 합니다.')
        return

    try:
        audio_info = validate_wav(audio)
    except ValueError as e:
        st.error(str(e))
        return

    st.subheader('입력 WAV 파일')

    st.audio(audio, format='audio/wav')

    st.json(audio_info)

    with st.spinner('실제 음성 인식 및 음성 합성 처리 중...'):
        result, marks, meta, error = run_pipeline(audio, model_size, voice_id)

    if error:
        st.error(f'처리 오류: {type(error).__name__}: {error}')
    else:

        transcript, reply, output, mime = result

        st.subheader('STT 인식 결과')
        st.write(transcript or '(인식된 음성이 없습니다)')

        st.subheader('응답 문장')
        st.write(reply)

        st.subheader('TTS 합성 음성')
        st.audio(output, format=mime)

        st.download_button('TTS MP3 저장', output, 'reply.mp3', mime=mime)

    st.metric('총 처리시간', f"{meta['total_ms']:.0f} ms")

    st.bar_chart(marks)

    st.json(meta)

if __name__ == '__main__':
    main()









































