"""
day37_practice.py
캠퍼스 시설 안내 음성 도우미

흐름: 녹음 → STT → 응답 생성 → TTS → 재생
     + 구간별 처리시간 측정(perf_counter) + 막대그래프
     + 개인정보 제외 실행 이력 저장 (JSONL 항상 / Neon DB는 DATABASE_URL 있을 때)

로컬 실행 : streamlit run day37_practice.py
Render 실행: streamlit run day37_practice.py --server.port $PORT --server.address 0.0.0.0
"""
from __future__ import annotations

import asyncio
import io
import json
import math
import os
import re
import struct
import uuid
import wave
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter

import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────
# 1. 기본 설정
# ─────────────────────────────────────────────
APP_TITLE = "캠퍼스 시설 안내 음성 도우미"
LOG_PATH = Path(__file__).with_name("voice_runs_campus.jsonl")
MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10MB
KST = timezone(timedelta(hours=9))
TTS_VOICE = "ko-KR-SunHiNeural"
DB_TABLE = "voice_runs_campus"

STT_DEMO = "데모 STT"
STT_WHISPER = "faster-whisper"
TTS_DEMO = "비프음 데모"
TTS_EDGE = "edge-tts"

# ─────────────────────────────────────────────
# 2. 시설 정보 (운영시간은 연습용 예시)
# ─────────────────────────────────────────────
FACILITIES = {
    "도서관": "도서관은 본관 옆 건물이며 평일 9시부터 18시까지 운영합니다.",
    "학생식당": "학생식당은 학생회관 1층에 있으며 11시부터 14시까지 운영합니다.",
    "취업지원센터": "취업지원센터는 학생회관 2층이며 취업 상담을 제공합니다.",
    "행정실": "행정실은 본관 1층이며 각종 증명서 발급을 안내합니다.",
}

# STT가 띄어쓰기·줄임말로 인식하는 경우 대비 (정식 명칭을 먼저 검사한 뒤 사용)
ALIASES = {
    "학식": "학생식당",
    "식당": "학생식당",
    "취업센터": "취업지원센터",
    "행정사무실": "행정실",
}

DEMO_SENTENCES = [
    "도서관 운영시간 알려줘",
    "학생식당 어디야?",
    "취업지원센터 이용하고 싶어",
    "행정실에서 증명서 발급돼?",
    "운동장에 대해서 알려줘",
]


# ─────────────────────────────────────────────
# 3. 응답 생성
# ─────────────────────────────────────────────
def find_facility(text: str) -> str | None:
    """질문에서 시설명을 찾는다. 공백 제거 후 정식 명칭 → 별칭 순으로 검사."""
    normalized = re.sub(r"\s+", "", text or "")
    for name in FACILITIES:
        if name in normalized:
            return name
    for alias, name in ALIASES.items():
        if alias in normalized:
            return name
    return None


def generate_answer(text: str) -> tuple[str | None, str]:
    facility = find_facility(text)
    if facility:
        return facility, FACILITIES[facility]
    options = ", ".join(FACILITIES)
    return None, f"죄송합니다. 해당 시설은 안내하지 않습니다. 안내 가능한 시설은 {options}입니다."


# ─────────────────────────────────────────────
# 4. 입력 검사
# ─────────────────────────────────────────────
def validate_audio(audio) -> bytes:
    if audio is None:
        raise ValueError("녹음된 음성이 없습니다. 먼저 마이크로 질문을 녹음하세요.")
    data = audio.getvalue()
    if not data:
        raise ValueError("녹음 파일이 비어 있습니다. 다시 녹음하세요.")
    if len(data) > MAX_AUDIO_BYTES:
        raise ValueError(
            f"음성 파일이 너무 큽니다. ({len(data) / 1024 / 1024:.1f}MB / 최대 10MB)"
        )
    return data


# ─────────────────────────────────────────────
# 5. STT
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="faster-whisper 모델 로딩 중... (최초 1회 다운로드)")
def load_whisper(model_size: str):
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device="cpu", compute_type="int8")


def run_stt(mode: str, audio_bytes: bytes, demo_text: str, model_size: str) -> str:
    if mode == STT_DEMO:
        return demo_text
    model = load_whisper(model_size)
    segments, _ = model.transcribe(
        io.BytesIO(audio_bytes), language="ko", beam_size=1, vad_filter=True
    )
    text = "".join(seg.text for seg in segments).strip()
    if not text:
        raise RuntimeError("음성에서 인식된 문장이 없습니다. 조금 더 크게 말해 보세요.")
    return text


# ─────────────────────────────────────────────
# 6. TTS
# ─────────────────────────────────────────────
def make_beep_wav(duration: float = 0.6, freq: int = 880, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = b"".join(
            struct.pack("<h", int(0.3 * 32767 * math.sin(2 * math.pi * freq * i / sr)))
            for i in range(int(sr * duration))
        )
        w.writeframes(frames)
    return buf.getvalue()


async def _edge_tts_bytes(text: str, voice: str) -> bytes:
    import edge_tts

    chunks: list[bytes] = []
    async for chunk in edge_tts.Communicate(text, voice).stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks)


def run_tts(mode: str, text: str) -> tuple[bytes, str]:
    if mode == TTS_DEMO:
        return make_beep_wav(), "audio/wav"
    data = asyncio.run(_edge_tts_bytes(text, TTS_VOICE))
    if not data:
        raise RuntimeError("edge-tts가 빈 음성을 반환했습니다. 인터넷 연결을 확인하세요.")
    return data, "audio/mpeg"


# ─────────────────────────────────────────────
# 7. 파이프라인 (오류가 나도 앱은 계속 동작)
# ─────────────────────────────────────────────
def run_pipeline(audio_bytes: bytes, stt_mode: str, tts_mode: str,
                 demo_text: str, model_size: str) -> tuple[dict, dict]:
    stt_label = f"{STT_WHISPER}-{model_size}" if stt_mode == STT_WHISPER else STT_DEMO
    record = {
        "request_id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now(KST).isoformat(timespec="seconds"),
        "stt_model": stt_label,
        "tts_model": TTS_VOICE if tts_mode == TTS_EDGE else TTS_DEMO,
        "facility": None,
        "matched": False,
        "error": False,
        "error_stage": None,
        "error_type": None,
        "stt_sec": None,
        "response_sec": None,
        "tts_sec": None,
        "total_sec": None,
    }
    result = {"text": None, "answer": None, "audio": None, "mime": None, "error_message": None}

    t_total = perf_counter()
    stage = "stt"
    try:
        t = perf_counter()
        text = run_stt(stt_mode, audio_bytes, demo_text, model_size)
        record["stt_sec"] = round(perf_counter() - t, 4)
        result["text"] = text

        stage = "response"
        t = perf_counter()
        facility, answer = generate_answer(text)
        record["response_sec"] = round(perf_counter() - t, 4)
        record["facility"] = facility
        record["matched"] = facility is not None
        result["answer"] = answer

        stage = "tts"
        t = perf_counter()
        audio_out, mime = run_tts(tts_mode, answer)
        record["tts_sec"] = round(perf_counter() - t, 4)
        result["audio"], result["mime"] = audio_out, mime
    except Exception as e:  # noqa: BLE001 - 어떤 오류든 화면에 표시하고 앱은 유지
        record["error"] = True
        record["error_stage"] = stage
        record["error_type"] = type(e).__name__
        result["error_message"] = f"[{stage}] {type(e).__name__}: {e}"
    finally:
        record["total_sec"] = round(perf_counter() - t_total, 4)
    return record, result


# ─────────────────────────────────────────────
# 8. 이력 저장 - JSONL (음성·인식 문장 등 개인정보는 저장하지 않음)
# ─────────────────────────────────────────────
def append_jsonl(record: dict) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(limit: int = 20) -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame()
    rows = []
    with LOG_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return pd.DataFrame(rows[-limit:][::-1])


# ─────────────────────────────────────────────
# 9. 이력 저장 - Neon(PostgreSQL)
# ─────────────────────────────────────────────
DDL = f"""
CREATE TABLE IF NOT EXISTS {DB_TABLE} (
    request_id   TEXT PRIMARY KEY,
    created_at   TIMESTAMPTZ NOT NULL,
    stt_model    TEXT,
    tts_model    TEXT,
    facility     TEXT,
    matched      BOOLEAN,
    error        BOOLEAN,
    error_stage  TEXT,
    error_type   TEXT,
    stt_sec      DOUBLE PRECISION,
    response_sec DOUBLE PRECISION,
    tts_sec      DOUBLE PRECISION,
    total_sec    DOUBLE PRECISION
)
"""

INSERT_SQL = f"""
INSERT INTO {DB_TABLE} (request_id, created_at, stt_model, tts_model, facility, matched,
    error, error_stage, error_type, stt_sec, response_sec, tts_sec, total_sec)
VALUES (%(request_id)s, %(timestamp)s, %(stt_model)s, %(tts_model)s, %(facility)s, %(matched)s,
    %(error)s, %(error_stage)s, %(error_type)s, %(stt_sec)s, %(response_sec)s, %(tts_sec)s,
    %(total_sec)s)
"""


def get_database_url() -> str:
    url = os.getenv("DATABASE_URL", "")
    if not url:
        try:
            url = st.secrets.get("DATABASE_URL", "")
        except Exception:  # secrets.toml 없음
            url = ""
    url = url.strip().strip('"').strip("'")
    # 흔한 오타 자동 보정 (허용값: disable / prefer / require)
    return url.replace("channel_binding=required", "channel_binding=require")


def db_connect(url: str):
    import psycopg

    return psycopg.connect(url, connect_timeout=10)


@st.cache_resource(show_spinner=False)
def ensure_table(url: str) -> bool:
    with db_connect(url) as conn:
        conn.execute(DDL)
    return True


def save_db(url: str, record: dict) -> None:
    ensure_table(url)
    with db_connect(url) as conn:
        conn.execute(INSERT_SQL, record)


def read_db(url: str, limit: int = 20) -> pd.DataFrame:
    ensure_table(url)
    with db_connect(url) as conn:
        cur = conn.execute(
            f"SELECT * FROM {DB_TABLE} ORDER BY created_at DESC LIMIT %s", (limit,)
        )
        cols = [c.name for c in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


# ─────────────────────────────────────────────
# 10. 화면
# ─────────────────────────────────────────────
def render_timing_chart(record: dict) -> None:
    timing = pd.DataFrame(
        {
            "구간": ["1_STT", "2_응답생성", "3_TTS", "4_총합"],
            "처리시간(초)": [
                record["stt_sec"] or 0,
                record["response_sec"] or 0,
                record["tts_sec"] or 0,
                record["total_sec"] or 0,
            ],
        }
    ).set_index("구간")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("STT", f"{timing.iloc[0, 0]:.3f}s")
    c2.metric("응답", f"{timing.iloc[1, 0] * 1000:.2f}ms")
    c3.metric("TTS", f"{timing.iloc[2, 0]:.3f}s")
    c4.metric("총 처리", f"{timing.iloc[3, 0]:.3f}s")
    st.bar_chart(timing)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🎙️", layout="centered")
    st.title("🎙️ " + APP_TITLE)
    st.caption("안내 시설: " + " · ".join(FACILITIES) + "  |  운영시간은 연습용 예시입니다.")

    db_url = get_database_url()

    # 사이드바: 모델 선택
    with st.sidebar:
        st.header("설정")
        stt_mode = st.radio("STT 방식", [STT_DEMO, STT_WHISPER])
        model_size = "tiny"
        demo_text = DEMO_SENTENCES[0]
        if stt_mode == STT_WHISPER:
            model_size = st.selectbox("Whisper 모델", ["tiny", "base", "small"], index=0,
                                      help="Render 무료 플랜(512MB)에서는 tiny 권장")
        else:
            demo_text = st.selectbox("데모 인식 문장", DEMO_SENTENCES)
        tts_mode = st.radio("TTS 방식", [TTS_DEMO, TTS_EDGE])
        st.divider()
        st.write("이력 저장소")
        st.write("✅ Neon DB + JSONL" if db_url else "📄 JSONL (DATABASE_URL 미설정)")

    # 녹음
    if hasattr(st, "audio_input"):
        audio = st.audio_input("질문을 녹음하세요 (예: 도서관 운영시간 알려줘)")
    else:
        audio = st.file_uploader("WAV 파일 업로드 (Streamlit 1.40 이상에서는 마이크 녹음 지원)",
                                 type=["wav"])

    if st.button("▶ 안내 받기", type="primary", use_container_width=True):
        try:
            audio_bytes = validate_audio(audio)
        except ValueError as e:
            st.error(str(e))  # 녹음 없음 / 용량 초과 → 처리하지 않음
        else:
            with st.spinner("음성 처리 중..."):
                record, result = run_pipeline(audio_bytes, stt_mode, tts_mode,
                                              demo_text, model_size)
            st.session_state["last"] = (record, result)

            try:
                append_jsonl(record)
            except OSError as e:
                st.warning(f"JSONL 저장 실패: {e}")
            if db_url:
                try:
                    save_db(db_url, record)
                except Exception as e:  # noqa: BLE001
                    st.warning(f"DB 실행 이력 저장 실패: {type(e).__name__}: {e}")

    # 결과
    if "last" in st.session_state:
        record, result = st.session_state["last"]
        st.subheader("결과")
        if result["error_message"]:
            st.error("처리 중 오류가 발생했습니다: " + result["error_message"])
        if result["text"]:
            st.write(f"**인식된 질문** : {result['text']}")
        if result["answer"]:
            (st.success if record["matched"] else st.info)(f"**안내** : {result['answer']}")
        if result["audio"]:
            st.audio(result["audio"], format=result["mime"])
        st.caption(f"요청 ID `{record['request_id']}` · {record['timestamp']} · "
                   f"{record['stt_model']} / {record['tts_model']}")

        st.subheader("구간별 처리시간")
        render_timing_chart(record)

    # 최근 이력
    st.subheader("최근 처리 이력")
    st.caption("음성 파일과 인식 문장은 저장하지 않습니다. (요청 ID·시각·모델·오류 여부·처리시간만 기록)")
    history = pd.DataFrame()
    if db_url:
        try:
            history = read_db(db_url)
        except Exception as e:  # noqa: BLE001
            st.warning(f"DB 이력 조회 실패 → JSONL로 대체: {type(e).__name__}: {e}")
    if history.empty:
        history = read_jsonl()
    if history.empty:
        st.write("아직 실행 이력이 없습니다.")
    else:
        st.dataframe(history, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
