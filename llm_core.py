"""
LLM Core — ส่วนกลางที่ใช้ร่วมกันระหว่าง CLI (`chatbot.py`) และเว็บ (`app.py`)

ประกอบด้วย
    - การโหลดและตรวจสอบค่าตั้งจาก .env
    - การจัดการ prompt และ conversation context
    - การเรียก LLM ผ่าน openai SDK (SGLang บน DGX Spark)
    - การแปลง error เป็นข้อความไทยที่บอกทางแก้
    - สถิติการใช้งาน (token, เวลา, ความเร็ว)

ไฟล์นี้ไม่มีการพิมพ์ออกจอเลย ทุกอย่างคืนค่าเป็นข้อมูล
เพื่อให้ front-end แต่ละแบบนำไปแสดงผลในสไตล์ของตัวเองได้
"""

import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

# ============================================
# CONFIG SPEC
# ============================================
# ตารางนี้คือรายการค่าตั้งทั้งหมดที่อ่านจาก .env
# รูปแบบ: (ชื่อคีย์ใน .env, ชื่อ attribute, ชนิดข้อมูล, บังคับ, คำอธิบาย)
# ถ้าคีย์บังคับหายไป โปรแกรมจะแจ้ง error และไม่เริ่มทำงาน

CONFIG_SPEC = (
    (
        "LLM_BASE_URL",
        "base_url",
        "str",
        True,
        "ที่อยู่ API ของ SGLang (ต้องมี /v1 ต่อท้าย)",
    ),
    (
        "LLM_MODEL",
        "model",
        "str",
        True,
        "ชื่อ model ต้องตรงกับ --served-model-name ของ SGLang",
    ),
    (
        "LLM_API_KEY",
        "api_key",
        "str",
        True,
        "API key — SGLang ปกติไม่ต้องใช้ ให้ใส่ dummy",
    ),
    (
        "LLM_SYSTEM_PROMPT",
        "system_prompt",
        "str",
        True,
        "persona ของผู้ช่วย",
    ),
    (
        "LLM_MAX_HISTORY_MESSAGES",
        "max_history_messages",
        "int",
        True,
        "จำนวนข้อความสูงสุดที่เก็บใน context (system message ไม่นับ)",
    ),
    (
        "LLM_TEMPERATURE",
        "temperature",
        "float",
        True,
        "ความสร้างสรรค์ของคำตอบ (0-2)",
    ),
    (
        "LLM_DISABLE_THINKING",
        "disable_thinking",
        "bool",
        True,
        "ปิดโหมดคิดก่อนตอบ ของ reasoning model (true = ตอบเร็วและประหยัด token)",
    ),
    (
        "LLM_SHOW_REASONING",
        "show_reasoning",
        "bool",
        True,
        "แสดงกระบวนการคิดของโมเดล (ใช้เมื่อไม่ได้ปิด thinking)",
    ),
    (
        "LLM_MAX_COMPLETION_TOKENS",
        "max_completion_tokens",
        "int",
        False,
        "จำนวน token สูงสุดของคำตอบ — เว้นว่างเพื่อใช้ค่า default ของ server",
    ),
    (
        "LLM_REQUEST_TIMEOUT",
        "request_timeout",
        "float",
        True,
        "timeout ต่อคำขอ หน่วยวินาที",
    ),
    (
        "LLM_MAX_RETRIES",
        "max_retries",
        "int",
        True,
        "จำนวนครั้งที่ SDK จะ retry อัตโนมัติ",
    ),
    (
        "LLM_INCLUDE_USAGE",
        "include_usage",
        "bool",
        True,
        "ขอ token usage ตอน stream (true/false)",
    ),
    (
        "LLM_THINKING_NOTICE_SECONDS",
        "thinking_notice_seconds",
        "float",
        True,
        "จำนวนวินาทีที่เงียบแล้วควรแจ้งว่ากำลังรอ (CLI ใช้แสดงข้อความ)",
    ),
    (
        "LLM_LOG_DIR",
        "log_dir",
        "str",
        True,
        "โฟลเดอร์เก็บประวัติการสนทนา",
    ),
)

ENV_FILE = ".env"
ENV_EXAMPLE_FILE = ".env.example"


# ============================================
# CONFIG LOADING
# ============================================


class ConfigError(Exception):
    """ค่าตั้งใน .env ไม่ครบหรือรูปแบบผิด"""

    def __init__(self, problems, env_file_found):
        super().__init__("ค่าตั้งใน .env ไม่ถูกต้อง")
        self.problems = problems
        self.env_file_found = env_file_found


class Config:
    """ค่าตั้งทั้งหมดของโปรแกรม อ่านมาจาก .env เท่านั้น"""

    def __init__(self, values):
        for key, attribute, *_ in CONFIG_SPEC:
            setattr(self, attribute, values[key])

        # ตัด / ตัวสุดท้ายออก เพราะ SDK จะต่อ /chat/completions เอง
        self.base_url = self.base_url.rstrip("/")


def _convert(raw, kind):
    """แปลงข้อความจาก .env เป็นชนิดข้อมูลที่ต้องการ (raise ValueError ถ้าแปลงไม่ได้)"""
    if kind == "str":
        return raw
    if kind == "int":
        return int(raw)
    if kind == "float":
        return float(raw)
    if kind == "bool":
        lowered = raw.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
        raise ValueError("ค่าต้องเป็น true หรือ false")
    raise ValueError(f"ชนิดข้อมูลไม่รองรับ: {kind}")


def load_config():
    """
    อ่านค่าตั้งทั้งหมดจากไฟล์ .env

    raise ConfigError ถ้าคีย์บังคับหายไปหรือรูปแบบผิด
    """
    env_file_found = load_dotenv()

    values = {}
    problems = []

    for key, attribute, kind, required, description in CONFIG_SPEC:
        raw = os.getenv(key)

        if raw is None or not raw.strip():
            if required:
                problems.append(f"{key} — {description}")
            else:
                values[key] = None
            continue

        try:
            values[key] = _convert(raw.strip(), kind)
        except ValueError as exc:
            problems.append(f'{key} — ค่าที่อ่านได้ "{raw}" ใช้ไม่ได้ ({exc})')

    if problems:
        raise ConfigError(problems, env_file_found)

    return Config(values)


def describe_config_error(error):
    """
    คืนรายการบรรทัดที่อธิบายวิธีแก้ เมื่อค่าตั้งใน .env ไม่ครบหรือผิด

    front-end แต่ละแบบนำไปแสดงเอง (CLI พิมพ์ออกจอ, เว็บใช้ st.error)
    """
    lines = ["ค่าตั้งใน .env ไม่ถูกต้อง", ""]

    if not error.env_file_found:
        lines.append(f"ไม่พบไฟล์ {ENV_FILE}")
        lines.append(f"เริ่มต้นด้วยคำสั่ง: Copy-Item {ENV_EXAMPLE_FILE} {ENV_FILE}")
        lines.append("")

    lines.append("รายการที่ต้องแก้:")
    for problem in error.problems:
        lines.append(f"  - {problem}")

    lines.append("")
    lines.append(f"ดูตัวอย่างค่าที่ถูกต้องได้จากไฟล์ {ENV_EXAMPLE_FILE}")

    return lines


# ============================================
# PROMPT
# ============================================


def build_system_message(prompt):
    """สร้าง system message จาก persona ที่ตั้งไว้ใน .env"""
    return {"role": "system", "content": prompt}


# ============================================
# CONVERSATION CONTEXT
# ============================================


class ConversationContext:
    """เก็บประวัติการสนทนาและจำกัดไม่ให้ context ยาวเกินกำหนด"""

    def __init__(self, system_prompt, max_messages):
        self.system_message = build_system_message(system_prompt)
        self.max_messages = max_messages
        self.messages = []
        self.trim_count = 0

    def to_payload(self):
        """คืน messages สำหรับส่งให้ API โดยมี system message อยู่ตัวแรกเสมอ"""
        return [self.system_message] + self.messages

    def add_user(self, text):
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text):
        self.messages.append({"role": "assistant", "content": text})

    def trim(self):
        """
        ตัดข้อความเก่าสุดออกทีละคู่ (คำถาม + คำตอบ) จนไม่เกิน max_messages

        ตัดเป็นคู่เพื่อไม่ให้ประวัติเริ่มต้นด้วยคำตอบของ assistant ซึ่งจะทำให้โมเดลสับสน
        คืนค่า True ถ้ามีการตัดเกิดขึ้น
        """
        trimmed = False
        while len(self.messages) > self.max_messages:
            if self.messages[0]["role"] == "user":
                self.messages.pop(0)
                if self.messages and self.messages[0]["role"] == "assistant":
                    self.messages.pop(0)
            else:
                self.messages.pop(0)
            trimmed = True

        if trimmed:
            self.trim_count += 1
        return trimmed

    def drop_last_user_message(self):
        """ถอนข้อความผู้ใช้ล่าสุดออก ใช้เมื่อคำขอล้มเหลวเพื่อไม่ให้ context เพี้ยน"""
        if self.messages and self.messages[-1]["role"] == "user":
            return self.messages.pop()
        return None

    def reset(self):
        """ล้างประวัติทั้งหมด คืนค่าจำนวนข้อความที่ถูกล้าง"""
        count = len(self.messages)
        self.messages.clear()
        self.trim_count = 0
        return count

    def stats(self):
        """สรุปสถิติของ context ปัจจุบัน"""
        return {
            "messages": len(self.messages),
            "turns": sum(1 for m in self.messages if m["role"] == "user"),
            "trims": self.trim_count,
        }


# ============================================
# TRANSCRIPT LOG
# ============================================


class TranscriptLogger:
    """บันทึกบทสนทนาลงไฟล์ log (โฟลเดอร์ logs/ ถูก .gitignore ไว้แล้ว)"""

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.log_dir / f"chat_{stamp}.log"
        self.path.touch(exist_ok=True)

    def _write(self, role, text):
        stamp = datetime.now().strftime("%H:%M:%S")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{stamp}] {role}: {text}\n")
            handle.flush()

    def log_user(self, text):
        self._write("ผู้ใช้", text)

    def log_assistant(self, text):
        self._write("ผู้ช่วย", text)

    def log_event(self, text):
        self._write("ระบบ", text)


# ============================================
# SESSION METRICS
# ============================================


class TurnTiming:
    """เก็บเวลาแต่ละช่วงของหนึ่งเทิร์น เพื่อดูว่า DGX ทำงานเร็วแค่ไหน"""

    def __init__(self):
        self.started = time.monotonic()
        self.first_token = None
        self.finished = None

    def mark_first_token(self):
        """เรียกเมื่อได้อะไรจากโมเดลเป็นชิ้นแรก ใช้คำนวณ time-to-first-token"""
        if self.first_token is None:
            self.first_token = time.monotonic()

    def finish(self):
        """ปิดการจับเวลา เรียกซ้ำได้แต่จะใช้ค่าแรก"""
        if self.finished is None:
            self.finished = time.monotonic()

    @property
    def ttft(self):
        """เวลาที่รอจนได้ token แรก (วินาที) — None ถ้าไม่ได้อะไรเลย"""
        if self.first_token is None:
            return None
        return self.first_token - self.started

    @property
    def elapsed(self):
        """เวลารวมของเทิร์นนี้ (วินาที)"""
        end = self.finished if self.finished is not None else time.monotonic()
        return end - self.started

    @property
    def gen_time(self):
        """เวลาที่ใช้สร้างคำตอบหลังได้ token แรก (วินาที)"""
        if self.first_token is None:
            return 0.0
        return self.elapsed - self.ttft


class SessionMetrics:
    """เก็บสถิติทั้ง session — ทั้งจำนวน token และเวลา เพื่อวัดประสิทธิภาพของ DGX"""

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.requests_ok = 0
        self.requests_failed = 0
        self.has_usage = False

        # ข้อมูลของคำขอล่าสุด ใช้ดูว่า context ใกล้เต็มหน้าต่างของโมเดลหรือยัง
        self.last_prompt_tokens = None

        # สถิติด้านเวลา (วินาที)
        self.timed_turns = 0
        self.ttft_total = 0.0
        self.ttft_min = None
        self.ttft_max = None
        self.elapsed_total = 0.0
        self.gen_time_total = 0.0

    def add_usage(self, usage):
        """สะสมค่าจาก usage ที่ server ส่งกลับ (เป็น None ได้ถ้า server ไม่ส่งมา)"""
        if usage is None:
            return
        self.has_usage = True
        self.prompt_tokens += usage.prompt_tokens or 0
        self.completion_tokens += usage.completion_tokens or 0
        self.total_tokens += usage.total_tokens or 0
        self.last_prompt_tokens = usage.prompt_tokens

    def add_timing(self, timing):
        """สะสมเวลาของเทิร์นที่สำเร็จ เพื่อคำนวณค่าเฉลี่ยทั้ง session"""
        self.timed_turns += 1
        self.elapsed_total += timing.elapsed
        self.gen_time_total += timing.gen_time

        if timing.ttft is not None:
            self.ttft_total += timing.ttft
            self.ttft_min = (
                timing.ttft if self.ttft_min is None else min(self.ttft_min, timing.ttft)
            )
            self.ttft_max = (
                timing.ttft if self.ttft_max is None else max(self.ttft_max, timing.ttft)
            )

    def average_ttft(self):
        """เวลาเฉลี่ยที่รอจนได้ token แรก (วินาที)"""
        if not self.timed_turns or not self.ttft_total:
            return None
        return self.ttft_total / self.timed_turns

    def average_elapsed(self):
        """เวลาเฉลี่ยต่อเทิร์น (วินาที)"""
        if not self.timed_turns:
            return None
        return self.elapsed_total / self.timed_turns

    def tokens_per_second(self):
        """ความเร็วในการสร้างคำตอบเฉลี่ยทั้ง session"""
        if self.gen_time_total <= 0 or not self.has_usage:
            return None
        return self.completion_tokens / self.gen_time_total

    def describe_turn(self, usage, timing, max_context):
        """
        สรุปผลของเทิร์นล่าสุดเป็นรายการบรรทัด

        max_context คือความจุสูงสุดของโมเดล (token) ใส่ None ได้ถ้าไม่ทราบ
        """
        lines = []

        # --- เวลา ---
        if timing.ttft is None:
            lines.append(f"⏱ ใช้เวลา {timing.elapsed:.2f}s (ไม่ได้ token ใด ๆ)")
        else:
            text = f"⏱ {timing.ttft:.2f}s ถึง token แรก · {timing.elapsed:.2f}s รวม"
            if usage is not None and timing.gen_time > 0:
                text += f" · {usage.completion_tokens / timing.gen_time:.1f} tok/s"
            lines.append(text)

        # --- token และ context ---
        if usage is None:
            lines.append("📊 server ไม่ได้ส่งข้อมูล token มา")
        else:
            text = (
                f"token {usage.prompt_tokens} prompt + {usage.completion_tokens} completion "
                f"= {usage.total_tokens}"
            )
            if max_context:
                percent = usage.prompt_tokens / max_context * 100
                text += f" · context {usage.prompt_tokens:,}/{max_context:,} ({percent:.2f}%)"
            lines.append(f"📊 {text}")

        return lines

    def describe_session(self, context, max_context):
        """สรุปสถิติทั้ง session สำหรับแสดงใน CLI และ sidebar ของเว็บ"""
        lines = []

        lines.append(
            f"จำนวนเทิร์น        : {self.requests_ok + self.requests_failed} "
            f"(สำเร็จ {self.requests_ok} · ล้มเหลว {self.requests_failed})"
        )

        if self.timed_turns:
            average_ttft = self.average_ttft()
            if average_ttft is not None:
                lines.append(
                    f"TTFT               : เฉลี่ย {average_ttft:.2f}s "
                    f"(ต่ำสุด {self.ttft_min:.2f}s · สูงสุด {self.ttft_max:.2f}s)"
                )
            lines.append(
                f"เวลาตอบ            : เฉลี่ย {self.average_elapsed():.2f}s "
                f"(รวม {self.elapsed_total:.2f}s)"
            )
            speed = self.tokens_per_second()
            if speed is not None:
                lines.append(f"ความเร็ว           : เฉลี่ย {speed:.1f} token/วินาที")

        if self.has_usage:
            lines.append(
                f"token รวม          : {self.total_tokens:,} "
                f"({self.prompt_tokens:,} prompt + {self.completion_tokens:,} completion)"
            )
        else:
            lines.append("token รวม          : ยังไม่มีข้อมูล (server ไม่ได้ส่ง usage มา)")

        stats = context.stats()
        if max_context and self.last_prompt_tokens is not None:
            percent = self.last_prompt_tokens / max_context * 100
            lines.append(
                f"context ปัจจุบัน    : {self.last_prompt_tokens:,} / {max_context:,} token "
                f"({percent:.2f}%)"
            )
        lines.append(
            f"ข้อความใน context  : {stats['messages']} / {context.max_messages} "
            f"(ตัดไปแล้ว {stats['trims']} ครั้ง)"
        )

        return lines

    def mark_ok(self):
        self.requests_ok += 1

    def mark_failed(self):
        self.requests_failed += 1


# ============================================
# ERROR HANDLING
# ============================================


def friendly_error_message(exc, config):
    """
    แปลง exception ของ openai SDK เป็นข้อความไทยที่บอกทางแก้

    คืนค่า (ข้อความ, level) โดย level เป็น "recoverable" หรือ "fatal"
    ข้อความเขียนให้ใช้ได้ทั้ง CLI และเว็บ จึงไม่อ้างถึงปุ่มหรือคำสั่งเฉพาะหน้าจอ
    """
    # APITimeoutError สืบทอดจาก APIConnectionError จึงต้องตรวจก่อน
    if isinstance(exc, APITimeoutError):
        return (
            f"โมเดลไม่ตอบภายใน {config.request_timeout:.0f} วินาที\n"
            "  ถ้าโมเดลยังโหลดน้ำหนักอยู่ ให้ลองใหม่อีกครั้ง\n"
            "  ถ้าเกิดบ่อยให้เพิ่ม LLM_REQUEST_TIMEOUT ใน .env",
            "recoverable",
        )

    if isinstance(exc, APIConnectionError):
        return (
            f"เชื่อมต่อ {config.base_url} ไม่ได้\n"
            "  ตรวจสอบว่าต่อ Tailscale อยู่ และเครื่อง DGX Spark เปิดทำงานอยู่\n"
            "  ถ้า IP หรือ port เปลี่ยนไป ให้แก้ LLM_BASE_URL ใน .env",
            "recoverable",
        )

    if isinstance(exc, AuthenticationError):
        return (
            "server ปฏิเสธ API key (401)\n"
            "  SGLang ปกติไม่ต้องใช้ key — ลองตั้ง LLM_API_KEY=dummy ใน .env\n"
            "  ถ้า server รันด้วย --api-key ให้ใส่ค่านั้นใน .env",
            "fatal",
        )

    if isinstance(exc, NotFoundError):
        return (
            f'ไม่พบ model "{config.model}" บน server (404)\n'
            "  ชื่อ model ต้องตรงกับ --served-model-name ของ SGLang\n"
            "  แก้ LLM_MODEL ใน .env แล้วลองใหม่อีกครั้ง",
            "recoverable",
        )

    if isinstance(exc, RateLimitError):
        return (
            "server จำกัดอัตราการเรียกใช้งาน (429)\n"
            "  รอสักครู่แล้วลองใหม่อีกครั้ง",
            "recoverable",
        )

    if isinstance(exc, BadRequestError):
        return (
            "server ปฏิเสธคำขอ (400)\n"
            "  อาจเกิดจากประวัติการสนทนายาวเกินความจุของโมเดล\n"
            "  ล้างบทสนทนาแล้วเริ่มใหม่ จะช่วยได้ในกรณีนี้",
            "recoverable",
        )

    if isinstance(exc, PermissionDeniedError):
        return ("ไม่มีสิทธิ์เรียก API (403)", "recoverable")

    if isinstance(exc, InternalServerError):
        return ("server เกิดข้อผิดพลาดภายใน (5xx) — ลองใหม่อีกครั้ง", "recoverable")

    if isinstance(exc, APIStatusError):
        request_id = getattr(exc, "request_id", None)
        detail = f"\n  request id: {request_id}" if request_id else ""
        return (f"server ตอบกลับด้วยสถานะ {exc.status_code}{detail}", "recoverable")

    if isinstance(exc, APIError):
        return (f"เกิดข้อผิดพลาดจาก API: {exc}", "recoverable")

    return (f"เกิดข้อผิดพลาดที่ไม่คาดคิด ({type(exc).__name__}): {exc}", "recoverable")


# ============================================
# LLM SESSION
# ============================================


class LLMSession:
    """ครอบการเรียก API ของ SGLang ผ่าน openai SDK"""

    def __init__(self, config):
        self.config = config
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.request_timeout,
            max_retries=config.max_retries,
        )
        # ความจุสูงสุดของ context (token) ของ model ที่ใช้อยู่ — เติมค่าตอนตรวจ endpoint
        self.max_context = None

    def models_info(self):
        """
        ขอรายชื่อ model ที่ server ให้บริการ

        คืนค่า dict {ชื่อ model: max_model_len หรือ None}
        max_model_len ไม่ใช่ฟิลด์มาตรฐานของ SDK จึงต้องอ่านผ่าน model_extra
        """
        info = {}
        for item in self.client.models.list().data:
            info[item.id] = (item.model_extra or {}).get("max_model_len")
        return info

    def stream_reply(self, messages):
        """
        ส่งบทสนทนาไปให้โมเดลและคืน generator ที่ให้ผลลัพธ์ทีละส่วน

        yield เป็น tuple:
            ("delta", ข้อความ)      ข้อความส่วนที่เพิ่มขึ้นมา
            ("reasoning", ข้อความ)  กระบวนการคิด (มาเฉพาะ reasoning model ที่ไม่ได้ปิด thinking)
            ("usage", usage)        สถิติ token (มาใน chunk สุดท้าย)
            ("finish", เหตุผล)      เหตุผลที่จบ เช่น "stop" หรือ "length"

        หมายเหตุ: SDK ไม่ retry ให้เมื่อการอ่าน stream ล้มเหลวกลางทาง
        เพราะการส่งคำขอซ้ำอาจทำให้ได้ข้อความซ้ำ จึงต้องจัดการที่ผู้เรียก
        """
        request = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "stream": True,
        }

        if self.config.include_usage:
            # SGLang ไม่ส่ง usage มาให้ตอน stream ถ้าไม่ขอ
            request["stream_options"] = {"include_usage": True}

        if self.config.max_completion_tokens is not None:
            request["max_completion_tokens"] = self.config.max_completion_tokens

        if self.config.disable_thinking:
            # Qwen3.8 เป็น reasoning model จะแยกกระบวนการคิดไปที่ reasoning_content
            # ถ้าไม่ปิด โมเดลอาจใช้ token หมดไปกับการคิดจนไม่เหลือคำตอบให้ผู้ใช้
            request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        stream = self.client.chat.completions.create(**request)
        try:
            for chunk in stream:
                # chunk สุดท้ายที่ส่ง usage มาอาจมี choices เป็น list ว่าง
                if not chunk.choices:
                    if chunk.usage is not None:
                        yield ("usage", chunk.usage)
                    continue

                choice = chunk.choices[0]

                # reasoning_content ไม่ใช่ฟิลด์มาตรฐานของ SDK จึงอ่านผ่าน model_extra
                reasoning = (choice.delta.model_extra or {}).get("reasoning_content")
                if reasoning:
                    yield ("reasoning", reasoning)

                text = choice.delta.content or ""
                if text:
                    yield ("delta", text)
                if choice.finish_reason:
                    yield ("finish", choice.finish_reason)
        finally:
            stream.close()

    def close(self):
        self.client.close()


def probe_endpoint(session, config):
    """
    ตรวจการเชื่อมต่อและรายชื่อ model แบบ best-effort

    ไม่ raise error และไม่บล็อกการทำงาน เพราะบาง server ไม่ได้เปิด /v1/models
    คืนค่า dict:
        ok            True ถ้าเชื่อมต่อได้และเจอ model ที่ต้องการ
        connected     True ถ้าเรียก /v1/models สำเร็จ
        problem       None | "connection" | "model_missing"
        available     รายชื่อ model ที่ server มี
        max_context   ความจุ context ของ model ที่ใช้ (None ถ้าไม่ทราบ)
        error         ข้อความจาก exception (ถ้ามี)
    """
    result = {
        "ok": False,
        "connected": False,
        "problem": None,
        "available": [],
        "max_context": None,
        "error": None,
    }

    try:
        info = session.models_info()
    except APIError as exc:
        result["problem"] = "connection"
        result["error"] = str(exc)
        return result

    result["connected"] = True
    result["available"] = list(info)

    # จำความจุ context ไว้แสดงผลว่าใกล้เต็มหน้าต่างของโมเดลหรือยัง
    session.max_context = info.get(config.model)
    result["max_context"] = session.max_context

    if config.model in info:
        result["ok"] = True
        return result

    result["problem"] = "model_missing"
    return result
