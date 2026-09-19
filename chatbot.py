"""
CLI Chatbot — คุยกับ LLM บน DGX Spark (SGLang)

เชื่อมต่อผ่าน OpenAI-compatible API ของ SGLang ที่รันอยู่บน DGX Spark
และเข้าถึงผ่าน Tailscale จึงต้องเปิด Tailscale ไว้ก่อนใช้งาน

วิธีใช้:
    python chatbot.py

การตั้งค่า:
    ค่าตั้งทั้งหมดของโปรแกรมอยู่ในไฟล์ .env — ไม่มีค่าตั้งต้นฝังอยู่ในไฟล์นี้
    ถ้ายังไม่มี .env ให้คัดลอกจาก .env.example แล้วแก้ค่าตามต้องการ
    ถ้าคีย์ใดหายไป โปรแกรมจะแจ้งว่าขาดอะไรและหยุดทำงานทันที

คำสั่งระหว่างคุย: พิมพ์ /help
"""

import os
import sys
import threading
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
        'แสดงข้อความ "กำลังคิด..." ถ้าโมเดลเงียบนานเกินกี่วินาที',
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

# คำสั่งที่ใช้ได้ระหว่างคุย (เป็นพฤติกรรมของโปรแกรม ไม่ใช่ค่าตั้ง จึงอยู่ในโค้ด)
COMMANDS = {
    "/help": "แสดงคำสั่งทั้งหมด",
    "/reset": "ล้างประวัติการสนทนา (context)",
    "/history": "แสดงจำนวนข้อความและสถิติ token",
    "/model": "แสดง model และ endpoint ที่กำลังใช้",
    "/save": "แสดงที่อยู่ไฟล์ log ของ session นี้",
    "/exit": "ออกจากโปรแกรม",
    "/quit": "เหมือน /exit",
}

THINKING_TEXT = "กำลังคิด..."


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
            problems.append(f"{key} — ค่าที่อ่านได้ \"{raw}\" ใช้ไม่ได้ ({exc})")

    if problems:
        raise ConfigError(problems, env_file_found)

    return Config(values)


def report_config_error(error):
    """แสดงวิธีแก้เมื่อค่าตั้งใน .env ไม่ครบหรือผิด"""
    print("❌ ค่าตั้งใน .env ไม่ถูกต้อง", file=sys.stderr)
    print(file=sys.stderr)

    if not error.env_file_found:
        print(f"   ไม่พบไฟล์ {ENV_FILE}", file=sys.stderr)
        print(f"   เริ่มต้นด้วยคำสั่ง: Copy-Item {ENV_EXAMPLE_FILE} {ENV_FILE}", file=sys.stderr)
        print(file=sys.stderr)

    print("   รายการที่ต้องแก้:", file=sys.stderr)
    for problem in error.problems:
        print(f"     - {problem}", file=sys.stderr)

    print(file=sys.stderr)
    print(f"   ดูตัวอย่างค่าที่ถูกต้องได้จากไฟล์ {ENV_EXAMPLE_FILE}", file=sys.stderr)


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
# TOKEN COUNTING
# ============================================


class TokenCounter:
    """นับ token ที่ใช้ไปทั้ง session"""

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.requests_ok = 0
        self.requests_failed = 0
        self.has_usage = False

    def add_usage(self, usage):
        """สะสมค่าจาก usage ที่ server ส่งกลับ (เป็น None ได้ถ้า server ไม่ส่งมา)"""
        if usage is None:
            return
        self.has_usage = True
        self.prompt_tokens += usage.prompt_tokens or 0
        self.completion_tokens += usage.completion_tokens or 0
        self.total_tokens += usage.total_tokens or 0

    def describe_last(self, usage):
        """ข้อความสรุป token ของคำตอบล่าสุด"""
        if usage is None:
            return "server ไม่ได้ส่งข้อมูล token มา"
        return (
            f"token: {usage.prompt_tokens} prompt + {usage.completion_tokens} completion "
            f"= {usage.total_tokens}"
        )

    def describe_total(self):
        """ข้อความสรุป token สะสมทั้ง session"""
        if not self.has_usage:
            return "ยังไม่มีข้อมูล token (server ไม่ได้ส่ง usage มา)"
        return (
            f"รวม session: {self.total_tokens} token "
            f"({self.prompt_tokens} prompt + {self.completion_tokens} completion) | "
            f"คำขอสำเร็จ {self.requests_ok} · ล้มเหลว {self.requests_failed}"
        )

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
            "  พิมพ์ /model เพื่อดูค่าที่ใช้อยู่ หรือแก้ LLM_MODEL ใน .env",
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
            "  ลองพิมพ์ /reset แล้วเริ่มบทสนทนาใหม่",
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


class ThinkingNotice:
    """
    แสดงข้อความ "กำลังคิด..." ถ้าโมเดลใช้เวลานานเกินกำหนด

    ใช้ Timer เพื่อไม่ให้การรอไปบล็อกการอ่าน stream
    """

    def __init__(self, delay):
        self.delay = delay
        self.shown = False
        self._timer = None

    def start(self):
        self.shown = False
        self._timer = threading.Timer(self.delay, self._show)
        self._timer.daemon = True
        self._timer.start()

    def _show(self):
        # ป้องกันการแสดงซ้ำ เพราะทั้ง Timer และตัวเรียกตรง ๆ ใช้เมธอดนี้ร่วมกัน
        if self.shown:
            return
        self.shown = True
        print(f"⏳ {THINKING_TEXT} ", end="", flush=True)

    def show_now(self):
        """แสดงข้อความแจ้งเตือนทันที ใช้เมื่อรู้ว่าโมเดลกำลังคิดอยู่จริง"""
        self._show()

    def stop(self):
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None


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

    def list_models(self):
        """ขอรายชื่อ model ที่ server ให้บริการ"""
        return [item.id for item in self.client.models.list().data]

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


def check_endpoint(session, config):
    """
    ตรวจการเชื่อมต่อและรายชื่อ model แบบ best-effort

    ไม่บล็อกการทำงาน เพราะบาง server ไม่ได้เปิด endpoint /v1/models
    คืนค่า True ถ้าพร้อมใช้งานเต็มรูปแบบ
    """
    try:
        available = session.list_models()
    except APIError as exc:
        print()
        print(f"⚠️  เชื่อมต่อ {config.base_url} ไม่ได้", file=sys.stderr)
        print("    ตรวจสอบว่าต่อ Tailscale อยู่ และเครื่อง DGX Spark เปิดทำงานอยู่", file=sys.stderr)
        print(f"    รายละเอียด: {exc}", file=sys.stderr)
        print("    จะลองคุยต่อ แต่ถ้าล้มเหลวให้แก้ LLM_BASE_URL ใน .env", file=sys.stderr)
        return False

    if config.model in available:
        print(f'✅ เชื่อมต่อสำเร็จ — model "{config.model}" พร้อมใช้งาน')
        return True

    print()
    print(f'⚠️  ไม่พบ model "{config.model}" บน server', file=sys.stderr)
    if available:
        print("    model ที่ server มี:", file=sys.stderr)
        for name in available:
            print(f"      - {name}", file=sys.stderr)
    else:
        print("    server ไม่ได้แจ้งรายชื่อ model", file=sys.stderr)
    print("    ชื่อ model ต้องตรงกับ --served-model-name ของ SGLang", file=sys.stderr)
    print("    แก้ LLM_MODEL ใน .env แล้วรันใหม่อีกครั้ง", file=sys.stderr)
    return False


# ============================================
# CLI
# ============================================


def handle_command(text, config, context, logger, tokens):
    """
    จัดการคำสั่งที่ขึ้นต้นด้วย "/"

    คืนค่า "message" ถ้าไม่ใช่คำสั่ง · "handled" ถ้าจัดการแล้ว · "exit" ถ้าต้องออก
    """
    if not text.startswith("/"):
        return "message"

    command = text.split()[0].lower()

    if command not in COMMANDS:
        print(f'ไม่รู้จักคำสั่ง "{command}" — พิมพ์ /help เพื่อดูคำสั่งทั้งหมด')
        return "handled"

    if command in ("/exit", "/quit"):
        return "exit"

    if command == "/help":
        print("\nคำสั่งที่ใช้ได้:")
        for name, detail in COMMANDS.items():
            print(f"  {name:<9} {detail}")

    elif command == "/reset":
        count = context.reset()
        print(f"ล้างประวัติการสนทนาแล้ว ({count} ข้อความ)")
        logger.log_event(f"ล้างประวัติการสนทนา {count} ข้อความ")

    elif command == "/history":
        stats = context.stats()
        print("\nประวัติการสนทนา:")
        print(f"  ข้อความใน context  : {stats['messages']} / {config.max_history_messages}")
        print(f"  จำนวนเทิร์น        : {stats['turns']}")
        print(f"  ครั้งที่ตัด context : {stats['trims']}")
        print(f"  {tokens.describe_total()}")

    elif command == "/model":
        print("\nการเชื่อมต่อ:")
        print(f"  Endpoint : {config.base_url}")
        print(f"  Model    : {config.model}")
        print(f"  ค่าตั้ง   : อ่านจาก {ENV_FILE}")

    elif command == "/save":
        print(f"ไฟล์ log ของ session นี้: {logger.path.resolve()}")

    return "handled"


def chat_once(config, session, context, logger, tokens):
    """
    รับข้อความจากผู้ใช้และตอบกลับหนึ่งเทิร์น

    คืนค่า True ถ้าควรคุยต่อ และ False ถ้าควรออกจากโปรแกรม
    """
    try:
        raw = input("\nคุณ: ")
    except (EOFError, KeyboardInterrupt):
        print()
        return False

    text = raw.strip()
    if not text:
        return True

    action = handle_command(text, config, context, logger, tokens)
    if action == "exit":
        return False
    if action == "handled":
        return True

    # ---- เตรียม context ----
    context.add_user(text)
    if context.trim():
        print(f"ℹ️  ตัดประวัติส่วนเก่าสุดออกให้ไม่เกิน {config.max_history_messages} ข้อความ")
    logger.log_user(text)

    # ---- เรียกโมเดล ----
    reply_parts = []
    usage = None
    finish_reason = None
    started = False
    reasoning_started = False
    incomplete = False
    notice = ThinkingNotice(config.thinking_notice_seconds)

    try:
        notice.start()
        try:
            for kind, value in session.stream_reply(context.to_payload()):
                if kind == "usage":
                    usage = value
                    continue
                if kind == "finish":
                    finish_reason = value
                    continue

                if kind == "reasoning":
                    if not config.show_reasoning:
                        # ไม่แสดงกระบวนการคิด แต่บอกผู้ใช้ว่าโมเดลยังทำงานอยู่
                        notice.stop()
                        notice.show_now()
                        continue
                    if not reasoning_started:
                        reasoning_started = True
                        notice.stop()
                        print("💭 ", end="", flush=True)
                    print(value, end="", flush=True)
                    continue

                if not started:
                    started = True
                    if notice.shown or reasoning_started:
                        # มีข้อความแจ้งเตือนหรือกระบวนการคิดอยู่แล้ว ขึ้นบรรทัดใหม่ก่อนตอบ
                        print()
                    print("ผู้ช่วย: ", end="", flush=True)
                print(value, end="", flush=True)
                reply_parts.append(value)
        finally:
            notice.stop()

        if not started:
            print("ผู้ช่วย: (ไม่ได้รับข้อความตอบกลับ)")
        print()

    except KeyboardInterrupt:
        notice.stop()
        incomplete = True
        print("\n⏹  ยกเลิกการตอบ (กด Ctrl+C)")
        tokens.mark_failed()
        logger.log_event("ผู้ใช้ยกเลิกคำตอบกลางทาง")

    except APIError as exc:
        notice.stop()
        incomplete = True
        print()
        message, level = friendly_error_message(exc, config)
        print(f"❌ {message}", file=sys.stderr)
        tokens.mark_failed()
        logger.log_event(f"เกิดข้อผิดพลาด: {type(exc).__name__}")
        if level == "fatal":
            return False

    except Exception as exc:  # noqa: BLE001 — กันไม่ให้ข้อผิดพลาดที่ไม่คาดคิดทำให้โปรแกรมหลุด
        notice.stop()
        incomplete = True
        print()
        print(f"❌ เกิดข้อผิดพลาดที่ไม่คาดคิด ({type(exc).__name__}): {exc}", file=sys.stderr)
        tokens.mark_failed()
        logger.log_event(f"เกิดข้อผิดพลาด: {type(exc).__name__}")

    # ---- เก็บผลลัพธ์ ----
    reply = "".join(reply_parts).strip()

    if reply:
        context.add_assistant(reply)
        logger.log_assistant(reply)
        tokens.mark_ok()
        tokens.add_usage(usage)
        print(f"   [{tokens.describe_last(usage)}]")
        if finish_reason == "length":
            print("   ⚠️  คำตอบถูกตัดเพราะยาวเกิน LLM_MAX_COMPLETION_TOKENS")
        elif incomplete:
            print("   ⚠️  คำตอบอาจไม่ครบเนื่องจากถูกขัดจังหวะกลางทาง")
    else:
        # ไม่มีคำตอบ → ถอนข้อความผู้ใช้ออก เพื่อไม่ให้ context เพี้ยน
        context.drop_last_user_message()
        print("ℹ️  ไม่มีคำตอบ ประวัติการสนทนาจึงไม่ถูกบันทึก")

    return True


def print_banner(config):
    """แสดงข้อมูลต้อนรับตอนเริ่มโปรแกรม"""
    line = "=" * 58
    print(line)
    print("  CLI Chatbot — DGX Spark (SGLang)")
    print(line)
    print(f"  Endpoint : {config.base_url}")
    print(f"  Model    : {config.model}")
    print(f"  Context  : สูงสุด {config.max_history_messages} ข้อความ")
    print(f"  ค่าตั้ง   : {ENV_FILE}")
    print("-" * 58)
    print("  พิมพ์ข้อความเพื่อคุย หรือ /help เพื่อดูคำสั่ง")
    print("  Ctrl+C = ยกเลิกคำตอบ · Ctrl+D = ออก")
    print(line)


def main():
    """จุดเริ่มต้นของโปรแกรม"""
    # ให้ข้อความภาษาไทยแสดงถูกต้องบน console ของ Windows
    try:
        # line_buffering ทำให้ข้อความออกตามลำดับถูกต้อง แม้ output จะถูก redirect ลงไฟล์
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    try:
        config = load_config()
    except ConfigError as error:
        report_config_error(error)
        sys.exit(1)

    if not config.base_url.endswith("/v1"):
        print(f"⚠️  LLM_BASE_URL ไม่ได้ลงท้ายด้วย /v1: {config.base_url}", file=sys.stderr)
        print("    SGLang มักต้องมี /v1 ต่อท้าย เช่น http://100.66.214.124:8888/v1", file=sys.stderr)

    context = ConversationContext(config.system_prompt, config.max_history_messages)
    logger = TranscriptLogger(config.log_dir)
    tokens = TokenCounter()
    session = LLMSession(config)

    print_banner(config)
    logger.log_event(f"เริ่ม session — endpoint={config.base_url} model={config.model}")

    try:
        check_endpoint(session, config)
        while chat_once(config, session, context, logger, tokens):
            pass
    except KeyboardInterrupt:
        print()
    finally:
        session.close()
        print("-" * 58)
        print(tokens.describe_total())
        print(f"บันทึกประวัติไว้ที่: {logger.path.resolve()}")
        print("=" * 58)
        print("ขอบคุณที่ใช้งานครับ")


if __name__ == "__main__":
    main()
