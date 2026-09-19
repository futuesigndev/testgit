"""
CLI Chatbot — คุยกับ LLM บน DGX Spark (SGLang)

เชื่อมต่อผ่าน OpenAI-compatible API ของ SGLang ที่รันอยู่บน DGX Spark
และเข้าถึงผ่าน Tailscale จึงต้องเปิด Tailscale ไว้ก่อนใช้งาน

วิธีใช้:
    python chatbot.py

การตั้งค่า:
    ค่าตั้งทั้งหมดอยู่ในไฟล์ .env — ดูตัวอย่างได้จาก .env.example
    ถ้าคีย์ใดหายไป โปรแกรมจะแจ้งว่าขาดอะไรและหยุดทำงานทันที

    logic ที่ใช้ร่วมกับเว็บแอปอยู่ใน llm_core.py
    ไฟล์นี้เหลือเฉพาะส่วนที่เกี่ยวกับการแสดงผลบน command line

คำสั่งระหว่างคุย: พิมพ์ /help
"""

import sys
import threading

from openai import APIError

from llm_core import (
    ENV_FILE,
    ConfigError,
    ConversationContext,
    LLMSession,
    SessionMetrics,
    TranscriptLogger,
    TurnTiming,
    describe_config_error,
    friendly_error_message,
    load_config,
    probe_endpoint,
)

# ============================================
# CLI CONFIG
# ============================================
# คำสั่งที่ใช้ได้ระหว่างคุย (เป็นเรื่องของหน้าจอ ไม่ใช่ค่าตั้ง จึงอยู่ในโค้ด)

COMMANDS = {
    "/help": "แสดงคำสั่งทั้งหมด",
    "/reset": "ล้างประวัติการสนทนา (context)",
    "/history": "แสดงจำนวนข้อความใน context",
    "/stats": "แสดงสถิติประสิทธิภาพ (เวลา, token, context)",
    "/model": "แสดง model และ endpoint ที่กำลังใช้",
    "/save": "แสดงที่อยู่ไฟล์ log ของ session นี้",
    "/exit": "ออกจากโปรแกรม",
    "/quit": "เหมือน /exit",
}

THINKING_TEXT = "กำลังคิด..."


# ============================================
# THINKING NOTICE
# ============================================


class ThinkingNotice:
    """
    แสดงข้อความ "กำลังคิด..." ถ้าโมเดลเงียบนานเกินกำหนด (ใช้เฉพาะบน CLI)

    ใช้ Timer เพื่อไม่ให้การรอไปบล็อกการอ่าน stream
    ต้องเรียก stop() ทันทีที่เริ่มได้คำตอบ ไม่งั้นข้อความจะไปโผล่กลางคำตอบ
    """

    def __init__(self, delay):
        self.delay = delay
        self.shown = False
        self._active = False
        self._timer = None

    def start(self):
        self.shown = False
        self._active = True
        self._timer = threading.Timer(self.delay, self._show)
        self._timer.daemon = True
        self._timer.start()

    def _print_notice(self):
        print(f"⏳ {THINKING_TEXT} ", end="", flush=True)

    def _show(self):
        # เรียกโดย Timer — ไม่ทำอะไรถ้าหยุดไปแล้วหรือเคยแสดงไปแล้ว
        if not self._active or self.shown:
            return
        self.shown = True
        self._print_notice()

    def show_now(self):
        """แสดงข้อความแจ้งเตือนทันที ใช้เมื่อรู้ว่าโมเดลกำลังคิดอยู่จริง"""
        if self.shown:
            return
        self.shown = True
        self._print_notice()

    def stop(self):
        self._active = False
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None


# ============================================
# CLI COMMANDS
# ============================================


def handle_command(text, config, session, context, logger, metrics):
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
        print("\nหมายเหตุ: สถิติเวลาและ token จะแสดงอัตโนมัติหลังทุกคำตอบ")

    elif command == "/reset":
        count = context.reset()
        print(f"ล้างประวัติการสนทนาแล้ว ({count} ข้อความ)")
        logger.log_event(f"ล้างประวัติการสนทนา {count} ข้อความ")

    elif command == "/history":
        stats = context.stats()
        print("\nประวัติการสนทนา:")
        print(f"  ข้อความใน context   : {stats['messages']} / {config.max_history_messages}")
        print(f"  จำนวนเทิร์นใน context: {stats['turns']}")
        print(f"  ครั้งที่ตัด context  : {stats['trims']}")

    elif command == "/stats":
        print("\nสถิติการทำงาน (session นี้):")
        for line in metrics.describe_session(context, session.max_context):
            print(f"  {line}")

    elif command == "/model":
        print("\nการเชื่อมต่อ:")
        print(f"  Endpoint : {config.base_url}")
        print(f"  Model    : {config.model}")
        if session.max_context:
            print(f"  Context  : สูงสุด {session.max_context:,} token (อ่านจาก server)")
        print(f"  ค่าตั้ง   : อ่านจาก {ENV_FILE}")

    elif command == "/save":
        print(f"ไฟล์ log ของ session นี้: {logger.path.resolve()}")

    return "handled"


# ============================================
# CHAT LOOP
# ============================================


def chat_once(config, session, context, logger, metrics):
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

    action = handle_command(text, config, session, context, logger, metrics)
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
    timing = TurnTiming()

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

                # ชิ้นแรกที่ได้จากโมเดล คือจุดสิ้นสุดของ time-to-first-token
                timing.mark_first_token()

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
                    # ต้องหยุดตัวจับเวลาทันที ไม่งั้นข้อความ "กำลังคิด..." จะไปโผล่กลางคำตอบ
                    notice.stop()
                    if notice.shown or reasoning_started:
                        # มีข้อความแจ้งเตือนหรือกระบวนการคิดอยู่แล้ว ขึ้นบรรทัดใหม่ก่อนตอบ
                        print()
                    print("ผู้ช่วย: ", end="", flush=True)
                print(value, end="", flush=True)
                reply_parts.append(value)
        finally:
            notice.stop()
            timing.finish()

        if not started:
            print("ผู้ช่วย: (ไม่ได้รับข้อความตอบกลับ)")
        print()

    except KeyboardInterrupt:
        notice.stop()
        incomplete = True
        print("\n⏹  ยกเลิกการตอบ (กด Ctrl+C)")
        metrics.mark_failed()
        logger.log_event("ผู้ใช้ยกเลิกคำตอบกลางทาง")

    except APIError as exc:
        notice.stop()
        incomplete = True
        print()
        message, level = friendly_error_message(exc, config)
        print(f"❌ {message}", file=sys.stderr)
        metrics.mark_failed()
        logger.log_event(f"เกิดข้อผิดพลาด: {type(exc).__name__}")
        if level == "fatal":
            return False

    except Exception as exc:  # noqa: BLE001 — กันไม่ให้ข้อผิดพลาดที่ไม่คาดคิดทำให้โปรแกรมหลุด
        notice.stop()
        incomplete = True
        print()
        print(f"❌ เกิดข้อผิดพลาดที่ไม่คาดคิด ({type(exc).__name__}): {exc}", file=sys.stderr)
        metrics.mark_failed()
        logger.log_event(f"เกิดข้อผิดพลาด: {type(exc).__name__}")

    # ---- เก็บผลลัพธ์ ----
    reply = "".join(reply_parts).strip()
    timing.finish()

    if reply:
        context.add_assistant(reply)
        logger.log_assistant(reply)
        metrics.mark_ok()
        metrics.add_usage(usage)
        metrics.add_timing(timing)
        for line in metrics.describe_turn(usage, timing, session.max_context):
            print(f"   {line}")
        if finish_reason == "length":
            print("   ⚠️  คำตอบถูกตัดเพราะยาวเกิน LLM_MAX_COMPLETION_TOKENS")
        elif incomplete:
            print("   ⚠️  คำตอบอาจไม่ครบเนื่องจากถูกขัดจังหวะกลางทาง")
    else:
        # ไม่มีคำตอบ → ถอนข้อความผู้ใช้ออก เพื่อไม่ให้ context เพี้ยน
        context.drop_last_user_message()
        print(f"ℹ️  ไม่มีคำตอบ ประวัติการสนทนาจึงไม่ถูกบันทึก (ใช้เวลา {timing.elapsed:.2f}s)")

    return True


# ============================================
# DISPLAY HELPERS
# ============================================


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


def print_config_error(error):
    """แสดงวิธีแก้เมื่อค่าตั้งใน .env ไม่ครบหรือผิด (ฉบับ CLI)"""
    lines = describe_config_error(error)
    print(f"❌ {lines[0]}", file=sys.stderr)
    for line in lines[1:]:
        print(f"   {line}" if line else "", file=sys.stderr)


def print_endpoint_report(session, config):
    """
    ตรวจการเชื่อมต่อแล้วพิมพ์ผลออกจอ (ฉบับ CLI)

    คืนค่า True ถ้าพร้อมใช้งานเต็มรูปแบบ
    """
    result = probe_endpoint(session, config)

    if result["problem"] == "connection":
        print()
        print(f"⚠️  เชื่อมต่อ {config.base_url} ไม่ได้", file=sys.stderr)
        print("    ตรวจสอบว่าต่อ Tailscale อยู่ และเครื่อง DGX Spark เปิดทำงานอยู่", file=sys.stderr)
        print(f"    รายละเอียด: {result['error']}", file=sys.stderr)
        print("    จะลองคุยต่อ แต่ถ้าล้มเหลวให้แก้ LLM_BASE_URL ใน .env", file=sys.stderr)
        return False

    if result["problem"] == "model_missing":
        print()
        print(f'⚠️  ไม่พบ model "{config.model}" บน server', file=sys.stderr)
        if result["available"]:
            print("    model ที่ server มี:", file=sys.stderr)
            for name in result["available"]:
                print(f"      - {name}", file=sys.stderr)
        else:
            print("    server ไม่ได้แจ้งรายชื่อ model", file=sys.stderr)
        print("    ชื่อ model ต้องตรงกับ --served-model-name ของ SGLang", file=sys.stderr)
        print("    แก้ LLM_MODEL ใน .env แล้วรันใหม่อีกครั้ง", file=sys.stderr)
        return False

    if result["max_context"]:
        print(
            f'✅ เชื่อมต่อสำเร็จ — model "{config.model}" '
            f"(context สูงสุด {result['max_context']:,} token)"
        )
    else:
        print(f'✅ เชื่อมต่อสำเร็จ — model "{config.model}" พร้อมใช้งาน')
    return True


# ============================================
# MAIN
# ============================================


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
        print_config_error(error)
        sys.exit(1)

    if not config.base_url.endswith("/v1"):
        print(f"⚠️  LLM_BASE_URL ไม่ได้ลงท้ายด้วย /v1: {config.base_url}", file=sys.stderr)
        print("    SGLang มักต้องมี /v1 ต่อท้าย เช่น http://100.66.214.124:8888/v1", file=sys.stderr)

    context = ConversationContext(config.system_prompt, config.max_history_messages)
    logger = TranscriptLogger(config.log_dir)
    metrics = SessionMetrics()
    session = LLMSession(config)

    print_banner(config)
    logger.log_event(f"เริ่ม session — endpoint={config.base_url} model={config.model}")

    try:
        print_endpoint_report(session, config)
        while chat_once(config, session, context, logger, metrics):
            pass
    except KeyboardInterrupt:
        print()
    finally:
        session.close()
        print("-" * 58)
        print("สรุปการทำงาน:")
        for line in metrics.describe_session(context, session.max_context):
            print(f"  {line}")
        print(f"บันทึกประวัติไว้ที่: {logger.path.resolve()}")
        print("=" * 58)
        print("ขอบคุณที่ใช้งานครับ")


if __name__ == "__main__":
    main()
