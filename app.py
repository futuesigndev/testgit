"""
Web Chatbot (Streamlit) — คุยกับ LLM บน DGX Spark (SGLang)

วิธีใช้:
    streamlit run app.py

การตั้งค่า:
    ใช้ไฟล์ .env ร่วมกับ CLI (chatbot.py) — ดูตัวอย่างได้จาก .env.example
    logic ทั้งหมดอยู่ใน llm_core.py ไฟล์นี้มีแต่ส่วนติดต่อผู้ใช้

หมายเหตุการทำงานของ Streamlit:
    สคริปต์นี้ถูก "รันใหม่ทั้งไฟล์" ทุกครั้งที่ผู้ใช้มีปฏิสัมพันธ์
    สถานะที่ต้องคงอยู่ข้ามการ rerun จึงเก็บไว้ใน st.session_state
    และ object ที่สร้างใหม่แพง (OpenAI client) ใช้ st.cache_resource
"""

from datetime import datetime

import streamlit as st
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

st.set_page_config(page_title="Chatbot — DGX Spark", page_icon="💬", layout="centered")


# ============================================
# BACKEND
# ============================================


@st.cache_resource(show_spinner=False)
def get_backend():
    """
    สร้าง config และ LLMSession ครั้งเดียวต่อโปรเซส

    ถ้าสร้างทุก rerun จะเปลืองทรัพยากรโดยเปล่าประโยชน์
    ผู้ใช้กดปุ่ม "โหลดค่าตั้งใหม่" ใน sidebar เพื่อล้าง cache นี้ได้
    """
    config = load_config()
    return config, LLMSession(config)


def init_state(config, session):
    """เตรียมสถานะของ session นี้ (แยกต่อผู้ใช้แต่ละคน ไม่ปนกัน)"""
    if "context" not in st.session_state:
        st.session_state.context = ConversationContext(
            config.system_prompt, config.max_history_messages
        )
    if "metrics" not in st.session_state:
        st.session_state.metrics = SessionMetrics()
    if "logger" not in st.session_state:
        st.session_state.logger = TranscriptLogger(config.log_dir)
        st.session_state.logger.log_event(
            f"เริ่ม session — endpoint={config.base_url} model={config.model}"
        )
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "endpoint_result" not in st.session_state:
        # ตรวจการเชื่อมต่อครั้งเดียวตอนเปิดหน้า
        st.session_state.endpoint_result = probe_endpoint(session, config)


# ============================================
# STREAMING
# ============================================


def stream_reply(session, context, timing, holder):
    """
    generator สำหรับ st.write_stream

    - แปลง tuple จาก LLMSession.stream_reply() ให้เหลือเฉพาะข้อความ
      เพราะ st.write_stream รับได้เฉพาะ str
    - สะสมข้อความและสถิติลง holder เพื่อให้ผู้เรียกใช้ค่าได้
      แม้เกิด error กลางทาง (กรณีนั้น st.write_stream จะไม่คืนค่าให้)
    """
    stream = session.stream_reply(context.to_payload())
    try:
        for kind, value in stream:
            if kind == "usage":
                holder["usage"] = value
                continue
            if kind == "finish":
                holder["finish_reason"] = value
                continue

            # ชิ้นแรกที่ได้จากโมเดล คือจุดสิ้นสุดของ time-to-first-token
            timing.mark_first_token()

            if kind == "reasoning":
                holder["reasoning"].append(value)
                continue

            holder["text_parts"].append(value)
            yield value
    finally:
        timing.finish()


# ============================================
# RENDERING
# ============================================


def render_message(message):
    """แสดงหนึ่งข้อความจากประวัติ"""
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["error"])
        if message["content"]:
            st.markdown(message["content"])
        for warning in message.get("warnings", []):
            st.warning(warning)
        if message.get("notice"):
            st.info(message["notice"])
        if message.get("reasoning"):
            with st.expander("💭 กระบวนการคิดของโมเดล"):
                st.markdown(message["reasoning"])
        if message.get("stats"):
            st.caption(" · ".join(message["stats"]))


def render_history():
    """แสดงบทสนทนาที่ผ่านมาทั้งหมด"""
    for message in st.session_state.messages:
        render_message(message)


def build_transcript():
    """สร้างข้อความ markdown ของบทสนทนาทั้งหมด สำหรับให้ผู้ใช้ดาวน์โหลด"""
    lines = [
        "# บทสนทนา — Chatbot (DGX Spark)",
        "",
        f"ดาวน์โหลดเมื่อ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
    ]
    for message in st.session_state.messages:
        speaker = "ผู้ใช้" if message["role"] == "user" else "ผู้ช่วย"
        if message["content"]:
            lines.append(f"**{speaker}:** {message['content']}")
        elif message.get("notice"):
            lines.append(f"**{speaker}:** _{message['notice']}_")
        if message.get("stats"):
            lines.append("")
            lines.append(f"> {' · '.join(message['stats'])}")
        lines.append("")
    return "\n".join(lines)


def render_sidebar(config, session, context, metrics, logger):
    """แถบด้านข้าง: ข้อมูลการเชื่อมต่อ สถิติ และปุ่มควบคุม"""
    with st.sidebar:
        st.header("การเชื่อมต่อ")
        st.markdown(f"**Endpoint**  \n`{config.base_url}`")
        st.markdown(f"**Model**  \n`{config.model}`")
        if session.max_context:
            st.markdown(f"**Context สูงสุด**  \n{session.max_context:,} token")
        st.caption(f"ค่าตั้งอ่านจาก `{ENV_FILE}`")

        st.divider()

        st.header("สถิติ session")

        average_ttft = metrics.average_ttft()
        speed = metrics.tokens_per_second()

        first, second = st.columns(2)
        first.metric("TTFT เฉลี่ย", f"{average_ttft:.2f}s" if average_ttft else "—")
        second.metric("ความเร็ว", f"{speed:.1f}" if speed else "—", help="token ต่อวินาที")

        third, fourth = st.columns(2)
        third.metric("token รวม", f"{metrics.total_tokens:,}" if metrics.has_usage else "—")

        used_percent = None
        if session.max_context and metrics.last_prompt_tokens is not None:
            used_percent = metrics.last_prompt_tokens / session.max_context * 100
        fourth.metric("context ใช้ไป", f"{used_percent:.2f}%" if used_percent else "—")

        st.code("\n".join(metrics.describe_session(context, session.max_context)))

        st.divider()

        if st.button("🗑 ล้างบทสนทนา", width="stretch"):
            cleared = context.reset()
            st.session_state.metrics = SessionMetrics()
            st.session_state.messages = []
            logger.log_event(f"ล้างประวัติการสนทนา {cleared} ข้อความ")
            st.rerun()

        st.download_button(
            "⬇️ ดาวน์โหลดบทสนทนา",
            data=build_transcript(),
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            width="stretch",
        )

        if st.button("🔄 โหลดค่าตั้งจาก .env ใหม่", width="stretch"):
            get_backend.clear()
            st.rerun()

        st.caption(f"log ของ session นี้: `{logger.path.name}`")


def render_endpoint_status(result, config):
    """แสดงผลการตรวจการเชื่อมต่อที่ทำไว้ตอนเปิดหน้า"""
    if result["problem"] == "connection":
        st.error(
            f"⚠️ เชื่อมต่อ {config.base_url} ไม่ได้\n\n"
            "ตรวจสอบว่าต่อ **Tailscale** อยู่ และเครื่อง **DGX Spark** เปิดทำงานอยู่  \n"
            f"ถ้า IP หรือ port เปลี่ยนไป ให้แก้ `LLM_BASE_URL` ใน `.env`\n\n"
            f"รายละเอียด: `{result['error']}`"
        )
    elif result["problem"] == "model_missing":
        available = "\n".join(f"- `{name}`" for name in result["available"]) or "_(ไม่มี)_"
        st.warning(
            f'⚠️ ไม่พบ model "{config.model}" บน server\n\n'
            f"model ที่ server มี:\n{available}\n\n"
            "ชื่อ model ต้องตรงกับ `--served-model-name` ของ SGLang — "
            "แก้ `LLM_MODEL` ใน `.env` แล้วกด **โหลดค่าตั้งจาก .env ใหม่**"
        )


# ============================================
# CHAT TURN
# ============================================


def handle_prompt(prompt, config, session, context, logger, metrics):
    """ประมวลผลหนึ่งเทิร์น: แสดงข้อความผู้ใช้ → stream คำตอบ → จัดการ error"""

    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    # ---- เตรียม context ----
    context.add_user(prompt)
    if context.trim():
        st.toast(
            f"ตัดประวัติส่วนเก่าสุดออกให้ไม่เกิน {config.max_history_messages} ข้อความ",
            icon="ℹ️",
        )
    logger.log_user(prompt)

    # ---- เรียกโมเดล ----
    holder = {"text_parts": [], "reasoning": [], "usage": None, "finish_reason": None}
    timing = TurnTiming()
    error_message = None
    fatal = False

    with st.chat_message("assistant"):
        try:
            st.write_stream(stream_reply(session, context, timing, holder))
        except APIError as exc:
            error_message, level = friendly_error_message(exc, config)
            fatal = level == "fatal"
            metrics.mark_failed()
            logger.log_event(f"เกิดข้อผิดพลาด: {type(exc).__name__}")
        except Exception as exc:  # noqa: BLE001 — กันไม่ให้จอขาว
            error_message = f"เกิดข้อผิดพลาดที่ไม่คาดคิด ({type(exc).__name__}): {exc}"
            metrics.mark_failed()
            logger.log_event(f"เกิดข้อผิดพลาด: {type(exc).__name__}")

        timing.finish()
        reply = "".join(holder["text_parts"]).strip()

        if error_message:
            # ข้อความที่ stream ไปแล้วยังคงอยู่บนจอ ผู้ใช้จึงเห็นว่าคำตอบถูกตัดตรงไหน
            st.error(f"❌ {error_message}")

        warnings = []
        if reply and error_message:
            warnings.append("⚠️ คำตอบอาจไม่ครบเนื่องจากเกิดข้อผิดพลาดกลางทาง")
        if reply and holder["finish_reason"] == "length":
            warnings.append("⚠️ คำตอบถูกตัดเพราะยาวเกิน LLM_MAX_COMPLETION_TOKENS")

        if reply:
            context.add_assistant(reply)
            logger.log_assistant(reply)
            metrics.mark_ok()
            metrics.add_usage(holder["usage"])
            metrics.add_timing(timing)

            stats_lines = metrics.describe_turn(holder["usage"], timing, session.max_context)
            reasoning = "".join(holder["reasoning"]) if config.show_reasoning else None

            for warning in warnings:
                st.warning(warning)
            if reasoning:
                with st.expander("💭 กระบวนการคิดของโมเดล"):
                    st.markdown(reasoning)
            st.caption(" · ".join(stats_lines))

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": reply,
                    "stats": stats_lines,
                    "reasoning": reasoning,
                    "warnings": warnings,
                }
            )
        else:
            # ไม่มีคำตอบ → ถอนข้อความผู้ใช้ออก เพื่อไม่ให้ context เพี้ยน
            context.drop_last_user_message()
            notice = (
                "ไม่มีคำตอบ ประวัติการสนทนาจึงไม่ถูกบันทึก "
                f"(ใช้เวลา {timing.elapsed:.2f}s)"
            )
            st.info(f"ℹ️ {notice}")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "notice": notice,
                    "error": error_message,
                    "warnings": [],
                }
            )

    if fatal:
        st.stop()


# ============================================
# MAIN
# ============================================


def main():
    """ประกอบทุกส่วนของหน้าเว็บ"""
    try:
        config, session = get_backend()
    except ConfigError as error:
        lines = describe_config_error(error)
        st.error(f"❌ {lines[0]}")
        st.markdown("**รายการที่ต้องแก้**")
        for problem in error.problems:
            st.markdown(f"- `{problem}`")
        if not error.env_file_found:
            st.info("ไม่พบไฟล์ `.env` — เริ่มต้นด้วยคำสั่ง `Copy-Item .env.example .env`")
        st.stop()

    init_state(config, session)

    context = st.session_state.context
    metrics = st.session_state.metrics
    logger = st.session_state.logger

    st.title("💬 Chatbot — DGX Spark")
    st.caption(
        f"LLM: `{config.model}` · ค่าตั้งจาก `{ENV_FILE}` · "
        "ต้องเปิด Tailscale และเครื่อง DGX Spark ไว้"
    )

    render_endpoint_status(st.session_state.endpoint_result, config)

    if not st.session_state.messages:
        st.info("เริ่มบทสนทนาได้เลย — พิมพ์ข้อความด้านล่าง")

    render_history()

    if prompt := st.chat_input("พิมพ์ข้อความ…"):
        handle_prompt(prompt.strip(), config, session, context, logger, metrics)

    # แสดง sidebar เป็นขั้นสุดท้าย เพื่อให้สถิติสะท้อนเทิร์นที่เพิ่งประมวลผลไปในรอบนี้ด้วย
    render_sidebar(config, session, context, metrics, logger)


main()
