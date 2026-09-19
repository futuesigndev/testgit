# test

โปรเจกต์ Python สำหรับงานวิเคราะห์ข้อมูลพื้นฐาน (Basic Data Analysis) บน Python 3.12

มาพร้อม virtual environment, รายการ dependencies ที่ pin version ไว้ และไฟล์ตัวอย่างสำหรับทดลองรัน

---

## ความต้องการของระบบ (Requirements)

- **Python 3.12** หรือใหม่กว่า ([ดาวน์โหลด](https://www.python.org/downloads/))
- **git** (สำหรับ clone และ push)
- OS: Windows / macOS / Linux
- **Tailscale + เซิร์ฟเวอร์ LLM บน DGX Spark** — เฉพาะกรณีจะรัน `chatbot.py`

---

## การติดตั้ง (Setup)

### 1. Clone โปรเจกต์

```bash
git clone https://github.com/futuesigndev/testgit.git
cd testgit
```

### 2. สร้างและเปิดใช้งาน virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

> เมื่อเปิดใช้งานสำเร็จ จะเห็น `(.venv)` นำหน้าบรรทัดคำสั่ง

### 3. ติดตั้ง dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. ตั้งค่า environment variables

```powershell
Copy-Item .env.example .env      # Windows PowerShell
```

```bash
cp .env.example .env             # macOS / Linux
```

จากนั้นเปิดไฟล์ `.env` แล้วแก้ค่าให้เป็นค่าจริง

> ⚠️ ไฟล์ `.env` ถูกกันไว้ใน `.gitignore` แล้ว — **ห้าม commit** เพราะอาจมีข้อมูลลับ

ตัวแปรที่ `chatbot.py` ใช้ — **ค่าตั้งทั้งหมดอยู่ใน `.env` เท่านั้น** ไม่มีค่าตั้งต้นฝังอยู่ในโค้ด
ถ้าคีย์ใดหายไป `chatbot.py` จะแจ้งว่าขาดอะไรและหยุดทำงานทันที

| ตัวแปร | ตัวอย่างค่า | ความหมาย |
|---|---|---|
| `LLM_BASE_URL` | `http://100.66.214.124:8888/v1` | ที่อยู่ API ของ SGLang (ต้องมี `/v1` ต่อท้าย) |
| `LLM_MODEL` | `qwen3.8-27b-sglang` | ชื่อ model ต้องตรงกับ `--served-model-name` ของ SGLang |
| `LLM_API_KEY` | `dummy` | SGLang ปกติไม่ต้องใช้ API key |
| `LLM_SYSTEM_PROMPT` | — | persona ของผู้ช่วย (system prompt) |
| `LLM_MAX_HISTORY_MESSAGES` | `20` | จำนวนข้อความสูงสุดใน context (system message ไม่นับ) |
| `LLM_TEMPERATURE` | `0.7` | ความสร้างสรรค์ของคำตอบ (0–2) |
| `LLM_DISABLE_THINKING` | `true` | ปิดโหมดคิดก่อนตอบของ reasoning model (ตอบเร็ว ประหยัด token) |
| `LLM_SHOW_REASONING` | `false` | แสดงกระบวนการคิดของโมเดล (ใช้เมื่อไม่ได้ปิด thinking) |
| `LLM_MAX_COMPLETION_TOKENS` | `4096` | token สูงสุดของคำตอบ — เว้นว่างเพื่อใช้ค่า default ของ server |
| `LLM_REQUEST_TIMEOUT` | `180` | timeout ต่อคำขอ (วินาที) |
| `LLM_MAX_RETRIES` | `2` | จำนวนครั้งที่ retry อัตโนมัติเมื่อ connection error / 429 / 5xx |
| `LLM_INCLUDE_USAGE` | `true` | ขอ token usage ตอน stream (SGLang ไม่ส่งมาให้ถ้าไม่ขอ) |
| `LLM_THINKING_NOTICE_SECONDS` | `3` | แสดง "กำลังคิด..." ถ้าโมเดลเงียบนานเกินกี่วินาที |
| `LLM_LOG_DIR` | `logs` | โฟลเดอร์เก็บประวัติการสนทนา |

```powershell
# ตรวจว่าเซิร์ฟเวอร์ LLM เข้าถึงได้ (ต้องเปิด Tailscale ก่อน)
curl.exe -s http://100.66.214.124:8888/v1/models
```

---

## วิธีใช้งาน (Usage)

### รันเกมทายเลข

```bash
python game.py
```

เกมจะสุ่มเลข 1–10 ให้ทาย โดยมีโอกาสทาย 3 ครั้ง พร้อมบอกใบ้ว่ามากกว่า/น้อยกว่า

### รันทดสอบการอ่านค่า `.env`

```bash
python test.py
```

จะพิมพ์ค่าของตัวแปร `USER_NAME` ที่ตั้งไว้ใน `.env` ออกมา

### รัน CLI Chatbot

```bash
python chatbot.py
```

แชทกับ LLM บน DGX Spark ผ่าน CLI — ตอบแบบ streaming ทีละคำ จำบริบทการสนทนาก่อนหน้าได้
และเมื่อ context ยาวเกินกำหนดจะตัดส่วนเก่าสุดออกให้อัตโนมัติ

**สิ่งที่ควรรู้:**

- ต้อง**เปิด Tailscale** และเครื่อง DGX Spark ต้องเปิดทำงานอยู่
- หลังทุกคำตอบจะแสดงสถิติอัตโนมัติ เพื่อดูประสิทธิภาพของ DGX:

  | ตัวเลข | ความหมาย |
  |---|---|
  | **TTFT** | time-to-first-token — เวลาที่รอจนได้คำแรก (สำคัญที่สุดต่อความรู้สึกว่า "เร็ว") |
  | **เวลารวม** | เวลาทั้งหมดของเทิร์นนั้น |
  | **tok/s** | ความเร็วในการสร้างคำตอบ (token ต่อวินาที) |
  | **token** | prompt / completion / รวม ของเทิร์นนั้น |
  | **context** | prompt token ของเทิร์นนั้น เทียบกับความจุสูงสุดของโมเดล (อ่านจาก server) |

- ประวัติการสนทนาถูกบันทึกไว้ที่ `logs/chat_YYYYMMDD_HHMMSS.log` (ไม่ถูก commit)
- `Ctrl+C` = ยกเลิกคำตอบที่กำลังพิมพ์ · `Ctrl+D` = ออก

**คำสั่งระหว่างคุย:**

| คำสั่ง | ความหมาย |
|---|---|
| `/help` | แสดงคำสั่งทั้งหมด |
| `/reset` | ล้างประวัติการสนทนา (context) |
| `/history` | แสดงจำนวนข้อความใน context |
| `/stats` | แสดงสถิติประสิทธิภาพทั้ง session (TTFT เฉลี่ย, ความเร็ว, token, context) |
| `/model` | แสดง model และ endpoint ที่กำลังใช้ |
| `/save` | แสดงที่อยู่ไฟล์ log ของ session นี้ |
| `/exit`, `/quit` | ออกจากโปรแกรม |

### รันเว็บแอป (Streamlit)

```bash
streamlit run app.py
```

แล้วเปิดเบราว์เซอร์ที่ <http://localhost:8501>

เว็บแอปใช้ LLM และค่าตั้งชุดเดียวกับ CLI (ไฟล์ `.env`) และมีฟีเจอร์เทียบเท่า:

- ตอบแบบ streaming ทีละคำในบับเบิลแชท
- จำบริบทการสนทนา และตัดประวัติส่วนเก่าสุดอัตโนมัติเมื่อยาวเกินกำหนด
- **Sidebar** แสดงการเชื่อมต่อ, สถิติ session (TTFT เฉลี่ย, ความเร็ว token/วินาที, token รวม, % ของ context ที่ใช้ไป)
- ปุ่ม **ล้างบทสนทนา**, **ดาวน์โหลดบทสนทนา** (ไฟล์ `.md`), และ **โหลดค่าตั้งจาก .env ใหม่**
- เปิดหลายแท็บพร้อมกันได้ ประวัติแยกกันไม่ปนกัน

> ถ้าอยากให้เครื่องอื่นใน Tailnet เข้าใช้ได้ด้วย:
> `streamlit run app.py --server.address 0.0.0.0`
> (ระวัง: Streamlit ไม่มีระบบ login ในตัว ใครเข้าได้ก็ใช้ LLM ของคุณได้)

**ไฟล์ที่เกี่ยวข้อง**

| ไฟล์ | หน้าที่ |
|---|---|
| `app.py` | ส่วนติดต่อผู้ใช้ของเว็บ |
| `chatbot.py` | ส่วนติดต่อผู้ใช้ของ CLI |
| `llm_core.py` | logic ที่ใช้ร่วมกันทั้งสองแบบ (config, context, การเรียก LLM, error, สถิติ) |

---

## โครงสร้างโปรเจกต์

```
test/
├── .env.example          # เทมเพลต environment variables (commit ได้)
├── .gitignore            # ไฟล์/โฟลเดอร์ที่ไม่ต้อง commit
├── requirements.txt      # รายการ dependencies พร้อมเวอร์ชัน
├── llm_core.py           # logic ที่ CLI และเว็บใช้ร่วมกัน
├── chatbot.py            # CLI chatbot คุยกับ LLM บน DGX Spark
├── app.py                # เว็บแอป (Streamlit) ของ chatbot เดียวกัน
├── game.py               # เกมทายเลข (ตัวอย่างการรับ input)
├── test.py               # ทดสอบอ่านค่าจาก .env
└── logs/                 # ประวัติการสนทนา (ไม่ถูก commit)
```

> โฟลเดอร์ `.venv/`, `logs/`, ไฟล์ `.env` และ `__pycache__/` จะไม่ถูก commit

---

## Dependencies หลัก

| หมวด | แพ็กเกจ |
|---|---|
| จัดการข้อมูล | `numpy`, `pandas` |
| อ่าน/เขียนไฟล์ | `openpyxl` (.xlsx), `xlrd` (.xls), `pyarrow` (Parquet) |
| กราฟ / Visualization | `matplotlib`, `seaborn` |
| สถิติ | `scipy`, `statsmodels` |
| Machine Learning | `scikit-learn` |
| Notebook | `jupyterlab`, `notebook`, `ipykernel`, `ipywidgets` |
| LLM API | `openai` (ใช้คุยกับ SGLang ซึ่งเป็น OpenAI-compatible) |
| Web UI | `streamlit` (เว็บแอปใน `app.py`) |
| อื่น ๆ | `tqdm`, `python-dotenv` |

---

## Git

- Branch หลัก: `main`
- Remote: `origin` → `https://github.com/futuesigndev/testgit.git`

```bash
git add .
git commit -m "ข้อความอธิบายการเปลี่ยนแปลง"
git push
```

### ถ้ามีไฟล์ที่เผลอ commit ไปแล้วแต่ควรถูก ignore

```bash
git rm --cached <ชื่อไฟล์>
```

---

## หมายเหตุ

- ถ้าต้องการใช้ไฟล์ข้อมูลจริง (`.csv`, `.xlsx` ฯลฯ) ร่วมกับทีม ให้เพิ่มบรรทัดยกเว้นใน `.gitignore` เช่น:

  ```
  !data/sample.csv
  ```

### เรื่องที่ควรรู้เกี่ยวกับโมเดล

- **ถ้าภาษาไทยเพี้ยน** (สระ/วรรณยุกต์หาย หรือตัวอักษรสลับกัน) สาเหตุคือ `LLM_TEMPERATURE` สูงเกินไป
  ให้ลดลงเหลือ `0.2`–`0.3` — ค่าเริ่มต้นในไฟล์นี้ตั้งไว้ `0.3` แล้ว
- **`qwen3.8-27b-sglang` เป็น reasoning model** — ถ้าไม่ปิดโหมดคิด โมเดลอาจใช้ token หมดไปกับ
  `reasoning_content` จนไม่เหลือคำตอบให้ผู้ใช้ ตั้ง `LLM_DISABLE_THINKING=true` (ค่าเริ่มต้น) เพื่อปิด
  หรือถ้าอยากดูกระบวนการคิด ให้ตั้ง `LLM_DISABLE_THINKING=false` และ `LLM_SHOW_REASONING=true`
- **ถ้าอยากให้จำบทสนทนายาวขึ้น** — โมเดลนี้รองรับ context 262,144 token แต่ `LLM_MAX_HISTORY_MESSAGES`
  ตั้งไว้แค่ 20 ข้อความ เพิ่มได้อีกมาก (ระวังหน่วยความจำของ DGX Spark ด้วย)
