# test

โปรเจกต์ Python สำหรับงานวิเคราะห์ข้อมูลพื้นฐาน (Basic Data Analysis) บน Python 3.12

มาพร้อม virtual environment, รายการ dependencies ที่ pin version ไว้ และไฟล์ตัวอย่างสำหรับทดลองรัน

---

## ความต้องการของระบบ (Requirements)

- **Python 3.12** หรือใหม่กว่า ([ดาวน์โหลด](https://www.python.org/downloads/))
- **git** (สำหรับ clone และ push)
- OS: Windows / macOS / Linux

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

---

## โครงสร้างโปรเจกต์

```
test/
├── .env.example          # เทมเพลต environment variables (commit ได้)
├── .gitignore            # ไฟล์/โฟลเดอร์ที่ไม่ต้อง commit
├── requirements.txt      # รายการ dependencies พร้อมเวอร์ชัน
├── game.py               # เกมทายเลข (ตัวอย่างการรับ input)
└── test.py               # ทดสอบอ่านค่าจาก .env
```

> โฟลเดอร์ `.venv/`, ไฟล์ `.env` และ `__pycache__/` จะไม่ถูก commit

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
