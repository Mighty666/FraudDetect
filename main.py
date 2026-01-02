import streamlit as st
from PIL import Image
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
import google.generativeai as genai
import io
import zipfile
import os
import tempfile
import json
import re
import threading
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# ========================== CONFIGURATION ==========================
OUTPUT_DIR = "ai_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
COMBINED_JSON_PATH = os.path.join(OUTPUT_DIR, "invoice_fraud_report.json")
SUPPORTED_TYPES = [".pdf", ".xlsx", ".csv", ".jpg", ".jpeg", ".png", ".docx"]
if "history" not in st.session_state:
    st.session_state["history"] = []
results = []

# ========================== GEMINI CONFIG ==========================
genai.configure(api_key="AIzaSyBnUwdBqbD__MXoOm05-0LaRPK7Cp8npBE")
model = genai.GenerativeModel("gemini-2.5-flash")

# ========================== HELPERS ==========================
def extract_docx_text(file):
    return "\n".join([p.text for p in Document(file).paragraphs])

def process_file(file_name, file_data):
    ext = file_name.lower()
    if ext.endswith(".pdf"):
        text = PdfReader(file_data).pages[0].extract_text() or "No text found."
        st.text_area(":page_facing_up: PDF Text", text, height=200)
        return text
    elif ext.endswith(".xlsx"):
        df = pd.read_excel(file_data)
        st.dataframe(df)
        return df
    elif ext.endswith(".csv"):
        df = pd.read_csv(file_data)
        st.dataframe(df)
        return df
    elif ext.endswith((".jpg", ".jpeg", ".png")):
        image = Image.open(file_data)
        image = image.resize((1000, int(1000 * image.height / image.width)))
        st.image(image, caption=":frame_with_picture: Image Preview", use_column_width=True)
        return image
    elif ext.endswith(".docx"):
        text = extract_docx_text(file_data)
        st.text_area(":page_facing_up: DOCX Text", text, height=200)
        return text
    return None

def structured_prompt(filename):
    return (
        f"You are an expert in financial fraud detection. Carefully analyze the provided invoice content.\n\n"
        f"Respond ONLY with a valid JSON object using this exact format:\n\n"
        f"{{\n"
        f"  \"file\": \"{filename}\",\n"
        f"  \"verdict\": \"Legitimate | Suspicious | Fraudulent\",\n"
        f"  \"reason\": \"Explain clearly in one natural sentence why you made this verdict.\",\n"
        f"  \"confidence\": <a number between 1 and 10>\n"
        f"}}\n\n"
        f"Do not include any explanation, comments, or Markdown — only output a single-line valid JSON object."
    )

def try_parse_json(raw_output):
    try:
        clean_output = raw_output.replace("“", '"').replace("”", '"').strip()
        json_match = re.search(r'{.*}', clean_output, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass
    return None

def get_gemini_with_timeout(content, prompt, timeout=30):
    result = {"text": None}
    def task():
        try:
            if isinstance(content, Image.Image):
                result["text"] = model.generate_content([content, prompt]).text
            elif isinstance(content, pd.DataFrame):
                result["text"] = model.generate_content(f"{prompt}\n\n{content.to_markdown(index=False)}").text
            elif isinstance(content, str):
                result["text"] = model.generate_content(f"{prompt}\n\n{content}").text
        except Exception as e:
            result["text"] = f"[ERROR] {str(e)}"
    thread = threading.Thread(target=task)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return "[ERROR] Gemini Vision timed out after 30 seconds"
    return result["text"]

def save_combined_json(data):
    with open(COMBINED_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def log_history(user, file, verdict, confidence):
    st.session_state["history"].append({
        "user": user,
        "file": file,
        "verdict": verdict,
        "confidence": confidence,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

def normalize_data(data):
    normalized = []
    for entry in data:
        normalized.append({k.lower(): v for k, v in entry.items()})
    return normalized

# ========================== UI CONFIG ==========================
st.set_page_config(page_title=":shield: FraudDetect AI", page_icon=":shield:", layout="wide")
st.title(":shield: FraudDetect AI")
user_id = st.text_input("User ID:", value="guest")
tab_names = ["Analyzer", "Why This Matters", "Trends", "Session History", "How It Works"]
tabs = st.tabs(tab_names)
st.sidebar.title(":clipboard: Tabs")
for name in tab_names:
    st.sidebar.markdown(f"- {name}")

# ========================== TABS ==========================
with tabs[0]:
    st.header(":file_folder: Upload & Analyze Invoices")
    uploaded = st.file_uploader(
        "Upload your invoice files (PDF, CSV, Excel, DOCX, Image, ZIP)",
        type=SUPPORTED_TYPES + ["zip"],
        accept_multiple_files=True,
    )
    if uploaded:
        for file in uploaded:
            fname = file.name
            st.subheader(f":open_file_folder: File: `{fname}`")
            try:
                if fname.lower().endswith(".zip"):
                    with zipfile.ZipFile(file) as z:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            z.extractall(tmpdir)
                            for root, _, files in os.walk(tmpdir):
                                for f in files:
                                    try:
                                        full_path = os.path.join(root, f)
                                        _, ext = os.path.splitext(f)
                                        if ext.lower() in SUPPORTED_TYPES:
                                            st.write(f":package: Inside ZIP: {f}")
                                            with open(full_path, "rb") as f_data:
                                                content = process_file(f, f_data)
                                                if content is not None:
                                                    prompt = structured_prompt(f)
                                                    ai_output = get_gemini_with_timeout(content, prompt)
                                                    parsed = try_parse_json(ai_output)
                                                    if parsed:
                                                        results.append(parsed)
                                                        log_history(user_id, f, parsed.get("verdict", "?"), parsed.get("confidence", "?"))
                                                    else:
                                                        st.warning(f"AI output for {f} was not valid JSON")
                                                        st.code(ai_output, language="json")
                                    except Exception as e:
                                        st.error(f"Failed to process {f}: {e}")
                else:
                    content = process_file(fname, file)
                    if content is not None:
                        prompt = structured_prompt(fname)
                        ai_output = get_gemini_with_timeout(content, prompt)
                        parsed = try_parse_json(ai_output)
                        if parsed:
                            results.append(parsed)
                            log_history(user_id, fname, parsed.get("verdict", "?"), parsed.get("confidence", "?"))
                        else:
                            st.warning(f"AI output for {fname} was not valid JSON")
                            st.code(ai_output, language="json")
            except Exception as e:
                st.error(f"Error: {e}")
        if results:
            save_combined_json(results)
            st.success(f":white_check_mark: Processed {len(results)} files. Combined JSON saved to '{COMBINED_JSON_PATH}'.")

with tabs[1]:
    st.header(":books: Why Invoice Fraud Detection Matters")
    st.markdown("""
### :exclamation: The Problem:
Invoice fraud is one of the most common forms of financial crime.
- Global losses exceed **$4.7 trillion** annually.
- Businesses regularly face risks from: ghost vendors, duplicate charges, forged documents.
### :zap: The Impact:
- Fraud drains company revenue and damages reputation.
- Manual reviews are time-consuming and error-prone.
- Small errors can cost companies millions.
### :brain: Why AI Is a Game-Changer:
- AI can process thousands of invoices in seconds.
- Detects hidden inconsistencies and patterns.
- Reduces workload for finance and audit teams.
- Offers objective, repeatable, and explainable analysis.
""")

with tabs[2]:
    st.header(":bar_chart: Trends & Insights")

    # Load data from combined JSON or session history
    combined_data = []
    if os.path.exists(COMBINED_JSON_PATH):
        try:
            with open(COMBINED_JSON_PATH, "r", encoding="utf-8") as f:
                combined_data = json.load(f)
                combined_data = normalize_data(combined_data)
        except Exception as e:
            st.error(f"Error reading combined JSON: {e}")

    if not combined_data and st.session_state["history"]:
        combined_data = normalize_data(st.session_state["history"])

    if not combined_data:
        st.info("No data available to show trends yet.")
    else:
        df = pd.DataFrame(combined_data)

        # Ensure columns exist, fill defaults if not
        for col, default in {
            'confidence': None,
            'verdict': 'unknown',
            'user': 'unknown',
            'time': None,
            'file': 'unknown'
        }.items():
            if col not in df.columns:
                df[col] = default

        # Clean & convert types
        df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df['filetype'] = df['file'].apply(lambda x: os.path.splitext(x)[1].lower() if isinstance(x, str) else 'unknown')

        # Prepare verdict counts for bar chart
        verdict_counts = df['verdict'].value_counts().reset_index()
        verdict_counts.columns = ['verdict', 'count']

        # Prepare filetype counts
        filetype_counts = df['filetype'].value_counts().reset_index()
        filetype_counts.columns = ['filetype', 'count']

        # Prepare confidence by verdict
        confidence_verdict = df.dropna(subset=['confidence'])
        confidence_verdict = confidence_verdict.groupby(['verdict'])['confidence'].describe().reset_index()

        # Start 3x3 grid columns for charts
        cols = st.columns(3)

        # Chart 1: Verdict Breakdown Bar Chart
        with cols[0]:
            st.subheader("Invoice Verdict Breakdown")
            fig1 = px.bar(verdict_counts, x='verdict', y='count', color='verdict', title="Invoice Verdict Breakdown")
            st.plotly_chart(fig1, use_container_width=True)

        # Chart 2: Confidence Score Distribution
        with cols[1]:
            st.subheader("Confidence Scores Frequency")
            if df['confidence'].notna().any():
                fig2 = px.histogram(df, x='confidence', nbins=20, title="Confidence Scores Frequency")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No Confidence data to display.")

        # Chart 3: Uploaded File Types Breakdown
        with cols[2]:
            st.subheader("Uploaded File Types Breakdown")
            fig3 = px.pie(filetype_counts, names='filetype', values='count', title="Uploaded File Types Breakdown")
            st.plotly_chart(fig3, use_container_width=True)

        # Chart 4: Confidence Score Range by Verdict (Box Plot)
        with cols[0]:
            st.subheader("Confidence Score Range by Verdict")
            if df['confidence'].notna().any():
                fig4 = px.box(df, x='verdict', y='confidence', color='verdict', title="Confidence Score Range by Verdict")
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("No Confidence data to display.")

        # Chart 5: AI Confidence Scores Over Time (Line Chart)
        with cols[1]:
            st.subheader("AI Confidence Scores Over Time")
            if df['confidence'].notna().any() and df['time'].notna().any():
                df_time_conf = df.dropna(subset=['confidence', 'time']).sort_values('time')
                fig5 = px.line(df_time_conf, x='time', y='confidence', title="AI Confidence Scores Over Time")
                st.plotly_chart(fig5, use_container_width=True)
            else:
                st.info("Insufficient Time or Confidence data to display.")

        # Chart 6: Invoices Analyzed Over Time (Count per day)
        with cols[2]:
            st.subheader("Invoices Analyzed Over Time")
            if df['time'].notna().any():
                df['date'] = df['time'].dt.date
                counts_by_date = df.groupby('date').size().reset_index(name='count')
                fig6 = px.line(counts_by_date, x='date', y='count', title="Invoices Analyzed Over Time")
                st.plotly_chart(fig6, use_container_width=True)
            else:
                st.info("No Time data to display.")

        # Start new row of 3 cols
        cols2 = st.columns(3)

        # Chart 7: Average AI Confidence per User (Bar Chart)
        with cols2[0]:
            st.subheader("Average AI Confidence per User")
            if df['confidence'].notna().any() and 'user' in df.columns:
                avg_conf_user = df.groupby('user')['confidence'].mean().reset_index()
                fig7 = px.bar(avg_conf_user, x='user', y='confidence', title="Average AI Confidence per User")
                st.plotly_chart(fig7, use_container_width=True)
            else:
                st.info("No Confidence data to display.")

        # Chart 8: Verdict Counts per User (Bar Chart)
        with cols2[1]:
            st.subheader("Verdict Counts per User")
            if 'user' in df.columns and 'verdict' in df.columns:
                verdict_user_counts = df.groupby(['user', 'verdict']).size().reset_index(name='count')
                fig8 = px.bar(verdict_user_counts, x='user', y='count', color='verdict', title="Verdict Counts per User", barmode='group')
                st.plotly_chart(fig8, use_container_width=True)
            else:
                st.info("Insufficient data for Verdict Counts per User.")

        # Chart 9: Confidence vs Time Scatter Plot
        with cols2[2]:
            st.subheader("Confidence vs Time Scatter Plot")
            if df['confidence'].notna().any() and df['time'].notna().any():
                fig9 = px.scatter(df.dropna(subset=['confidence', 'time']), x='time', y='confidence', color='verdict',
                                  title="Confidence vs Time Scatter Plot")
                st.plotly_chart(fig9, use_container_width=True)
            else:
                st.info("Insufficient data for Confidence vs Time plot.")

with tabs[3]:
    st.header(":open_file_folder: Session History")
    if os.path.exists(COMBINED_JSON_PATH):
        with open(COMBINED_JSON_PATH, "r", encoding="utf-8") as f:
            combined_data = json.load(f)
            combined_data = normalize_data(combined_data)
            if combined_data:
                df_summary = pd.DataFrame(combined_data)
                st.dataframe(df_summary)
                export_format = st.radio("Export as:", ["JSON", "CSV"], horizontal=True)
                if export_format == "JSON":
                    st.download_button("Download JSON", json.dumps(combined_data, indent=2), file_name="invoice_fraud_report.json", mime="application/json")
                else:
                    csv_data = df_summary.to_csv(index=False).encode("utf-8")
                    st.download_button("Download CSV", csv_data, file_name="invoice_fraud_report.csv", mime="text/csv")
            else:
                st.info("No data available yet.")
    else:
        st.info("No combined report found.")

with tabs[4]:
    st.header(":brain: How FraudDetect AI Works")
    st.markdown("""
FraudDetect AI leverages the latest in multimodal AI technology from **Gemini 1.5 Vision**.
### :mag: What Happens:
1. :outbox_tray: **Upload:** User uploads invoice in any format.
2. :brain: **Extraction:** Text, tables, or image data is extracted.
3. :receipt: **Prompt Engineering:** Carefully crafted prompt simulates a forensic fraud investigator.
4. :robot_face: **AI Review:** Gemini generates an expert response based on content.
5. :bar_chart: **Logging:** Each analysis is stored with verdict, confidence, and timestamp.
### :shield: Security & Privacy:
- Files are processed **locally**.
- No files are stored after session ends.
- Future versions can support audit logs, encryption, and team dashboards.
""")



