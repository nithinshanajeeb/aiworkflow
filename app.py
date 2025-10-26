# app.py
# Streamlit: Simplified Applicant Intake Form (education/household/housing/financial tabs removed)
# Uses eligibility.py for a simple decision
# Run: streamlit run app.py

import importlib.util
import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Optional

import streamlit as st

# Import the decision logic
from eligibility import decide_eligibility, verify_document_consistency
from db import ApplicationCreate, ApplicationCRUD, DocumentMarkdownCreate
from ollama import OllamaClient

st.set_page_config(page_title="Social Support — Applicant Intake", page_icon="📝", layout="wide")

# ---------- Helpers ----------
EMIRATES = ["Abu Dhabi", "Dubai", "Sharjah", "Ajman", "Umm Al Quwain", "Ras Al Khaimah", "Fujairah"]
NATIONALITIES = ["UAE", "India", "Pakistan", "Bangladesh", "Philippines", "Egypt", "Other"]
EMPLOYMENT_STATUS = [
    "Employed (Full-time)", "Employed (Part-time)", "Self-Employed",
    "Unemployed (seeking work)", "Unemployed (not seeking)", "Student", "Retired"
]
CONTACT_PREF = ["Email", "Phone", "SMS", "WhatsApp"]

# Temporary feature flag
DISABLE_DOCUMENTS = False
UPLOAD_TMP_DIR = Path(tempfile.gettempdir()) / "aiworkflow_uploads"
UPLOAD_TMP_DIR.mkdir(parents=True, exist_ok=True)


def _load_document_parser_module():
    module_path = Path(__file__).with_name("document-parser.py")
    spec = importlib.util.spec_from_file_location("document_parser_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load document-parser.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


document_parser_module = _load_document_parser_module()
DocumentParser = document_parser_module.DocumentParser

ollama_client = OllamaClient(base_url=os.getenv("OLLAMA_BASE_URL"), api_key=os.getenv("OLLAMA_API_KEY"))
if os.getenv("OLLAMA_SKIP_SETUP") != "1":
    try:
        ollama_client.ensure_model()
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Ollama model setup failed: {exc}")

document_parser = DocumentParser(UPLOAD_TMP_DIR, ollama_client=ollama_client)

st.markdown("### 📝 Social Support — Applicant Intake (Simplified)")
st.caption(
    "Prototype form: collects applicant details, income, banking, and required documents,"
    " and persists submissions to PostgreSQL."
)


def _discover_database_url() -> Optional[str]:
    if "DATABASE_URL" in st.secrets:
        return st.secrets["DATABASE_URL"]
    return os.getenv("DATABASE_URL")


@st.cache_resource(show_spinner=False)
def _get_application_crud(dsn: str) -> ApplicationCRUD:
    return ApplicationCRUD(dsn)


DATABASE_URL = _discover_database_url()
CRUD: Optional[ApplicationCRUD] = None
db_init_error: Optional[str] = None

if DATABASE_URL:
    try:
        CRUD = _get_application_crud(DATABASE_URL)
    except Exception as exc:  # noqa: BLE001
        db_init_error = str(exc)

if CRUD is None:
    if db_init_error:
        st.error(f"Unable to initialise database connection: {db_init_error}")
    else:
        st.warning("Database connection not configured; submissions cannot be saved yet.")

with st.form("intake_form", border=True):
    tabs = st.tabs(["Applicant", "Employment & Income", "Banking", "Documents", "Consent & Submit"])

    # --- Tab 1: Applicant ---
    with tabs[0]:
        st.subheader("Applicant Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            full_name = st.text_input("Full Name *", placeholder="e.g., Fatima Al Zaabi")
            dob = st.date_input("Date of Birth *", min_value=date(1900,1,1), max_value=date.today())
            gender = st.selectbox("Gender", ["Female", "Male", "Prefer not to say", "Other"])
        with col2:
            nationality = st.selectbox("Nationality *", NATIONALITIES, index=0)
            emirate = st.selectbox("Emirate of Residence *", EMIRATES, index=1)
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed", "Other"])
        with col3:
            emirates_id_num = st.text_input("Emirates ID Number *", placeholder="784-XXXX-XXXXXXX-X")
            phone = st.text_input("Mobile Number", placeholder="+9715XXXXXXXX")
            email = st.text_input("Email", placeholder="name@example.com")

        st.text_area("Residential Address", placeholder="Building, Street, Area, City")

    # --- Tab 2: Employment & Income ---
    with tabs[1]:
        st.subheader("Employment & Income")
        c1, c2, c3 = st.columns(3)
        with c1:
            employment_status = st.selectbox("Employment Status *", EMPLOYMENT_STATUS)
            employer = st.text_input("Current/Last Employer", placeholder="Company name")
        with c2:
            job_title = st.text_input("Job Title / Role", placeholder="e.g., Sales Associate")
            months_employed = st.number_input("Months in Current/Last Job", min_value=0, value=0, step=1)
        with c3:
            monthly_income_aed = st.number_input("Monthly Income (AED) *", min_value=0, value=0, step=100)

        other_income = st.number_input("Other Monthly Income (AED)", min_value=0, value=0, step=50)
        income_notes = st.text_area("Income Notes", placeholder="Commission, allowances, etc. (optional)")

    # --- Tab 3: Banking ---
    with tabs[2]:
        st.subheader("Banking Summary")
        c1, c2, c3 = st.columns(3)
        with c1:
            bank_name = st.text_input("Primary Bank", placeholder="e.g., Emirates NBD")
        with c2:
            avg_balance_6m = st.number_input("Avg. Balance (Last 6 months, AED)", min_value=0, value=0, step=100)
        with c3:
            credit_score = st.number_input("Credit Score (if known)", min_value=0, max_value=1000, value=0, step=1)

        st.text_area("Banking Notes", placeholder="Overdrafts, returned cheques, etc. (optional)")

    # --- Tab 4: Documents (Uploads only) ---
    with tabs[3]:
        st.subheader("Documents Upload")
        if DISABLE_DOCUMENTS:
            st.info("Document uploads are temporarily disabled.")
            bank_statements = None
        else:
            st.caption("Upload clear scans; PDFs preferred.")
            bank_statements = st.file_uploader(
                "Bank Statements (Last 6 months) *", type=["pdf"], accept_multiple_files=True
            )

    # --- Tab 5: Consent & Submit ---
    with tabs[4]:
        st.subheader("Consent & Contact")
        contact_pref = st.selectbox("Preferred Contact Method", CONTACT_PREF, index=0)
        best_time = st.text_input("Best Time to Contact", placeholder="e.g., Weekdays 9am–6pm")
        additional_info = st.text_area("Additional Information", placeholder="Anything else we should know?")

        consent_data = st.checkbox(
            "I confirm that the provided information is accurate and I consent to its use for eligibility assessment. *"
        )
        consent_documents = st.checkbox(
            "I confirm that I own the rights to share the uploaded documents for verification. *"
        )

        submit = st.form_submit_button("Submit Application", use_container_width=True)

    if submit:
        # Basic required validation
        if DISABLE_DOCUMENTS:
            required_ok = all([
                full_name, dob, nationality, emirate, emirates_id_num,
                employment_status, monthly_income_aed is not None,
                consent_data, consent_documents
            ])
        else:
            required_ok = all([
                full_name, dob, nationality, emirate, emirates_id_num,
                employment_status, monthly_income_aed is not None,
                bank_statements,
                consent_data, consent_documents
            ])

        if not required_ok:
            st.error("⚠️ Please fill all required fields and consents before submitting.")
        else:
            saved_statement_paths = []
            if not DISABLE_DOCUMENTS and bank_statements:
                saved_statement_paths = document_parser.save(bank_statements)
            parsed_documents_markdown = []
            fraud_analysis_markdown = None
            if saved_statement_paths:
                ollama_model_name = getattr(ollama_client, "default_model", "granite3.2-vision")
                for statement_path in saved_statement_paths:
                    try:
                        markdown_pages = document_parser.pdf_to_markdown(
                            statement_path,
                            model_name=ollama_model_name,
                        )
                        parsed_documents_markdown.append(
                            {
                                "source": statement_path,
                                "pages": markdown_pages,
                            }
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.warning(
                            f"Failed to transcribe {Path(statement_path).name} with {ollama_model_name}: {exc}"
                        )
                if parsed_documents_markdown:
                    applicant_profile = {
                        "full_name": full_name,
                        "date_of_birth": dob.isoformat() if isinstance(dob, date) else str(dob),
                        "emirates_id_number": emirates_id_num,
                        "nationality": nationality,
                        "emirate": emirate,
                        "employment_status": employment_status,
                        "employer": employer,
                        "job_title": job_title,
                        "months_employed": months_employed,
                        "monthly_income_aed": monthly_income_aed,
                        "other_income_aed": other_income,
                        "bank_name": bank_name,
                        "average_balance_last_6_months_aed": avg_balance_6m,
                        "credit_score": credit_score,
                    }
                    try:
                        fraud_analysis_markdown = verify_document_consistency(
                            ollama_client,
                            applicant_profile,
                            parsed_documents_markdown,
                            model_name=ollama_model_name,
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"Document verification check failed: {exc}")
                        fraud_analysis_markdown = None

            # --- Decision: use credit_score and monthly_income_aed ---
            cs_val = int(credit_score) if credit_score and credit_score > 0 else None
            inc_val = float(monthly_income_aed) if monthly_income_aed is not None else None
            decision = decide_eligibility(cs_val, inc_val)

            status = decision["status"]
            reasons = "\n".join([f"• {r}" for r in decision.get("reasons", [])])
            saved_record = None
            persistence_error = None

            if status == "approve":
                st.success("✅ Eligible for loan (Approve)")
                if reasons:
                    st.markdown(reasons)
            elif status == "approve_conditional":
                st.info("🟨 Eligible (Conditional): meets minimum thresholds")
                if reasons:
                    st.markdown(reasons)
            elif status == "soft_decline":
                st.error("❌ Not eligible (Soft decline)")
                if reasons:
                    st.markdown(reasons)
            else:  # insufficient
                st.warning("ℹ️ Unable to decide: please provide credit score and monthly income.")

            if fraud_analysis_markdown:
                st.markdown("#### Document Consistency Check")
                st.markdown(fraud_analysis_markdown)

            # Show a compact payload preview
            st.divider()
            st.caption("Submission captured (demo). Below is a structured preview:")
            payload = {
                "full_name": full_name,
                "dob": str(dob),
                "nationality": nationality,
                "emirate": emirate,
                "emirates_id": emirates_id_num,
                "employment_status": employment_status,
                "monthly_income_aed": monthly_income_aed,
                "credit_score": credit_score,
                "ai_veridct": fraud_analysis_markdown,
                "decision": decision,
                "contact_pref": contact_pref,
                "best_time": best_time,
                "notes": additional_info,
                "bank_statement_paths": saved_statement_paths,
                "parsed_documents_markdown": parsed_documents_markdown,
                "fraud_analysis_markdown": fraud_analysis_markdown,
            }
            st.json(payload)

            if CRUD is None:
                st.error("Submission preview only: configure DATABASE_URL to enable persistence.")
            else:
                with st.spinner("Saving application..."):
                    try:
                        db_payload = ApplicationCreate(
                            full_name=full_name,
                            dob=dob,
                            nationality=nationality,
                            emirate=emirate,
                            emirates_id=emirates_id_num,
                            employment_status=employment_status,
                            monthly_income_aed=inc_val,
                            credit_score=cs_val,
                            contact_pref=contact_pref,
                            best_time=best_time or None,
                            notes=additional_info or None,
                            ai_veridct=fraud_analysis_markdown,
                            decision=decision,
                        )
                        saved_record = CRUD.create(db_payload)
                        if saved_record is not None and parsed_documents_markdown:
                            document_entries = []
                            for document in parsed_documents_markdown:
                                source_path = document.get("source")
                                for page_index, content in enumerate(document.get("pages") or [], start=1):
                                    document_entries.append(
                                        DocumentMarkdownCreate(
                                            source_path=source_path,
                                            page_number=page_index,
                                            content_markdown=content or "",
                                        )
                                    )
                            if document_entries:
                                CRUD.create_document_markdowns(saved_record.id, document_entries)
                    except Exception as exc:  # noqa: BLE001
                        persistence_error = str(exc)

                if saved_record is not None:
                    st.success(f"Application stored with ID #{saved_record.id}.")
                elif persistence_error:
                    st.error(f"Failed to save application: {persistence_error}")

st.markdown(
    """
    <style>
    .stTabs [data-baseweb=\"tab-list\"] { gap: 0.5rem; }
    .stTabs [data-baseweb=\"tab\"] { padding: 8px 14px; border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True
)
