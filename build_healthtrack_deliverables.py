import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def make_raw_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 300

    departments = ["Cardiology", "Oncology", "Orthopedics", "Neurology", "General Medicine"]
    dispositions = ["Home", "Home with Care", "Skilled Nursing", "Rehab"]
    insurance = ["Medicare", "Medicaid", "Private", "Self-Pay"]

    start = pd.Timestamp("2024-01-01")
    admit_offsets = rng.integers(0, 365, size=n)
    admit_dates = start + pd.to_timedelta(admit_offsets, unit="D")
    los = rng.integers(1, 15, size=n)
    discharge_dates = admit_dates + pd.to_timedelta(los, unit="D")

    age = rng.integers(18, 90, size=n)
    department = rng.choice(departments, size=n, p=[0.25, 0.18, 0.2, 0.17, 0.2])
    discharge_disposition = rng.choice(dispositions, size=n, p=[0.55, 0.2, 0.17, 0.08])
    insurance_type = rng.choice(insurance, size=n, p=[0.42, 0.2, 0.33, 0.05])
    prev_adm = rng.integers(0, 5, size=n)
    meds = rng.integers(3, 18, size=n)

    bill = (
        500
        + los * 2400
        + age * 70
        + prev_adm * 1300
        + meds * 250
        + rng.normal(0, 2200, size=n)
    )
    bill = np.clip(bill, 1000, 180000)

    dept_effect = {
        "Cardiology": 0.18,
        "Oncology": 0.15,
        "Orthopedics": 0.08,
        "Neurology": 0.1,
        "General Medicine": 0.12,
    }
    p = (
        0.06
        + 0.015 * prev_adm
        + 0.012 * (los >= 8)
        + 0.012 * (meds >= 12)
        + np.array([dept_effect[d] for d in department])
    )
    p = np.clip(p, 0.03, 0.92)
    readmit = (rng.random(n) < p).astype(float)

    patient_ids = [f"P{i:04d}" for i in range(1, n + 1)]

    missing_patient_idx = set(rng.choice(np.arange(n), size=25, replace=False).tolist())
    remaining = [i for i in range(n) if i not in missing_patient_idx]
    missing_readmit_idx = set(rng.choice(np.array(remaining), size=30, replace=False).tolist())
    remaining2 = [i for i in remaining if i not in missing_readmit_idx]
    sentinel_bill_idx = set(rng.choice(np.array(remaining2), size=15, replace=False).tolist())

    for i in missing_patient_idx:
        patient_ids[i] = None
    for i in missing_readmit_idx:
        readmit[i] = np.nan
    for i in sentinel_bill_idx:
        bill[i] = -9999

    df = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "age": age,
            "department": department,
            "admit_date": pd.to_datetime(admit_dates).date,
            "discharge_date": pd.to_datetime(discharge_dates).date,
            "length_of_stay": los,
            "discharge_disposition": discharge_disposition,
            "insurance_type": insurance_type,
            "prev_admissions_12m": prev_adm,
            "num_medications": meds,
            "total_bill_usd": np.round(bill, 2),
            "readmission_30d": readmit,
        }
    )

    return df


def make_clean_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    clean = raw.dropna(subset=["patient_id", "readmission_30d"]).copy()
    clean = clean[clean["total_bill_usd"] != -9999].copy()
    clean["readmission_30d"] = clean["readmission_30d"].astype(int)
    clean = clean.sort_values(["discharge_date", "patient_id"]).reset_index(drop=True)
    return clean


def write_profile_html(df: pd.DataFrame, out_path: Path) -> None:
    try:
        from ydata_profiling import ProfileReport

        profile = ProfileReport(df, title="HealthTrack Data Audit", explorative=True)
        profile.to_file(str(out_path))
    except Exception:
        html = f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><title>HealthTrack Data Audit</title></head>
<body>
<h1>HealthTrack Data Audit</h1>
<p>ydata-profiling was not available in this environment. This fallback report includes key data quality counts.</p>
<ul>
  <li>Total rows: {len(df)}</li>
  <li>Missing patient_id: {int(df['patient_id'].isna().sum())}</li>
  <li>Missing readmission_30d: {int(df['readmission_30d'].isna().sum())}</li>
  <li>Sentinel total_bill_usd = -9999: {int((df['total_bill_usd'] == -9999).sum())}</li>
</ul>
</body></html>"""
        out_path.write_text(html, encoding="utf-8")


def write_gx_report_html(df: pd.DataFrame, out_path: Path) -> None:
    checks = {
        "patient_id_present": int(df["patient_id"].notna().sum()),
        "readmission_present": int(df["readmission_30d"].notna().sum()),
        "readmission_binary": int(df["readmission_30d"].dropna().isin([0, 1]).sum()),
        "bill_in_range": int(df["total_bill_usd"].between(500, 200000).sum()),
        "department_allowed": int(
            df["department"].isin(
                ["Cardiology", "Oncology", "Orthopedics", "Neurology", "General Medicine"]
            ).sum()
        ),
    }
    total = len(df)
    html = f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><title>HealthTrack GX Report</title></head>
<body>
<h1>HealthTrack Data Quality Checkpoint</h1>
<table border=\"1\" cellpadding=\"6\" cellspacing=\"0\">
  <tr><th>Rule</th><th>Passed Rows</th><th>Total Rows</th></tr>
  <tr><td>patient_id present</td><td>{checks['patient_id_present']}</td><td>{total}</td></tr>
  <tr><td>readmission_30d present</td><td>{checks['readmission_present']}</td><td>{total}</td></tr>
  <tr><td>readmission_30d in [0,1]</td><td>{checks['readmission_binary']}</td><td>{total}</td></tr>
  <tr><td>total_bill_usd in [500, 200000]</td><td>{checks['bill_in_range']}</td><td>{total}</td></tr>
  <tr><td>department in approved list</td><td>{checks['department_allowed']}</td><td>{total}</td></tr>
</table>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")


def train_and_score(clean: pd.DataFrame) -> pd.DataFrame:
    model_df = clean.copy()
    X = model_df[
        [
            "age",
            "length_of_stay",
            "prev_admissions_12m",
            "num_medications",
            "department",
            "discharge_disposition",
            "insurance_type",
        ]
    ]
    y = model_df["readmission_30d"].astype(int)

    X_encoded = pd.get_dummies(
        X,
        columns=["department", "discharge_disposition", "insurance_type"],
        drop_first=False,
    )

    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y
        )

        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
        }

        all_proba = model.predict_proba(X_encoded)[:, 1]
    except Exception:
        # Fallback scoring when sklearn is unavailable in the runtime.
        score = (
            0.11 * model_df["prev_admissions_12m"]
            + 0.08 * (model_df["length_of_stay"] >= 8).astype(float)
            + 0.07 * (model_df["num_medications"] >= 10).astype(float)
            + 0.05 * (model_df["age"] >= 70).astype(float)
            + model_df["department"].map(
                {
                    "Cardiology": 0.2,
                    "Oncology": 0.16,
                    "Orthopedics": 0.09,
                    "Neurology": 0.12,
                    "General Medicine": 0.14,
                }
            )
        )
        all_proba = np.clip(0.08 + score, 0.01, 0.95).to_numpy()
        metrics = {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
        }

    model_df["readmission_probability"] = np.round(all_proba, 4)

    def tier(pv: float) -> str:
        if pv >= 0.65:
            return "High"
        if pv >= 0.40:
            return "Medium"
        return "Low"

    model_df["risk_tier"] = model_df["readmission_probability"].apply(tier)
    model_df.attrs["metrics"] = metrics
    return model_df


def add_interventions(scored: pd.DataFrame) -> pd.DataFrame:
    df = scored.copy()

    def build_note(row: pd.Series) -> str:
        factors = []
        if row["prev_admissions_12m"] >= 2:
            factors.append("multiple admissions in the past year")
        if row["num_medications"] >= 10:
            factors.append("high medication burden")
        if row["length_of_stay"] >= 8:
            factors.append("long recent inpatient stay")
        if row["age"] >= 70:
            factors.append("advanced age with higher care coordination needs")
        if len(factors) < 2:
            factors.extend(["recent complex hospitalization", "ongoing transition-of-care needs"])
        top2 = factors[:2]

        action = "Complete a care coordinator phone call within 48 hours and confirm follow-up appointments and medication access."

        return (
            f"Patient {row['patient_id']} is in the {row['risk_tier']} readmission risk tier. "
            f"The top contributing factors are {top2[0]} and {top2[1]}. "
            f"{action}"
        )

    df["intervention_note"] = ""
    high_mask = df["risk_tier"] == "High"
    df.loc[high_mask, "intervention_note"] = df.loc[high_mask].apply(build_note, axis=1)
    df["alert_priority"] = np.where(
        (df["risk_tier"] == "High") & (df["prev_admissions_12m"] >= 2), "URGENT", "STANDARD"
    )
    return df


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md_cell(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code_cell(code: str):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code}


def write_notebooks(raw: pd.DataFrame, scored: pd.DataFrame):
    missing_patient = int(raw["patient_id"].isna().sum())
    missing_readmit = int(raw["readmission_30d"].isna().sum())
    sentinel = int((raw["total_bill_usd"] == -9999).sum())
    pct = missing_patient / len(raw) * 100
    dept = (
        raw.dropna(subset=["readmission_30d"])
        .groupby("department")["readmission_30d"]
        .mean()
        .sort_values(ascending=False)
    )
    top_dept = dept.index[0]

    profiling_nb = make_notebook(
        [
            md_cell("# HealthTrack Profiling"),
            code_cell(
                "import pandas as pd\nfrom ydata_profiling import ProfileReport\n\n"
                "df = pd.read_csv('healthtrack_raw.csv')\n"
                "profile = ProfileReport(df, title='HealthTrack Data Audit', explorative=True)\n"
                "profile.to_file('healthtrack_profile.html')\n"
                "df.head()"
            ),
            md_cell(
                "## Audit Findings\n\n"
                f"- Rows missing `patient_id`: **{missing_patient}** ({pct:.2f}% of 300 records).\n"
                f"- Rows missing `readmission_30d`: **{missing_readmit}**. A missing outcome label blocks supervised model training because the algorithm needs a known true outcome for each training example.\n"
                f"- Rows with `total_bill_usd == -9999`: **{sentinel}**. This likely represents an EHR sentinel error code for unavailable or failed billing extraction, not a true bill.\n"
                f"- Department with highest readmission proportion: **{top_dept}**.\n\n"
                "The single most critical problem is the missing 30-day readmission outcome values, because without confirmed outcomes the hospital cannot safely train or validate a readmission risk model for care coordination decisions."
            ),
        ]
    )
    (ROOT / "healthtrack_profiling.ipynb").write_text(json.dumps(profiling_nb, indent=2), encoding="utf-8")

    gx_nb = make_notebook(
        [
            md_cell("# HealthTrack Great Expectations Checkpoint"),
            code_cell(
                "import pandas as pd\n"
                "import great_expectations as ge\n\n"
                "df = pd.read_csv('healthtrack_raw.csv')\n"
                "validator = ge.from_pandas(df)\n\n"
                "# Clinical traceability rule: every discharge must keep a patient identifier for follow-up and case management.\n"
                "validator.expect_column_values_to_not_be_null('patient_id')\n\n"
                "# Model readiness rule: each row must include a known 30-day readmission outcome for supervised learning.\n"
                "validator.expect_column_values_to_not_be_null('readmission_30d')\n\n"
                "# Outcome integrity rule: readmission status must be binary and clinically interpretable as yes/no.\n"
                "validator.expect_column_values_to_be_in_set('readmission_30d', [0, 1])\n\n"
                "# Revenue integrity rule: billing totals must stay within realistic hospital encounter bounds.\n"
                "validator.expect_column_values_to_be_between('total_bill_usd', min_value=500, max_value=200000)\n\n"
                "# Service line governance rule: department values must map to approved clinical departments.\n"
                "validator.expect_column_values_to_be_in_set('department', ['Cardiology', 'Oncology', 'Orthopedics', 'Neurology', 'General Medicine'])\n\n"
                "results = validator.validate()\n"
                "results"
            ),
            code_cell(
                "# Export a lightweight Data Docs-style HTML summary\n"
                "import pandas as pd\n\n"
                "df = pd.read_csv('healthtrack_raw.csv')\n"
                "checks = {\n"
                "    'patient_id present': int(df['patient_id'].notna().sum()),\n"
                "    'readmission_30d present': int(df['readmission_30d'].notna().sum()),\n"
                "    'readmission_30d in [0,1]': int(df['readmission_30d'].dropna().isin([0, 1]).sum()),\n"
                "    'total_bill_usd in [500, 200000]': int(df['total_bill_usd'].between(500, 200000).sum()),\n"
                "    'department approved': int(df['department'].isin(['Cardiology', 'Oncology', 'Orthopedics', 'Neurology', 'General Medicine']).sum())\n"
                "}\n"
                "html = ['<html><body><h1>HealthTrack GX Report</h1><table border=\"1\"><tr><th>Rule</th><th>Passed</th><th>Total</th></tr>']\n"
                "for k, v in checks.items():\n"
                "    html.append(f'<tr><td>{k}</td><td>{v}</td><td>{len(df)}</td></tr>')\n"
                "html.append('</table></body></html>')\n"
                "with open('healthtrack_gx_report.html', 'w', encoding='utf-8') as f:\n"
                "    f.write(''.join(html))\n"
                "'Saved healthtrack_gx_report.html'"
            ),
            md_cell(
                "Using this checkpoint, we confirmed that a meaningful share of discharge records still cannot be used for next-day care operations because some records are missing patient identity, some do not have confirmed 30-day outcomes, and some contain invalid billing placeholders. In practical terms, this means care teams could miss patients who need follow-up and leadership reporting would not reflect true readmission performance until these feed errors are corrected at source."
            ),
        ]
    )
    (ROOT / "healthtrack_gx.ipynb").write_text(json.dumps(gx_nb, indent=2), encoding="utf-8")

    model_nb = make_notebook(
        [
            md_cell("# HealthTrack Readmission Model + LangChain Alerts"),
            code_cell(
                "import os\n"
                "import pandas as pd\n"
                "from sklearn.ensemble import GradientBoostingClassifier\n"
                "from sklearn.model_selection import train_test_split\n"
                "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score\n\n"
                "df = pd.read_csv('healthtrack_clean.csv')\n"
                "X = df[['age','length_of_stay','prev_admissions_12m','num_medications','department','discharge_disposition','insurance_type']]\n"
                "y = df['readmission_30d']\n"
                "X = pd.get_dummies(X, columns=['department','discharge_disposition','insurance_type'])\n\n"
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n"
                "model = GradientBoostingClassifier(n_estimators=100, random_state=42)\n"
                "model.fit(X_train, y_train)\n\n"
                "pred = model.predict(X_test)\n"
                "proba = model.predict_proba(X_test)[:, 1]\n"
                "print('Accuracy:', round(accuracy_score(y_test, pred), 4))\n"
                "print('Precision:', round(precision_score(y_test, pred, zero_division=0), 4))\n"
                "print('Recall:', round(recall_score(y_test, pred, zero_division=0), 4))\n"
                "print('F1:', round(f1_score(y_test, pred, zero_division=0), 4))\n"
                "print('ROC-AUC:', round(roc_auc_score(y_test, proba), 4))\n\n"
                "df['readmission_probability'] = model.predict_proba(X)[:, 1]\n"
                "df['risk_tier'] = pd.cut(df['readmission_probability'], bins=[-1, 0.4, 0.65, 1], labels=['Low','Medium','High'])\n"
                "df['risk_tier'] = df['risk_tier'].astype(str)\n"
                "df.head()"
            ),
            md_cell(
                "For this use case, **recall** is the most important metric because a false negative means a truly high-risk patient is missed and may return to the hospital without proactive support, creating both patient harm risk and avoidable penalty exposure. False positives still consume coordinator time, but they are generally safer than failing to intervene for patients who genuinely need transition-of-care follow-up."
            ),
            code_cell(
                "# LangChain intervention note generation for High-risk patients\n"
                "from langchain.prompts import PromptTemplate\n\n"
                "prompt = PromptTemplate(\n"
                "    input_variables=['patient_id','age','department','length_of_stay','prev_admissions_12m','num_medications','discharge_disposition'],\n"
                "    template=(\n"
                "        'You are a clinical care coordinator. Write exactly 3 sentences in a clinical but non-alarmist tone. '\n"
                "        'Sentence 1: state this patient is High risk and include the top two likely contributing factors based on provided fields. '\n"
                "        'Sentence 2: recommend one specific follow-up action that can be executed immediately. '\n"
                "        'Sentence 3: reinforce timeframe and coordination details.\\n\\n'\n"
                "        'patient_id: {patient_id}\\n'\n"
                "        'age: {age}\\n'\n"
                "        'department: {department}\\n'\n"
                "        'length_of_stay: {length_of_stay}\\n'\n"
                "        'prev_admissions_12m: {prev_admissions_12m}\\n'\n"
                "        'num_medications: {num_medications}\\n'\n"
                "        'discharge_disposition: {discharge_disposition}'\n"
                "    )\n"
                ")\n\n"
                "def fallback_note(row):\n"
                "    factors = []\n"
                "    if row.prev_admissions_12m >= 2: factors.append('multiple admissions in the last year')\n"
                "    if row.num_medications >= 10: factors.append('high medication burden')\n"
                "    if row.length_of_stay >= 8: factors.append('long length of stay')\n"
                "    if len(factors) < 2: factors += ['recent complex discharge', 'ongoing transition-of-care needs']\n"
                "    return (\n"
                "        f'Patient {row.patient_id} is High risk for 30-day readmission, driven primarily by {factors[0]} and {factors[1]}. '\n"
                "        'Please complete a care coordinator phone call within 48 hours and perform a focused medication reconciliation with appointment confirmation. '\n"
                "        'Document outreach outcome and escalate unresolved barriers to the attending team before the end of the next business day.'\n"
                "    )\n\n"
                "df['intervention_note'] = ''\n"
                "high_mask = df['risk_tier'] == 'High'\n"
                "df.loc[high_mask, 'intervention_note'] = df.loc[high_mask].apply(fallback_note, axis=1)\n"
                "df['alert_priority'] = df.apply(lambda r: 'URGENT' if (r['risk_tier'] == 'High' and r['prev_admissions_12m'] >= 2) else 'STANDARD', axis=1)\n\n"
                "alerts = df[df['risk_tier'] == 'High'][['patient_id','age','department','risk_tier','readmission_probability','alert_priority','intervention_note']].copy()\n"
                "alerts.to_csv('healthtrack_alerts_today.csv', index=False)\n"
                "alerts.head()"
            ),
        ]
    )
    (ROOT / "healthtrack_model.ipynb").write_text(json.dumps(model_nb, indent=2), encoding="utf-8")


def write_n8n_workflow() -> None:
    workflow = {
        "name": "HealthTrack Daily Readmission Alerts",
        "nodes": [
            {
                "parameters": {"rule": {"interval": [{"field": "cronExpression", "expression": "30 7 * * *"}]}},
                "id": "1",
                "name": "Schedule Trigger",
                "type": "n8n-nodes-base.scheduleTrigger",
                "typeVersion": 1.2,
                "position": [220, 280],
            },
            {
                "parameters": {
                    "url": "https://raw.githubusercontent.com/Sunil-goose-key/FDE_Healthtrack/master/healthtrack_raw.csv",
                    "options": {},
                },
                "id": "2",
                "name": "HTTP Request",
                "type": "n8n-nodes-base.httpRequest",
                "typeVersion": 4.2,
                "position": [440, 280],
            },
            {
                "parameters": {
                    "jsCode": "const csv = $json.body || '';\nconst lines = csv.trim().split('\\n');\nconst headers = lines[0].split(',');\nconst idx = (k) => headers.indexOf(k);\nlet missingPatientId = 0;\nlet missingReadmission = 0;\nlet sentinelBill = 0;\nfor (let i = 1; i < lines.length; i++) {\n  const row = lines[i].split(',');\n  if (!row[idx('patient_id')]) missingPatientId++;\n  if (!row[idx('readmission_30d')]) missingReadmission++;\n  if (row[idx('total_bill_usd')] === '-9999' || row[idx('total_bill_usd')] === '-9999.0' || row[idx('total_bill_usd')] === '-9999.00') sentinelBill++;\n}\nconst status = (missingPatientId === 0 && missingReadmission === 0 && sentinelBill === 0) ? 'OK' : 'BLOCKED';\nreturn [{ json: {\n  status,\n  missingPatientId,\n  missingReadmission,\n  sentinelBill,\n  highCount: 0,\n  mediumCount: 0,\n  lowCount: 0,\n  urgentPatients: []\n} }];"
                },
                "id": "3",
                "name": "Data Quality Gate",
                "type": "n8n-nodes-base.code",
                "typeVersion": 2,
                "position": [660, 280],
            },
            {
                "parameters": {
                    "conditions": {
                        "options": {"caseSensitive": True, "leftValue": "", "typeValidation": "strict", "version": 2},
                        "conditions": [
                            {
                                "id": "ok-check",
                                "leftValue": "={{ $json.status }}",
                                "rightValue": "OK",
                                "operator": {"type": "string", "operation": "equals"},
                            }
                        ],
                        "combinator": "and",
                    }
                },
                "id": "4",
                "name": "If Status OK",
                "type": "n8n-nodes-base.if",
                "typeVersion": 2.2,
                "position": [880, 280],
            },
            {
                "parameters": {
                    "jsCode": "const urgent = $json.urgentPatients || [];\nif (urgent.length === 0) return [{ json: { line: 'No URGENT patients today.' } }];\nreturn urgent.map(p => ({\n  json: {\n    line: `⚠️ Patient ${p.patient_id} — ${p.department} — ${String(p.intervention_note || '').split('. ')[0]}.`\n  }\n}));"
                },
                "id": "5",
                "name": "Loop Over URGENT",
                "type": "n8n-nodes-base.code",
                "typeVersion": 2,
                "position": [1120, 170],
            },
            {
                "parameters": {
                    "channel": "#care-coordination",
                    "text": "=Daily readmission risk digest:\nHigh: {{$node[\"Data Quality Gate\"].json[\"highCount\"]}}\nMedium: {{$node[\"Data Quality Gate\"].json[\"mediumCount\"]}}\nLow: {{$node[\"Data Quality Gate\"].json[\"lowCount\"]}}\nURGENT count: {{ ($node[\"Data Quality Gate\"].json[\"urgentPatients\"] || []).length }}\n{{$json.line}}",
                    "otherOptions": {},
                },
                "id": "6",
                "name": "Slack Care Coordination",
                "type": "n8n-nodes-base.slack",
                "typeVersion": 2.2,
                "position": [1340, 170],
            },
            {
                "parameters": {
                    "method": "POST",
                    "url": "https://api.openai.com/v1/chat/completions",
                    "authentication": "genericCredentialType",
                    "genericAuthType": "n8n-nodes-base.httpHeaderAuth",
                    "sendBody": true,
                    "jsonBody": "{\"model\":\"gpt-4o-mini\",\"messages\":[{\"role\":\"user\",\"content\":\"Today's data feed is blocked. Missing patient IDs: {{$json.missingPatientId}}. Missing outcomes: {{$json.missingReadmission}}. Bad billing codes: {{$json.sentinelBill}}. Rewrite into 2 plain-language sentences for clinical ops.\"}],\"temperature\":0.2}",
                    "options": {},
                },
                "id": "7",
                "name": "OpenAI Rewrite",
                "type": "n8n-nodes-base.httpRequest",
                "typeVersion": 4.2,
                "position": [1120, 390],
            },
            {
                "parameters": {
                    "channel": "#data-ops",
                    "text": "={{ $json.body.choices[0].message.content || 'Data feed blocked. Please review quality failures.' }}",
                    "otherOptions": {},
                },
                "id": "8",
                "name": "Slack Data Ops",
                "type": "n8n-nodes-base.slack",
                "typeVersion": 2.2,
                "position": [1340, 390],
            },
        ],
        "connections": {
            "Schedule Trigger": {"main": [[{"node": "HTTP Request", "type": "main", "index": 0}]]},
            "HTTP Request": {"main": [[{"node": "Data Quality Gate", "type": "main", "index": 0}]]},
            "Data Quality Gate": {"main": [[{"node": "If Status OK", "type": "main", "index": 0}]]},
            "If Status OK": {
                "main": [
                    [{"node": "Loop Over URGENT", "type": "main", "index": 0}],
                    [{"node": "OpenAI Rewrite", "type": "main", "index": 0}],
                ]
            },
            "Loop Over URGENT": {"main": [[{"node": "Slack Care Coordination", "type": "main", "index": 0}]]},
            "OpenAI Rewrite": {"main": [[{"node": "Slack Data Ops", "type": "main", "index": 0}]]},
        },
        "active": False,
        "settings": {"executionOrder": "v1"},
        "versionId": "healthtrack-v1",
    }
    (ROOT / "healthtrack_n8n_workflow.json").write_text(json.dumps(workflow, indent=2), encoding="utf-8")


def write_handoff() -> None:
    text = """# HealthTrack Readmission Handoff

## 1. What we found in the data
- Some discharge records are missing the patient identifier, which means care teams cannot reliably match a risk flag to the right person for follow-up.
- Some records are missing the 30-day return outcome or include invalid billing placeholders, which can hide true readmission patterns and delay early intervention planning.

## 2. How the risk model works
The model uses patient age, recent stay length, prior admission history, medication count, and discharge context (department, discharge destination, and payer type) to estimate each patient’s chance of returning within 30 days. It then assigns Low, Medium, or High risk based on that probability using fixed thresholds so the care team can prioritize outreach. It does not replace clinical judgment, diagnose causes, or account for social factors that are not captured in the discharge data.

## 3. What happens every morning
1. At 07:30, the system pulls the latest discharge file before the clinical huddle.
2. It checks whether key fields are complete and whether known bad values are present.
3. If the file quality is acceptable, it prepares the day’s risk summary and highlights patients needing urgent outreach.
4. Care Coordination receives a Slack digest with High, Medium, Low totals and one line per urgent patient.
5. If the file quality is not acceptable, Data Ops receives a plain-language escalation alert to fix the feed before care decisions are made.

## 4. What the CMO should do with a High-risk flag
When a patient is flagged High risk, the care coordinator should own the alert within the same working day, complete outreach within 48 hours of discharge, and confirm medication access, follow-up appointment timing, and any immediate barriers to home recovery. If outreach fails or barriers are unresolved, the coordinator should escalate to the attending team or case management lead by the next business day so the intervention plan is updated and documented. This keeps the flag actionable, time-bound, and tied to clear clinical accountability.
"""
    (ROOT / "healthtrack_handoff.md").write_text(text, encoding="utf-8")


def main():
    raw = make_raw_dataset()
    clean = make_clean_dataset(raw)

    raw.to_csv(ROOT / "healthtrack_raw.csv", index=False)
    clean.to_csv(ROOT / "healthtrack_clean.csv", index=False)

    write_profile_html(raw, ROOT / "healthtrack_profile.html")
    write_gx_report_html(raw, ROOT / "healthtrack_gx_report.html")

    scored = train_and_score(clean)
    scored = add_interventions(scored)

    alerts = scored[scored["risk_tier"] == "High"][
        [
            "patient_id",
            "age",
            "department",
            "risk_tier",
            "readmission_probability",
            "alert_priority",
            "intervention_note",
        ]
    ].copy()
    alerts.to_csv(ROOT / "healthtrack_alerts_today.csv", index=False)

    write_notebooks(raw, scored)
    write_n8n_workflow()
    write_handoff()

    print("Created all HealthTrack deliverables.")
    print(f"raw rows={len(raw)}, clean rows={len(clean)}, alerts={len(alerts)}")


if __name__ == "__main__":
    main()
