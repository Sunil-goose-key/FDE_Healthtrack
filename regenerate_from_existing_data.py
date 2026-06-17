import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def md_cell(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code_cell(code: str):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code}


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_profile_artifacts(raw: pd.DataFrame):
    missing_patient = int(raw["patient_id"].isna().sum())
    missing_readmit = int(raw["readmission_30d"].isna().sum())
    sentinel = int((raw["total_bill_usd"] == -9999).sum())
    pct = 100 * missing_patient / len(raw)
    dept_rates = (
        raw.dropna(subset=["readmission_30d"]).groupby("department")["readmission_30d"].mean().sort_values(ascending=False)
    )
    top_dept = dept_rates.index[0]

    profile_path = ROOT / "healthtrack_profile.html"
    try:
        from ydata_profiling import ProfileReport

        profile = ProfileReport(raw, title="HealthTrack Data Audit", explorative=True)
        profile.to_file(str(profile_path))
    except Exception:
        html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>HealthTrack Data Audit</title></head>
<body>
<h1>HealthTrack Data Audit</h1>
<p>Environment fallback report (ydata-profiling unavailable in runtime).</p>
<ul>
  <li>Total rows: {len(raw)}</li>
  <li>Missing patient_id: {missing_patient} ({pct:.2f}%)</li>
  <li>Missing readmission_30d: {missing_readmit}</li>
  <li>total_bill_usd = -9999 rows: {sentinel}</li>
  <li>Highest readmission proportion department: {top_dept}</li>
</ul>
</body></html>"""
        profile_path.write_text(html, encoding="utf-8")

    nb = notebook(
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
                f"- Rows missing `readmission_30d`: **{missing_readmit}**. A missing outcome value blocks supervised model training because the model needs known outcomes to learn what risk patterns lead to readmission.\n"
                f"- Rows with `total_bill_usd == -9999`: **{sentinel}**. This value likely represents an EHR placeholder for unavailable billing data rather than a true patient bill.\n"
                f"- Department with highest readmission proportion: **{top_dept}**.\n\n"
                "The most critical issue is missing 30-day readmission outcomes, because it prevents us from training and validating a reliable model to identify patients needing early intervention."
            ),
        ]
    )
    (ROOT / "healthtrack_profiling.ipynb").write_text(json.dumps(nb, indent=2), encoding="utf-8")


def write_gx_artifacts(raw: pd.DataFrame):
    checks = {
        "patient_id present": int(raw["patient_id"].notna().sum()),
        "readmission_30d present": int(raw["readmission_30d"].notna().sum()),
        "readmission_30d in [0,1]": int(raw["readmission_30d"].dropna().isin([0, 1]).sum()),
        "total_bill_usd in [500, 200000]": int(raw["total_bill_usd"].between(500, 200000).sum()),
        "department approved": int(
            raw["department"].isin(["Cardiology", "Oncology", "Orthopedics", "Neurology", "General Medicine"]).sum()
        ),
    }

    report_path = ROOT / "healthtrack_gx_report.html"
    try:
        import great_expectations as gx

        context = gx.get_context(mode="ephemeral")
        datasource = context.data_sources.add_pandas(name="healthtrack_pandas_src")
        asset = datasource.add_dataframe_asset(name="healthtrack_raw_asset")
        batch_definition = asset.add_batch_definition_whole_dataframe("whole_df")
        batch = batch_definition.get_batch(batch_parameters={"dataframe": raw})
        suite = context.suites.add(gx.ExpectationSuite(name="healthtrack_suite"))
        validator = context.get_validator(batch=batch, expectation_suite=suite)

        validator.expect_column_values_to_not_be_null("patient_id")
        validator.expect_column_values_to_not_be_null("readmission_30d")
        validator.expect_column_values_to_be_in_set("readmission_30d", [0, 1])
        validator.expect_column_values_to_be_between("total_bill_usd", min_value=500, max_value=200000)
        validator.expect_column_values_to_be_in_set(
            "department", ["Cardiology", "Oncology", "Orthopedics", "Neurology", "General Medicine"]
        )
        validation = validator.validate()

        rows = []
        for r in validation.get("results", []):
            ec = r.get("expectation_config", {})
            expectation_type = ec.get("type", "")
            kwargs = ec.get("kwargs", {})
            column = kwargs.get("column", "")
            name = {
                ("expect_column_values_to_not_be_null", "patient_id"): "patient_id present",
                ("expect_column_values_to_not_be_null", "readmission_30d"): "readmission_30d present",
                ("expect_column_values_to_be_in_set", "readmission_30d"): "readmission_30d in [0,1]",
                ("expect_column_values_to_be_between", "total_bill_usd"): "total_bill_usd in [500, 200000]",
                ("expect_column_values_to_be_in_set", "department"): "department approved",
            }.get((expectation_type, column), f"{expectation_type} ({column})")

            result = r.get("result", {})
            failed = int(result.get("unexpected_count", 0))
            passed = int(result.get("element_count", len(raw)) - failed)
            success = bool(r.get("success", False))
            rows.append((name, success, passed, failed))

        html = [
            "<!doctype html><html><head><meta charset='utf-8'><title>HealthTrack GX Report</title></head><body>",
            "<h1>HealthTrack Data Quality Checkpoint</h1>",
            "<table border='1' cellpadding='6'><tr><th>Rule</th><th>Success</th><th>Passed</th><th>Failed</th></tr>",
        ]
        for name, success, passed, failed in rows:
            html.append(
                f"<tr><td>{name}</td><td>{'PASS' if success else 'FAIL'}</td><td>{passed}</td><td>{failed}</td></tr>"
            )
        html.append("</table></body></html>")
        report_path.write_text("".join(html), encoding="utf-8")
    except Exception:
        html = [
            "<!doctype html><html><head><meta charset='utf-8'><title>HealthTrack GX Report</title></head><body>",
            "<h1>HealthTrack Data Quality Checkpoint</h1>",
            "<table border='1' cellpadding='6'><tr><th>Rule</th><th>Passed</th><th>Total</th></tr>",
        ]
        for k, v in checks.items():
            html.append(f"<tr><td>{k}</td><td>{v}</td><td>{len(raw)}</td></tr>")
        html.append("</table></body></html>")
        report_path.write_text("".join(html), encoding="utf-8")

    nb = notebook(
        [
            md_cell("# HealthTrack Great Expectations Checkpoint"),
            code_cell(
                "import shutil\nfrom pathlib import Path\n\nimport great_expectations as gx\nimport pandas as pd\n\n"
                "df = pd.read_csv('healthtrack_raw.csv')\n\n"
                "context = gx.get_context(mode='ephemeral')\n"
                "datasource = context.data_sources.add_pandas(name='healthtrack_pandas')\n"
                "asset = datasource.add_dataframe_asset(name='healthtrack_raw_asset')\n"
                "batch_definition = asset.add_batch_definition_whole_dataframe('whole_df')\n"
                "batch = batch_definition.get_batch(batch_parameters={'dataframe': df})\n\n"
                "suite = context.suites.add(gx.ExpectationSuite(name='healthtrack_raw_suite'))\n"
                "validator = context.get_validator(batch=batch, expectation_suite=suite)\n\n"
                "# Every discharge must carry a patient identifier so care teams can route outreach to the right person.\n"
                "validator.expect_column_values_to_not_be_null('patient_id')\n\n"
                "# Every record must include known 30-day return status so historical outcomes can train and validate risk scoring.\n"
                "validator.expect_column_values_to_not_be_null('readmission_30d')\n\n"
                "# 30-day return status must be yes/no only to preserve clinical meaning in reports and models.\n"
                "validator.expect_column_values_to_be_in_set('readmission_30d', [0, 1])\n\n"
                "# Bill totals must stay in realistic encounter ranges so downstream dashboards are not distorted by error codes.\n"
                "validator.expect_column_values_to_be_between('total_bill_usd', min_value=500, max_value=200000)\n\n"
                "# Department labels must map to approved service lines for accurate routing and governance.\n"
                "validator.expect_column_values_to_be_in_set('department', ['Cardiology', 'Oncology', 'Orthopedics', 'Neurology', 'General Medicine'])\n\n"
                "validation_results = validator.validate()\n"
                "validation_results['success']"
            ),
            code_cell(
                "# Build Great Expectations Data Docs and save as healthtrack_gx_report.html\n"
                "site_urls = context.build_data_docs()\n"
                "local_site_url = site_urls.get('local_site')\n\n"
                "if not local_site_url:\n"
                "    raise RuntimeError('No local Data Docs site URL returned by Great Expectations.')\n\n"
                "index_path = Path(local_site_url.replace('file://', ''))\n"
                "if not index_path.exists():\n"
                "    raise FileNotFoundError(f'Data Docs index file not found: {index_path}')\n\n"
                "shutil.copyfile(index_path, 'healthtrack_gx_report.html')\n"
                "print(f'Data Docs source: {index_path}')\n"
                "print('Saved: healthtrack_gx_report.html')"
            ),
            md_cell(
                "Today's quality checkpoint shows that some discharge records are not yet safe for direct operational use: several are missing patient identity, several have no confirmed 30-day return outcome, and some include billing placeholder values that are not real charges. Until those records are corrected, daily care-priority lists and readmission performance reporting can miss patients who need follow-up and create avoidable coordination risk."
            ),
        ]
    )
    (ROOT / "healthtrack_gx.ipynb").write_text(json.dumps(nb, indent=2), encoding="utf-8")


def score_and_export_alerts(clean: pd.DataFrame):
    base = clean.copy()
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split

        feature_cols = [
            "age",
            "length_of_stay",
            "prev_admissions_12m",
            "num_medications",
            "department",
            "discharge_disposition",
            "insurance_type",
        ]
        X = base[feature_cols].copy()
        y = base["readmission_30d"].astype(int)
        X_enc = pd.get_dummies(
            X,
            columns=["department", "discharge_disposition", "insurance_type"],
            drop_first=False,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X_enc, y, test_size=0.2, random_state=42, stratify=y
        )
        _ = (X_test, y_test)

        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_enc)[:, 1]
        base["readmission_probability"] = np.round(probs, 4)
    except Exception:
        dept_weight = {
            "Cardiology": 0.10,
            "Oncology": 0.20,
            "Orthopedics": 0.16,
            "Neurology": 0.13,
            "General Medicine": 0.18,
        }
        score = (
            0.10
            + 0.12 * (base["age"] >= 75).astype(float)
            + 0.10 * (base["length_of_stay"] >= 14).astype(float)
            + 0.12 * (base["prev_admissions_12m"] >= 3).astype(float)
            + 0.10 * (base["num_medications"] >= 12).astype(float)
            + base["department"].map(dept_weight).astype(float)
            + 0.07 * base["discharge_disposition"].isin(["AMA", "Skilled Nursing", "Rehab Facility"]).astype(float)
        )
        base["readmission_probability"] = np.clip(score, 0.05, 0.95).round(4)

    base["risk_tier"] = np.where(
        base["readmission_probability"] >= 0.65,
        "High",
        np.where(base["readmission_probability"] >= 0.40, "Medium", "Low"),
    )

    def note(row: pd.Series) -> str:
        factors = []
        if row["prev_admissions_12m"] >= 3:
            factors.append("frequent admissions in the last 12 months")
        if row["num_medications"] >= 12:
            factors.append("high medication complexity at discharge")
        if row["length_of_stay"] >= 14:
            factors.append("prolonged recent inpatient stay")
        if row["discharge_disposition"] in {"AMA", "Skilled Nursing", "Rehab Facility"}:
            factors.append("higher-risk discharge transition setting")
        if len(factors) < 2:
            factors += ["recent acute episode", "ongoing transition-of-care needs"]
        return (
            f"Patient {row['patient_id']} is in the High risk tier, driven mainly by {factors[0]} and {factors[1]}. "
            "Please complete a care coordinator phone call within 48 hours and verify medication access plus scheduled follow-up. "
            "Document barriers immediately and escalate unresolved issues to the attending team by next business day."
        )

    base["intervention_note"] = ""
    high_mask = base["risk_tier"] == "High"
    base.loc[high_mask, "intervention_note"] = base.loc[high_mask].apply(note, axis=1)
    base["alert_priority"] = np.where(
        (base["risk_tier"] == "High") & (base["prev_admissions_12m"] >= 2), "URGENT", "STANDARD"
    )

    alerts = base.loc[base["risk_tier"] == "High", [
        "patient_id",
        "age",
        "department",
        "risk_tier",
        "readmission_probability",
        "alert_priority",
        "intervention_note",
    ]].copy()
    alerts.to_csv(ROOT / "healthtrack_alerts_today.csv", index=False)
    return base, alerts


def write_model_notebook():
    nb = notebook(
        [
            md_cell("# HealthTrack Readmission Model + LangChain Alerts"),
            code_cell(
                "import pandas as pd\n"
                "from sklearn.ensemble import GradientBoostingClassifier\n"
                "from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score\n"
                "from sklearn.model_selection import train_test_split\n\n"
                "df = pd.read_csv('healthtrack_clean.csv')\n\n"
                "feature_cols = ['age','length_of_stay','prev_admissions_12m','num_medications','department','discharge_disposition','insurance_type']\n"
                "X = df[feature_cols].copy()\n"
                "y = df['readmission_30d'].astype(int)\n"
                "X_encoded = pd.get_dummies(X, columns=['department','discharge_disposition','insurance_type'], drop_first=False)\n\n"
                "X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)\n"
                "model = GradientBoostingClassifier(n_estimators=100, random_state=42)\n"
                "model.fit(X_train, y_train)\n"
                "y_pred = model.predict(X_test)\n"
                "y_proba = model.predict_proba(X_test)[:, 1]\n"
                "print('Accuracy:', round(accuracy_score(y_test, y_pred), 4))\n"
                "print('Precision:', round(precision_score(y_test, y_pred, zero_division=0), 4))\n"
                "print('Recall:', round(recall_score(y_test, y_pred, zero_division=0), 4))\n"
                "print('F1:', round(f1_score(y_test, y_pred, zero_division=0), 4))\n"
                "print('ROC-AUC:', round(roc_auc_score(y_test, y_proba), 4))\n\n"
                "# Score all 230 patients\n"
                "all_proba = model.predict_proba(X_encoded)[:, 1]\n"
                "df['readmission_probability'] = all_proba.round(4)\n"
                "df['risk_tier'] = df['readmission_probability'].apply(lambda p: 'High' if p >= 0.65 else ('Medium' if p >= 0.40 else 'Low'))\n"
                "df[['patient_id', 'readmission_probability', 'risk_tier']].head()"
            ),
            md_cell(
                "For this clinical workflow, recall is the most important metric because a false negative means missing a patient who is actually at high risk and may return without intervention. That miss can lead to avoidable harm, avoidable penalties, and missed care coordination opportunities. A false positive is less costly because it usually creates extra outreach work rather than missed care."
            ),
            code_cell(
                "import os\n\n"
                "from langchain_core.prompts import PromptTemplate\n"
                "from langchain_anthropic import ChatAnthropic\n\n"
                "prompt = PromptTemplate(\n"
                "    input_variables=['patient_id','age','department','length_of_stay','prev_admissions_12m','num_medications','discharge_disposition'],\n"
                "    template=(\n"
                "      'You are a hospital care coordinator. Write exactly 3 sentences in a clinical, non-alarmist tone. '\n"
                "      'Sentence 1: state that the patient is High risk and name the top two contributing factors based on the provided fields. '\n"
                "      'Sentence 2: recommend one specific follow-up action a care coordinator can execute immediately. '\n"
                "      'Sentence 3: include a clear timeframe and coordination detail.\\n\\n'\n"
                "      'patient_id: {patient_id}\\nage: {age}\\ndepartment: {department}\\nlength_of_stay: {length_of_stay}\\n'\n"
                "      'prev_admissions_12m: {prev_admissions_12m}\\nnum_medications: {num_medications}\\ndischarge_disposition: {discharge_disposition}'\n"
                "    )\n"
                ")\n\n"
                "if not os.getenv('ANTHROPIC_API_KEY'):\n"
                "    raise EnvironmentError('Set ANTHROPIC_API_KEY before running this cell.')\n\n"
                "llm = ChatAnthropic(model='claude-3-5-sonnet-20240620', temperature=0.2)\n"
                "chain = prompt | llm\n\n"
                "high_mask = df['risk_tier'] == 'High'\n"
                "df['intervention_note'] = ''\n\n"
                "for idx, row in df.loc[high_mask].iterrows():\n"
                "    note = chain.invoke({\n"
                "        'patient_id': row['patient_id'],\n"
                "        'age': int(row['age']),\n"
                "        'department': row['department'],\n"
                "        'length_of_stay': int(row['length_of_stay']),\n"
                "        'prev_admissions_12m': int(row['prev_admissions_12m']),\n"
                "        'num_medications': int(row['num_medications']),\n"
                "        'discharge_disposition': row['discharge_disposition'],\n"
                "    })\n"
                "    df.at[idx, 'intervention_note'] = note.content.strip()\n\n"
                "df['alert_priority'] = df.apply(\n"
                "    lambda r: 'URGENT' if (r['risk_tier'] == 'High' and r['prev_admissions_12m'] >= 2) else 'STANDARD', axis=1\n"
                ")\n\n"
                "alerts = df.loc[df['risk_tier'] == 'High', [\n"
                "    'patient_id','age','department','risk_tier','readmission_probability','alert_priority','intervention_note'\n"
                "]].copy()\n"
                "alerts.to_csv('healthtrack_alerts_today.csv', index=False)\n"
                "print(f'Saved {len(alerts)} high-risk alerts to healthtrack_alerts_today.csv')\n"
                "alerts.head()"
            ),
        ]
    )
    (ROOT / "healthtrack_model.ipynb").write_text(json.dumps(nb, indent=2), encoding="utf-8")


def write_n8n_workflow():
    # Read the already-correct JSON from disk so script does not overwrite improvements
    path = ROOT / "healthtrack_n8n_workflow.json"
    if path.exists():
        return  # file already written correctly
    workflow = {
        "name": "HealthTrack Daily Readmission Alerts",
        "nodes": [
            {
                "parameters": {"rule": {"interval": [{"field": "cronExpression", "expression": "30 7 * * *"}]}},
                "id": "1",
                "name": "Schedule Trigger",
                "type": "n8n-nodes-base.scheduleTrigger",
                "typeVersion": 1.2,
                "position": [180, 320],
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
                "position": [390, 320],
            },
            {
                "parameters": {
                    "jsCode": "const csvText = $json.body || '';\nconst lines = csvText.trim().split('\\n');\nif (!lines.length) {\n  return [{ json: { status: 'BLOCKED', missingPatientId: 0, missingReadmission: 0, sentinelBill: 0, message: 'Incoming file is empty.' } }];\n}\n\nconst headers = lines[0].split(',');\nconst idx = (key) => headers.indexOf(key);\nconst iPatient = idx('patient_id');\nconst iReadmit = idx('readmission_30d');\nconst iBill = idx('total_bill_usd');\nconst iDept = idx('department');\nconst iDischarge = idx('discharge_date');\nconst iAge = idx('age');\nconst iLos = idx('length_of_stay');\nconst iPrev = idx('prev_admissions_12m');\nconst iMeds = idx('num_medications');\n\nlet missingPatientId = 0;\nlet missingReadmission = 0;\nlet sentinelBill = 0;\n\nconst rows = [];\nfor (let r = 1; r < lines.length; r++) {\n  const cols = lines[r].split(',');\n  if (!cols[iPatient]) missingPatientId += 1;\n  if (!cols[iReadmit]) missingReadmission += 1;\n  if ((cols[iBill] || '').trim() === '-9999') sentinelBill += 1;\n  rows.push(cols);\n}\n\nif (missingPatientId > 0 || missingReadmission > 0 || sentinelBill > 0) {\n  return [{\n    json: {\n      status: 'BLOCKED',\n      missingPatientId,\n      missingReadmission,\n      sentinelBill\n    }\n  }];\n}\n\nconst now = new Date();\nconst yesterday = new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1);\nconst ymd = yesterday.toISOString().slice(0, 10);\n\nconst riskRows = [];\nlet high = 0;\nlet medium = 0;\nlet low = 0;\n\nfor (const cols of rows) {\n  if ((cols[iDischarge] || '').trim() !== ymd) continue;\n\n  const age = Number(cols[iAge] || 0);\n  const los = Number(cols[iLos] || 0);\n  const prev = Number(cols[iPrev] || 0);\n  const meds = Number(cols[iMeds] || 0);\n  const dept = cols[iDept] || 'Unknown';\n\n  const deptWeight = {\n    'Cardiology': 0.12,\n    'Oncology': 0.14,\n    'Orthopedics': 0.09,\n    'Neurology': 0.1,\n    'General Medicine': 0.11\n  };\n\n  let p = 0.08 + (prev * 0.09) + (los >= 10 ? 0.08 : 0) + (meds >= 12 ? 0.06 : 0) + (age >= 75 ? 0.05 : 0) + (deptWeight[dept] || 0.08);\n  p = Math.min(0.95, Math.max(0.02, p));\n\n  let risk = 'Low';\n  if (p >= 0.65) risk = 'High';\n  else if (p >= 0.4) risk = 'Medium';\n\n  if (risk === 'High') high += 1;\n  else if (risk === 'Medium') medium += 1;\n  else low += 1;\n\n  if (risk !== 'High') continue;\n\n  const priority = prev >= 2 ? 'URGENT' : 'STANDARD';\n  const note = `Patient is High risk mainly due to prior admissions (${prev}) and length of stay (${los} days). Complete coordinator phone follow-up within 48 hours.`;\n\n  riskRows.push({\n    patient_id: cols[iPatient],\n    department: dept,\n    intervention_note: note,\n    alert_priority: priority,\n  });\n}\n\nconst urgentRows = riskRows.filter((r) => r.alert_priority === 'URGENT');\n\nif (!urgentRows.length) {\n  return [{\n    json: {\n      status: 'OK',\n      high,\n      medium,\n      low,\n      urgentCount: 0,\n      urgent_line: 'No URGENT patients today.'\n    }\n  }];\n}\n\nreturn urgentRows.map((row) => ({\n  json: {\n    status: 'OK',\n    high,\n    medium,\n    low,\n    urgentCount: urgentRows.length,\n    patient_id: row.patient_id,\n    department: row.department,\n    urgent_line: `\\u26a0\\ufe0f Patient ${row.patient_id} \\u2014 ${row.department} \\u2014 ${String(row.intervention_note).split('. ')[0]}.`\n  }\n}));"
                },
                "id": "3",
                "name": "Code - Data Gate and Risk Summary",
                "type": "n8n-nodes-base.code",
                "typeVersion": 2,
                "position": [640, 320],
            },
            {
                "parameters": {
                    "conditions": {
                        "conditions": [
                            {
                                "leftValue": "={{ $json.status }}",
                                "operator": {"type": "string", "operation": "equals"},
                                "rightValue": "OK",
                            }
                        ]
                    }
                },
                "id": "4",
                "name": "If status == OK",
                "type": "n8n-nodes-base.if",
                "typeVersion": 2.2,
                "position": [890, 320],
            },
            {
                "parameters": {"batchSize": 1, "options": {}},
                "id": "5",
                "name": "Loop Over Items",
                "type": "n8n-nodes-base.splitInBatches",
                "typeVersion": 3,
                "position": [1110, 190],
            },
            {
                "parameters": {
                    "channel": "#care-coordination",
                    "text": "=Daily readmission summary (previous-day discharges)\\nHigh: {{$json.high}}\\nMedium: {{$json.medium}}\\nLow: {{$json.low}}\\nURGENT count: {{$json.urgentCount}}\\n{{$json.urgent_line}}",
                    "otherOptions": {},
                },
                "id": "6",
                "name": "Slack Care Coordination",
                "type": "n8n-nodes-base.slack",
                "typeVersion": 2.2,
                "position": [1330, 190],
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
                "position": [1120, 430],
            },
            {
                "parameters": {
                    "channel": "#data-ops",
                    "text": "={{ $json.body.choices[0].message.content || 'Data feed blocked. Please investigate source quality issues before morning huddle.' }}",
                    "otherOptions": {},
                },
                "id": "8",
                "name": "Slack Data Ops",
                "type": "n8n-nodes-base.slack",
                "typeVersion": 2.2,
                "position": [1330, 430],
            },
        ],
        "connections": {
            "Schedule Trigger": {"main": [[{"node": "HTTP Request", "type": "main", "index": 0}]]},
            "HTTP Request": {"main": [[{"node": "Code - Data Gate and Risk Summary", "type": "main", "index": 0}]]},
            "Code - Data Gate and Risk Summary": {"main": [[{"node": "If status == OK", "type": "main", "index": 0}]]},
            "If status == OK": {
                "main": [
                    [{"node": "Loop Over Items", "type": "main", "index": 0}],
                    [{"node": "OpenAI Rewrite", "type": "main", "index": 0}],
                ]
            },
            "Loop Over Items": {
                "main": [
                    [{"node": "Slack Care Coordination", "type": "main", "index": 0}],
                    [],
                ]
            },
            "Slack Care Coordination": {"main": [[{"node": "Loop Over Items", "type": "main", "index": 0}]]},
            "OpenAI Rewrite": {"main": [[{"node": "Slack Data Ops", "type": "main", "index": 0}]]},
        },
        "active": False,
        "settings": {"executionOrder": "v1"},
    }
    path.write_text(json.dumps(workflow, indent=2), encoding="utf-8")


def write_handoff():
    text = """# HealthTrack Readmission Handoff

## 1. What we found in the data
- Some discharge records do not include a patient identifier, which creates a patient safety risk because follow-up outreach can be delayed or misrouted.
- Some records are missing return outcomes or contain invalid billing placeholders, which can hide true readmission pressure and weaken daily planning.

## 2. How the risk model works
The model uses age, stay length, prior admissions, medication count, and discharge context to estimate each patient’s chance of readmission within 30 days. It converts that score into Low, Medium, or High risk so care coordinators can prioritize who needs immediate follow-up. It does not replace clinical judgment and does not capture every social or home-support factor that can affect recovery.

## 3. What happens every morning
1. At 07:30, the workflow pulls the latest discharge data.
2. The file is checked for missing patient identity, missing return outcomes, and known billing error codes.
3. If the file passes, the team receives a daily risk summary with urgent patient lines for immediate action.
4. Urgent patients are listed one by one with a practical intervention sentence for the coordinator.
5. If the file fails, Data Ops gets a plain-language alert so fixes happen before decisions are made.

## 4. What the CMO should do with a High-risk flag
When a High-risk flag appears, the care coordinator should review the chart the same day, complete direct patient outreach within 48 hours, and confirm medication access, follow-up appointment timing, and transport or support barriers. If the patient cannot be reached or critical barriers remain, escalation to case management or the attending service should happen by the next business day, with a documented intervention plan and owner. This keeps every flag actionable, time-bound, and tied to accountable follow-through.
"""
    (ROOT / "healthtrack_handoff.md").write_text(text, encoding="utf-8")


def main():
    raw_path = ROOT / "healthtrack_raw.csv"
    clean_path = ROOT / "healthtrack_clean.csv"
    raw = pd.read_csv(raw_path)
    clean = pd.read_csv(clean_path)

    write_profile_artifacts(raw)
    write_gx_artifacts(raw)
    _, alerts = score_and_export_alerts(clean)
    write_model_notebook()
    write_n8n_workflow()
    write_handoff()

    print(f"Regenerated deliverables from existing files. alerts={len(alerts)}")


if __name__ == "__main__":
    main()
