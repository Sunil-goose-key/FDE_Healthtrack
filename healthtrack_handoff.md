# HealthTrack Readmission Handoff

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
