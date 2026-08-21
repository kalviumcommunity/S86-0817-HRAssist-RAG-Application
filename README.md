# HRAssist — AI-Powered HR Policy Assistant

> **"Ask HR. Find the Policy. Get the Answer."**

![Status](https://img.shields.io/badge/Status-Draft-yellow)
![Version](https://img.shields.io/badge/Version-1.0-blue)
![Product Type](https://img.shields.io/badge/Product-AI--Powered%20HR%20Self--Service-green)

---

## 📌 Executive Summary

**HRAssist** is an enterprise AI-powered HR knowledge assistant designed to streamline employee self-service. Operating on a **Retrieval-Augmented Generation (RAG)** architecture, HRAssist allows employees to ask HR-related questions in natural language and receive accurate, concise answers grounded directly in approved company policy documents.

By dynamically prioritizing policies specific to an employee's region (e.g., Leave, Benefits, Handbooks), HRAssist eliminates repetitive inquiries sent to HR personnel while guaranteeing compliance and transparency.

---

## 🎯 Business Problem & Vision

### Problem
HR departments spend significant manual effort resolving repetitive questions whose answers already exist across regional handbooks and benefit guides (e.g., leave entitlements, approval workflows, regional benefits). This creates:
* High operational workload for HR teams.
* Delays for employees seeking quick, accurate policy answers.
* Information silos and risks of cross-region policy misinterpretation.

### Product Vision
HRAssist acts as a centralized, region-aware AI knowledge partner. It **does not replace HR**, but offloads routine informational queries so HR professionals can focus on strategic initiatives, complex employee relations, and policy development.

---

## 🚀 Key Features

* 💬 **Natural Language Querying:** Ask complex HR questions in everyday language.
* 🌍 **Region-Aware Retrieval:** Automatically filters policy documents based on the employee's assigned location/region.
* 📌 **Grounded Answers with Citations:** Responses include explicit references to the source policy section or document.
* 🛡️ **Zero-Hallucination Fallback:** If policy context is insufficient, HRAssist provides a safe fallback and directs the employee to HR instead of generating unsupported answers.
* 📑 **HR Admin Knowledge Hub:** HR admins can upload, categorize, update, and assign region tags to policy documents seamlessly.
* 📊 **Feedback & Analytics:** Employees can submit 👍/👎 feedback with comments; HR admins can monitor unanswered questions to identify documentation gaps.

---

## 👤 Target Users & Roles

| User Role | Description & Primary Actions |
| :--- | :--- |
| **Employees (Primary)** | Log in, ask natural-language questions, view region-specific policy answers with source citations, view past chat history, and provide answer feedback. |
| **HR Team (Secondary)** | Upload and manage regional HR documents, replace obsolete policies, review unanswered queries, and monitor feedback analytics. |

---

## 🏗️ Technical Architecture & RAG Pipeline

```
                    ┌───────────────┐
                    │   Employee    │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   Frontend    │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  Backend API  │
                    └───────┬───────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
       ┌──────────────┐            ┌──────────────┐
       │ Authentication│            │  AI / RAG    │
       │ & User Data   │            │   Pipeline   │
       └──────────────┘            └───────┬──────┘
                                           │
                                           ▼
                                   ┌──────────────┐
                                   │ Vector /     │
                                   │ Search DB    │
                                   └───────┬──────┘
                                           │
                                           ▼
                                   ┌──────────────┐
                                   │ HR Documents │
                                   └──────────────┘
```

### Data & Retrieval Workflow
1. **Document Ingestion:** HR Admin uploads documents $\rightarrow$ Text Extraction $\rightarrow$ Cleaning & Chunking $\rightarrow$ Metadata Tagging (Region, Document Type, Version) $\rightarrow$ Embedding & Indexing in Vector Database.
2. **Query Processing:** Employee asks a question $\rightarrow$ System retrieves Employee Region $\rightarrow$ Performs Region-Filtered Semantic Search against Vector DB $\rightarrow$ Extracts top relevant policy chunks.
3. **Grounded Generation:** LLM generates a concise response restricted to retrieved context $\rightarrow$ Appends document/section references $\rightarrow$ Displays answer to Employee.

---

## 📱 User Interface (UX) Blueprints

<details>
<summary><b>1. Employee Home Screen</b></summary>

```text
┌──────────────────────────────────┐
│          HRAssist                │
│     Your HR Policy Assistant     │
├──────────────────────────────────┤
│                                  │
│  How can I help you?             │
│                                  │
│  ┌────────────────────────────┐  │
│  │ Ask your HR question...    │  │
│  └────────────────────────────┘  │
│                                  │
│  Try asking:                     │
│  • How many leave days do I get? │
│  • What benefits am I eligible? │
│  • Can I carry forward leave?    │
│                                  │
├──────────────────────────────────┤
│        Recent Questions          │
│                                  │
│  Leave entitlement               │
│  Benefits eligibility            │
│  Work-from-home policy           │
└──────────────────────────────────┘
```
</details>

<details>
<summary><b>2. Answer & Citation Screen</b></summary>

```text
┌──────────────────────────────────┐
│            HRAssist              │
├──────────────────────────────────┤
│                                  │
│ Your Question                    │
│ "How many leave days can I take?"│
│                                  │
│ ──────────────────────────────── │
│                                  │
│ Answer                           │
│ According to the leave policy    │
│ applicable to your region, you   │
│ are entitled to X leave days...  │
│                                  │
│ ──────────────────────────────── │
│ Source                           │
│ Regional Leave Policy            │
│ Section X                        │
│                                  │
│ Was this helpful?                │
│     👍 Yes       👎 No            │
└──────────────────────────────────┘
```
</details>

<details>
<summary><b>3. HR Admin Dashboard</b></summary>

```text
┌──────────────────────────────────┐
│          HRAssist Admin          │
├──────────────────────────────────┤
│ Documents | Questions | Feedback │
├──────────────────────────────────┤
│                                  │
│ HR Documents                     │
│ Employee Handbook     Global     │
│ Leave Policy          India      │
│ Benefits Policy       India      │
│ Regional Policy       US         │
│                                  │
│       [+ Upload Document]        │
│                                  │
├──────────────────────────────────┤
│ Frequently Asked Questions       │
│ 1. Leave carry-forward           │
│ 2. Health benefits               │
│ 3. Eligibility                   │
│ 4. Work-from-home policy         │
└──────────────────────────────────┘
```
</details>

---

## 🔒 Security, Access Control & Compliance

* **Role-Based Access Control (RBAC):** Strict separation between Employee and HR Admin functionalities.
* **Regional Isolation:** Metadata filtering prevents employees from accessing policy documents from unassigned regions.
* **Grounding & Safety:** Fallback triggers prevent AI hallucinations. If information is absent from approved sources, the system safely routes the query to HR.
* **Privacy First:** User interaction data is strictly used for self-service functionality and quality feedback.

---

## 📋 Feature Scope

### In Scope (V1.0)
- [x] Secure Employee & HR Admin Authentication.
- [x] Natural Language QA powered by RAG.
- [x] Region-aware document filtering.
- [x] Source citations (Document & Section).
- [x] Fallback handling for missing policy data.
- [x] HR Admin Document Upload, Categorization & Region Tagging.
- [x] Question & Feedback Monitoring for HR.

### Out of Scope (V1.0)
- Automated leave application submission or approval.
- Payroll or performance management processing.
- Direct HRMS database modifications.
- Legal advice or automated decision-making.
- Voice support.

---

## 📈 Success Metrics & KPIs

| KPI Metric | Measurement Method | Target | Timeline |
| :--- | :--- | :--- | :--- |
| **HR Question Deflection** | % queries resolved without HR intervention | Baseline + Target TBD | 30–60 Days |
| **Answer Accuracy** | % answers verified & grounded in policy | High Accuracy Target | 30 Days |
| **Employee Satisfaction** | Positive (👍) vs Total Rated Responses | High CSAT Target | 30 Days |
| **Unanswered Questions** | Reduction in escalated tickets | Reduction Target | 60 Days |

---

## 🔮 Future Roadmap (V2.0)

* 🔌 **HRMS & Slack/Teams Integration:** Embed assistant directly into enterprise chat platforms and HR software.
* 🗓️ **Leave Balance Queries:** Direct lookup of personalized employee leave balances.
* 🌐 **Multilingual Support:** Auto-translate regional policies for global workforces.
* 🎫 **Automated HR Ticket Creation:** One-click escalation to HR service desks for unanswered queries.

---

## 📄 License & Ownership

This project is owned and maintained by the **HR & Engineering Teams**. All rights reserved.
