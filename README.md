# AI-Integrated CV Management System

This repository is a small, practical demo for receiving candidate CVs (via Twilio), extracting useful fields and storing them in Google Sheets. It was purpose-built to be easy to run locally or on a small cloud VM, and uses a hybrid extraction approach:

- AI extraction (Google Gemini / Generative AI) when `GEMINI_API_KEY` is provided
- Local regex and heuristics fallback when the AI key is not configured or the API call fails

Key extracted fields

The project extracts the following candidate fields from resume text or parsed PDF/DOCX content:

- Name
- Email
- Phone
- Received At (date/time when the application was received or first-detected date in the resume)
- Position Applied For (job title / role the candidate applied for)
- Status (application status such as 'shortlisted', 'in review', 'rejected', etc.)

Files in this repository

- `app.py` — Flask webhook that receives Twilio inbound messages, downloads attached media, extracts text and candidate fields, and appends rows to a Google Sheet.
- `extractor.py` — PDF/DOCX text extraction and candidate parsing; tries Gemini AI first (if configured) and falls back to regex/heuristics.
- `requirements.txt` — Pin list of required Python packages.
- `credentials.json` — (not included) Google service account key used by `gspread` to write to Sheets.
- `README.md` — this file.

Quick start (preserves original quick-start guidance)

1. Create a Python virtual environment and activate it:

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

2. Place your Google service account `credentials.json` into the project root and share the target Google Sheet with the service account's `client_email`.

3. Copy `.env.example` to `.env` and set environment variables you need. If you set `GEMINI_API_KEY` (and `GEMINI_API_URL`/`GEMINI_MODEL` if necessary), `extractor.py` will try the AI route. If not, it will use a robust regex fallback.

Configuration and environment variables

At minimum, the app uses the following environment variables (can be added to a `.env` file):

- `SHEET_NAME` — name of the Google Sheet (defaults to "Candidate CVs")
- `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN` — (optional) used when downloading Twilio-hosted media
- `GEMINI_API_KEY` — optional; if present the extractor will attempt to call the Gemini/Generative AI API
- `GEMINI_API_URL`, `GEMINI_MODEL` — optional overrides for the AI endpoint/model

How the extraction works (implementation notes)

- `extractor.py` contains two major parts:
	- binary → text extraction helpers:
		- `extract_text_from_pdf(path)` — uses `pdfplumber` to extract text from PDF pages.
		- `extract_text_from_docx(path)` — uses `python-docx` to extract paragraphs from DOCX files.
	- parsing and field extraction:
		- If `GEMINI_API_KEY` is set, the code attempts `_call_gemini_extract` which sends the resume text to an AI endpoint and expects JSON with the keys: `Name`, `Email`, `Phone`, `Received At`, `Position Applied For`, `Status`.
		- If the AI call fails or is not configured, `_regex_extract` runs a series of regular expressions and heuristics to find the same fields.

Notes about the new fields

- Position Applied For: The extractor looks for common label patterns like `Position:`, `Position Applied For:`, `Job Title:` and also inspects the top lines for typical role keywords (Engineer, Developer, Manager, Analyst, Designer, etc.).
- Status: The extractor looks for labeled fields such as `Status:` or common status keywords (`shortlisted`, `rejected`, `selected`, `in review`, `hired`, `offer`). When no status is explicit, the regex fallback returns an empty string and the webhook sets a default of `Reviewing` when appending to the sheet.
- Received At: The extractor tries to find explicit "received" or "applied on" labels, otherwise searches the resume for the first date-like token (formats like `YYYY-MM-DD`, `DD/MM/YYYY`, `Month DD, YYYY`, etc.). Note: when messages come through the webhook, `app.py` sets `candidate['Received At'] = datetime.utcnow().isoformat()` to record the actual receive time; the extracted Received At is preserved if present but the webhook currently overwrites it with the current UTC timestamp — you can change this behavior in `app.py` if you prefer to keep the extracted date instead.

Running the Flask webhook locally

1. Ensure environment variables and `credentials.json` are in place.
2. Run:

```powershell
python app.py
```

3. Expose your local server to Twilio (e.g. with `ngrok`) and configure your Twilio webhook URL to point to `https://<your-tunnel>/webhook`.

Example: Using the extractor directly

You can call the parsing helper to inspect a text sample. Example (Python REPL):

```python
from extractor import parse_candidate
sample = "My name is Jane Doe. Email: jane.doe@example.com. Phone: +1 555-555-5555. Position: Senior Engineer. Applied on 2024-05-10. Status: shortlisted."
print(parse_candidate(sample))
```

Expected output (keys may vary slightly depending on your environment):

```json
{
	"Name": "vedant chaudhari",
	"Email": "vedant.chaudhari@example.com",
	"Phone": "+19 884-773-1860",
	"Received At": "2025-10-26",
	"Position Applied For": "Senior Engineer",
	"Status": "shortlisted"
}
```

Testing

- The project contains `pytest` in `requirements.txt`; add or run tests as needed. A simple test can import `extractor.parse_candidate` and assert expected keys for a sample string.

Dependencies

See `requirements.txt` for pinned package versions. Core packages include Flask, pdfplumber, python-docx, gspread and requests.

Troubleshooting / notes

- If Google Sheets writes fail, ensure `credentials.json` is present and the Sheet is shared with the service account email.
- If media downloads from Twilio return 401/403, set `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN` so the webhook can authenticate when downloading Twilio-hosted media.
- AI extraction depends on the exact Gemini/Generative AI endpoint and payload structure. The `_call_gemini_extract` implementation is intentionally lenient; if your provider requires a different client library or payload, replace that function with the correct implementation for your environment.

Contact / next steps

If you want, I can:

- add unit tests for `extractor._regex_extract` covering the new fields,
- switch the webhook to preserve extracted `Received At` instead of overwriting with UTC,
- or add a small example script that runs the extractor against a set of sample resume files.

----

This README retains the original quick-start guidance while expanding documentation about the new fields (`Position Applied For`, `Status`) and the `Received At` behavior.

