import os
import re
import logging
import json
from typing import Dict

import pdfplumber
import docx
import requests
import jsonschema
from dateutil import parser as dateparser

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'text-bison-001')

# Reusable email regex
EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')


def _normalize_phone_str(raw: str) -> str:
    if not raw:
        return ''
    raw = raw.strip()
    # keep leading + if present
    if raw.startswith('+'):
        return '+' + re.sub(r'\D', '', raw[1:])
    digits = re.sub(r'\D', '', raw)
    if len(digits) == 10:
        return '+91' + digits
    if len(digits) == 11 and digits.startswith('0'):
        return '+91' + digits[1:]
    if len(digits) >= 11 and digits.startswith('91'):
        return '+' + digits
    if 8 <= len(digits) <= 9:
        return '+91' + digits
    return digits


def _sanitize_one_line(s: str) -> str:
    return ' '.join(s.strip().split()) if s else ''


def _validate_and_normalize(parsed: dict) -> dict:
    """Validate the parsed JSON fields and normalize values. Returns normalized dict or None on invalid."""
    if not isinstance(parsed, dict):
        return None
    # Expected keys
    keys = ['Name', 'Email', 'Phone', 'Received At', 'Position Applied For', 'Status']
    out = {}
    for k in keys:
        v = parsed.get(k, '') if k in parsed else parsed.get(k.replace(' ', ''), '')
        if v is None:
            v = ''
        if not isinstance(v, str):
            try:
                v = str(v)
            except Exception:
                v = ''
        v = _sanitize_one_line(v)
        out[k] = v

    # Validate email
    if out['Email'] and not EMAIL_RE.search(out['Email']):
        out['Email'] = ''

    # Normalize phone
    out['Phone'] = _normalize_phone_str(out['Phone'])
    # If phone is very short after normalization, clear it
    digits_only = re.sub(r'\D', '', out['Phone'])
    if len(digits_only) < 8:
        out['Phone'] = ''

    # Normalize Received At to ISO date where possible
    if out['Received At']:
        try:
            dt = dateparser.parse(out['Received At'], fuzzy=True)
            out['Received At'] = dt.date().isoformat()
        except Exception:
            # keep original sanitized value if parsing fails
            out['Received At'] = out['Received At']

    return out


def extract_text_from_pdf(path: str) -> str:
    try:
        text = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text.append(t)
        return "\n".join(text)
    except Exception as e:
        logger.error('PDF extraction failed: %s', e)
        return ''


def extract_text_from_docx(path: str) -> str:
    try:
        doc = docx.Document(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text]
        return "\n".join(paragraphs)
    except Exception as e:
        logger.error('DOCX extraction failed: %s', e)
        return ''


def _regex_extract(text: str) -> Dict[str, str]:
    # Basic regex-based extraction for Name, Email, Phone
    email_re = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
    # phone pattern: capture groups of digits allowing common separators
    phone_re = re.compile(r'(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}')

    # Position extraction removed from regex fallback - we rely on AI inference when needed

    def _infer_position_with_gemini(text: str) -> str:
        """If GEMINI_API_KEY is present, ask the model to infer the candidate's likely role/domain.

        Returns a short string (or empty string on failure).
        """
        if not GEMINI_API_KEY:
            return ''
        try:
            # reuse the generic call endpoint but with a focused prompt
            endpoint = os.getenv('GEMINI_API_URL')
            model = os.getenv('GEMINI_MODEL', 'gemini-flash-2.0')
            if not endpoint:
                endpoint = f'https://generativeai.googleapis.com/v1beta2/models/{model}:predict'

            # Build prompt using LangChain PromptTemplate if available (templating only)
            try:
                from langchain.prompts import PromptTemplate
                template = (
                    "Read the resume text and return a single short job title or domain that best describes "
                    "the candidate (for example: 'Data Scientist', 'Frontend Developer', 'Marketing Specialist'). "
                    "Return plain text only, no extra labels. If you can't determine, return an empty string.\n\n"
                    "Resume:\n{resume}"
                )
                pt = PromptTemplate(input_variables=["resume"], template=template)
                prompt = pt.format(resume=text)
            except Exception:
                # LangChain not available or templating failed — use simple formatting
                prompt = (
                    "Read the resume text and return a single short job title or domain that best describes "
                    "the candidate (for example: 'Data Scientist', 'Frontend Developer', 'Marketing Specialist'). "
                    "Return plain text only, no extra labels. If you can't determine, return an empty string.\n\n"
                    f"Resume:\n{text}"
                )

            headers = {
                'Authorization': f'Bearer {GEMINI_API_KEY}',
                'Content-Type': 'application/json'
            }
            payload = {'prompt': prompt, 'maxOutputTokens': 64}
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=20)
            resp.raise_for_status()
            data = resp.text
            # try to extract a short line
            # prefer first non-empty line
            for line in data.splitlines():
                line = line.strip()
                if not line:
                    continue
                # if the line looks like JSON, try to parse and find single string
                if line.startswith('{') and line.endswith('}'):
                    try:
                        parsed = json.loads(line)
                        # look for common keys
                        for k in ('Position Applied For', 'Position', 'Role', 'job'):
                            if k in parsed and parsed[k]:
                                return str(parsed[k]).split('\n')[0].strip()
                    except Exception:
                        pass
                # otherwise return cleaned first line
                return line.split('\n')[0].strip()
        except Exception:
            return ''

    def _extract_status(text: str) -> str:
        # Look for explicit status labels
        patterns = [
            r'(?:Status|Application Status|Current Status)[:\-\s]{1,30}(.+)',
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return m.group(1).split('\n')[0].strip()
        # fallback: search for common status keywords
        keywords = ['shortlisted', 'rejected', 'selected', 'in review', 'under review', 'hired', 'offer', 'screened']
        for k in keywords:
            if re.search(r'\b' + re.escape(k) + r'\b', text, re.IGNORECASE):
                return k
        return ''

    def _extract_received_at(text: str) -> str:
        # Try to find labeled received/apply dates first
        received_patterns = [
            r'(?:Received(?: at| on)?|Received Date|Application received|Applied on|Applied:)[:\-\s]{0,30}(.+)',
        ]
        # date patterns to try if label not present
        date_patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b',
            r'\b[A-Za-z]{3,9}\s+\d{1,2},\s*\d{4}\b',
        ]
        for p in received_patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                candidate = m.group(1).split('\n')[0].strip()
                # if candidate contains a date, return; else return raw candidate
                for dp in date_patterns:
                    if re.search(dp, candidate):
                        return candidate
                return candidate
        # no label found, search for first date in document
        for dp in date_patterns:
            m = re.search(dp, text)
            if m:
                return m.group(0)
        return ''

    email = email_re.search(text)

    # Find plausible phone candidates (allow separators) and pick the best one
    raw_phone_candidates = re.findall(r"\+?\d[\d\-\s().]{6,}\d", text)

    def _choose_phone(cands):
        best = ''
        best_len = 0
        for c in cands:
            digits = re.sub(r'\D', '', c)
            if not digits:
                continue
            # prefer longest digit sequence
            if len(digits) > best_len:
                best = c
                best_len = len(digits)
        # require a minimum reasonable length (at least 8 digits); otherwise treat as missing
        if best_len < 8:
            return ''
        # normalize
        raw = best.strip()
        if raw.startswith('+'):
            return '+' + re.sub(r'\D', '', raw[1:])
        digits = re.sub(r'\D', '', raw)
        if len(digits) == 10:
            return '+91' + digits
        if len(digits) == 11 and digits.startswith('0'):
            return '+91' + digits[1:]
        if len(digits) >= 11 and digits.startswith('91'):
            return '+' + digits
        if 8 <= len(digits) <= 9:
            return '+91' + digits
        return digits

    phone_val = _choose_phone(raw_phone_candidates)

    # Heuristic for name: look for explicit label first, then top lines, then email local-part
    name = ''
    # try 'Name:' labels
    m_name_label = re.search(r"Name[:\-\s]{1,30}([A-Za-z ,.'\"\-\n]{2,80})", text, re.IGNORECASE)
    if m_name_label:
        cand = m_name_label.group(1).split('\n')[0].strip()
        if cand:
            name = cand

    # header words to ignore
    header_words = ('education', 'experience', 'skills', 'projects', 'activities', 'summary', 'objective')

    if not name:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        # prefer a top-line that looks like a person's name: Title Case, 2-4 words
        title_name_re = re.compile(r'^[A-Z][a-z]+(?: [A-Z][a-z]+){0,3}$')
        for line in lines[:12]:
            low = line.lower()
            # skip lines that look like emails or phones or contain header words or are UPPERCASE headings
            if email_re.search(line) or re.search(r"\d{4}", line) or any(w in low for w in header_words):
                continue
            if line.isupper():
                continue
            if title_name_re.match(line):
                name = line
                break
        # fallback: take first line that's two words with capitalized starts
        if not name:
            for line in lines[:12]:
                words = line.split()
                if 1 < len(words) <= 4 and all(w[0].isupper() for w in words if w):
                    # ensure it is not a header
                    if not any(h in line.lower() for h in header_words):
                        name = ' '.join(words)
                        break

    # fallback: 'My name is ...' pattern
    if not name:
        m = re.search(r"My name is\s+([A-Za-z ,.'\"\-]{2,80})", text, re.IGNORECASE)
        if m:
            name = m.group(1).split('\n')[0].strip()

    # final fallback: derive from email local-part
    if not name and email:
        local = email.group(0).split('@')[0]
        parts = re.split(r'[._\-]+', local)
        parts = [p for p in parts if p and not p.isdigit()]
        if parts:
            name = ' '.join(p.capitalize() for p in parts[:3])

    # normalize email
    email_val = email.group(0).strip() if email else ''

    # determine position: use AI inference when available (regex-based position extraction removed)
    position_val = _infer_position_with_gemini(text)

    # status
    status_val = _extract_status(text) or 'Reviewing'

    # sanitize all values to single-line strings
    def _one_line(s: str) -> str:
        return ' '.join(s.strip().split()) if s else ''

    return {
        'Name': _one_line(name),
        'Email': _one_line(email_val),
        'Phone': _one_line(phone_val),
        'Received At': _one_line(_extract_received_at(text)),
        'Position Applied For': _one_line(position_val),
        'Status': _one_line(status_val)
    }


def _call_gemini_extract(text: str) -> Dict[str, str]:
    if not GEMINI_API_KEY:
        raise RuntimeError('No GEMINI_API_KEY configured')

    # NOTE: The exact endpoint for Gemini may change. This function attempts a generic HTTP POST
    # to a Google Generative AI endpoint. If this does not match your environment, replace
    # this function with the correct client call.
    endpoint = os.getenv('GEMINI_API_URL')
    if not endpoint:
        # sane default pattern (may require adjustment)
        endpoint = f'https://generativeai.googleapis.com/v1beta2/models/{GEMINI_MODEL}:predict'

    # Build a strict JSON prompt
    json_prompt_template = (
        "You are given a resume text. Return a single valid JSON object with exactly the following keys:"
        " Name, Email, Phone, Received At, Position Applied For, Status. \n"
        "Each value must be a string. If a value cannot be determined, return an empty string for that key."
        " Do not add any extra keys or commentary — output must be raw JSON only.\n\nResume:\n{resume}"
    )

    # Try using LangChain's VertexAI LLM if available (preferred). If not, fallback to HTTP request.
    try:
        resp_text = None
        try:
            # LangChain LLM usage
            from langchain.prompts import PromptTemplate
            from langchain.llms import VertexAI
            from langchain import LLMChain

            prompt_template = PromptTemplate(input_variables=['resume'], template=json_prompt_template)
            # VertexAI reads credentials from environment / ADC; model name from GEMINI_MODEL
            model_name = os.getenv('GEMINI_MODEL', GEMINI_MODEL)
            llm = VertexAI(model_name=model_name, max_output_tokens=512, temperature=0)
            chain = LLMChain(llm=llm, prompt=prompt_template)
            resp_text = chain.run(resume=text)
        except Exception:
            # fallback to direct HTTP POST to configured Gemini endpoint
            headers = {
                'Authorization': f'Bearer {GEMINI_API_KEY}',
                'Content-Type': 'application/json'
            }
            payload = {'prompt': json_prompt_template.format(resume=text), 'maxOutputTokens': 512}
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            # try to use text response
            try:
                resp_text = resp.text
            except Exception:
                resp_text = str(resp.content)

        # Parse first JSON object from resp_text
        if not resp_text:
            return _regex_extract(text)

        m = re.search(r'\{.*\}', resp_text, re.DOTALL)
        parsed = None
        if m:
            try:
                parsed = json.loads(m.group(0))
            except Exception:
                parsed = None
        else:
            try:
                parsed = json.loads(resp_text)
            except Exception:
                parsed = None

        if parsed and isinstance(parsed, dict):
            validated = _validate_and_normalize(parsed)
            if validated:
                return validated
            else:
                return _regex_extract(text)
        else:
            return _regex_extract(text)
    except Exception as e:
        logger.error('Gemini/LLM call failed: %s', e)
        return _regex_extract(text)


def parse_candidate(text: str) -> Dict[str, str]:
    """Return a dict with Name, Email, Phone.

    If GEMINI_API_KEY is set, attempt to call the API; otherwise use local regex fallback.
    """
    if not text:
        return {'Name': '', 'Email': '', 'Phone': ''}

    if GEMINI_API_KEY:
        try:
            return _call_gemini_extract(text)
        except Exception:
            return _regex_extract(text)
    else:
        return _regex_extract(text)


if __name__ == '__main__':
    sample = """
    My name is vedant chaudhari. You can email me at vedant.chaudhari@example.com or call me at (555) 123-4567. I am a software developer.
    """
    print(parse_candidate(sample))
