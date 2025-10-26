import os
import tempfile
import logging
from datetime import datetime

from flask import Flask, request, jsonify
import requests
from dotenv import load_dotenv

import gspread
from google.oauth2 import service_account

from extractor import extract_text_from_pdf, extract_text_from_docx, parse_candidate

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load env
SHEET_NAME = os.getenv('SHEET_NAME', 'Candidate CVs')
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')

# Initialize gspread client using local credentials.json service account file
def init_gspread_client():
    creds_path = os.path.join(os.getcwd(), 'credentials.json')
    if not os.path.exists(creds_path):
        logger.warning('credentials.json not found. Google Sheets integration will fail until provided.')
        return None
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    creds = service_account.Credentials.from_service_account_file(creds_path, scopes=scopes)
    client = gspread.authorize(creds)
    logger.info('gspread client initialized')
    return client

gc = init_gspread_client()
sheet = None
if gc:
    try:
        sheet = gc.open(SHEET_NAME).sheet1
    except Exception as e:
        logger.warning(f'Could not open sheet "{SHEET_NAME}": {e}')


def append_candidate_to_sheet(candidate: dict):
    if not sheet:
        logger.info('No sheet client available; printing candidate instead:')
        logger.info(candidate)
        return
    row = [candidate.get('Name', 'Empty'), candidate.get('Email', ''), candidate.get('Phone', ''), candidate.get('Received At', ''), candidate.get('Status', 'Reviewing')]
    try:
        sheet.append_row(row)
        logger.info('Appended row to Google Sheet')
    except Exception as e:
        logger.error('Failed to append to sheet: %s', e)


@app.route('/webhook', methods=['POST'])
def webhook():
    # Twilio posts form-encoded data
    data = request.form
    logger.info('Incoming webhook: keys=%s', list(data.keys()))

    body = data.get('Body', '')
    num_media = int(data.get('NumMedia', '0'))
    extracted_text = ''

    if num_media and 'MediaUrl0' in data:
        media_url = data.get('MediaUrl0')
        content_type = data.get('MediaContentType0', '')
        logger.info('Received media: %s (%s)', media_url, content_type)

        # Download media (Twilio media requires basic auth using Account SID:Auth Token)
        auth = None
        if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
            auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        try:
            resp = requests.get(media_url, auth=auth, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            logger.error('Failed to download media: %s', e)
            return ('', 500)

        # Save to temporary file
        suffix = ''
        if 'pdf' in content_type.lower():
            suffix = '.pdf'
        elif 'word' in content_type.lower() or 'msword' in content_type.lower() or 'officedocument' in content_type.lower():
            suffix = '.docx'
        else:
            # try to guess from URL
            if media_url.lower().endswith('.pdf'):
                suffix = '.pdf'
            elif media_url.lower().endswith('.docx'):
                suffix = '.docx'

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            tf.write(resp.content)
            temp_path = tf.name

        if temp_path.endswith('.pdf'):
            extracted_text = extract_text_from_pdf(temp_path)
        elif temp_path.endswith('.docx'):
            extracted_text = extract_text_from_docx(temp_path)
        else:
            # unknown binary type — fallback to treating as text
            extracted_text = resp.text if hasattr(resp, 'text') else ''

    else:
        # No media — use plain text body
        extracted_text = body

    # Parse candidate details using AI if available (or fallback)
    candidate = parse_candidate(extracted_text)
    # Preserve extracted 'Received At' when available; otherwise set to current UTC time
    if not candidate.get('Received At'):
        candidate['Received At'] = datetime.utcnow().isoformat()

    append_candidate_to_sheet(candidate)

    # Twilio expects a valid TwiML or 200 OK
    return ('', 200)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
