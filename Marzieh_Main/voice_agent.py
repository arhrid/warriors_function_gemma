"""
Voice Agent for WhatsApp and Email via function calling.

Supports:
- Voice input (microphone via SpeechRecognition) or typed commands
- WhatsApp messages via pywhatkit
- Professional email via Gmail SMTP (Gemini-formatted)
- Contact lookup from contacts.json with fuzzy matching

Usage:
    python voice_agent.py

Environment variables:
    GEMINI_API_KEY  - Google Gemini API key (for email formatting + function calling)
    GMAIL_USER      - Gmail address for sending emails
    GMAIL_APP_PASS  - Gmail App Password (not regular password)
"""

import json
import os
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from difflib import SequenceMatcher

# Import generate_hybrid from Marzieh's main.py
from main import generate_hybrid

# ── Contact Lookup ──────────────────────────────────────────────

_this_dir = os.path.dirname(os.path.abspath(__file__))
CONTACTS_PATH = os.path.join(_this_dir, "contacts.json")


def _load_contacts():
    try:
        with open(CONTACTS_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _find_contact(query, contacts=None):
    """Fuzzy match a contact name. Returns (name, phone, email) or None."""
    if contacts is None:
        contacts = _load_contacts()
    query_lower = query.lower().strip()
    best_match = None
    best_score = 0.0
    for c in contacts:
        name = c.get("name", "")
        score = SequenceMatcher(None, query_lower, name.lower()).ratio()
        # Also check first name match
        first_name = name.split()[0].lower() if name else ""
        first_score = SequenceMatcher(None, query_lower, first_name).ratio()
        top = max(score, first_score)
        if top > best_score:
            best_score = top
            best_match = c
    if best_match and best_score >= 0.5:
        return best_match
    return None


# ── Tool Definitions ────────────────────────────────────────────

TOOLS = [
    {
        "name": "send_whatsapp",
        "description": "Send a WhatsApp message to a contact by name or phone number",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {
                    "type": "string",
                    "description": "Contact name or phone number (with country code)",
                },
                "message": {
                    "type": "string",
                    "description": "The message text to send",
                },
            },
            "required": ["recipient", "message"],
        },
    },
    {
        "name": "send_email",
        "description": "Send a professional email to a contact by name or email address",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {
                    "type": "string",
                    "description": "Contact name or email address",
                },
                "body": {
                    "type": "string",
                    "description": "The main content/topic of the email",
                },
            },
            "required": ["recipient", "body"],
        },
    },
]


# ── WhatsApp Sending ───────────────────────────────────────────

def _send_whatsapp(recipient, message):
    """Send WhatsApp message using pywhatkit."""
    try:
        import pywhatkit
    except ImportError:
        print("  [!] pywhatkit not installed. Run: pip install pywhatkit")
        return False

    # Resolve contact name to phone number
    phone = recipient
    if not recipient.startswith("+"):
        contact = _find_contact(recipient)
        if contact:
            phone = contact["phone"]
            print(f"  Found contact: {contact['name']} -> {phone}")
        else:
            print(f"  [!] Contact '{recipient}' not found. Using as-is.")
            phone = recipient

    print(f"  Sending WhatsApp to {phone}: {message[:50]}...")
    try:
        pywhatkit.sendwhatmsg_instantly(
            phone_no=phone,
            message=message,
            wait_time=10,
            tab_close=True,
        )
        print("  WhatsApp message sent!")
        return True
    except Exception as e:
        print(f"  [!] WhatsApp error: {e}")
        return False


# ── Email Sending ──────────────────────────────────────────────

def _format_email_with_gemini(recipient_name, body_topic):
    """Use Gemini to format a professional email."""
    try:
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None

        client = genai.Client(api_key=api_key)
        prompt = (
            f"Write a short, professional email to {recipient_name}. "
            f"Topic: {body_topic}\n\n"
            "Include a proper greeting, the main content, and a professional sign-off. "
            "Keep it concise (3-5 sentences). Return ONLY the email body, no subject line."
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        print(f"  [!] Gemini formatting failed: {e}")
        return None


def _generate_subject_with_gemini(body):
    """Use Gemini to generate a concise email subject line."""
    try:
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return "Message"

        client = genai.Client(api_key=api_key)
        prompt = (
            f"Generate a concise email subject line (max 8 words) for this email:\n\n{body}\n\n"
            "Return ONLY the subject line, nothing else."
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text.strip().strip('"').strip("'")
    except Exception:
        return "Message"


def _send_email(recipient, body):
    """Send email via Gmail SMTP with Gemini-formatted content."""
    gmail_user = os.environ.get("GMAIL_USER")
    gmail_pass = os.environ.get("GMAIL_APP_PASS")

    if not gmail_user or not gmail_pass:
        print("  [!] Set GMAIL_USER and GMAIL_APP_PASS environment variables.")
        print("  [!] Get an App Password at: https://myaccount.google.com/apppasswords")
        return False

    # Resolve recipient
    email_addr = recipient
    recipient_name = recipient
    if "@" not in recipient:
        contact = _find_contact(recipient)
        if contact:
            email_addr = contact["email"]
            recipient_name = contact["name"]
            print(f"  Found contact: {recipient_name} -> {email_addr}")
        else:
            print(f"  [!] Contact '{recipient}' not found and no email address given.")
            return False

    # Format email body with Gemini
    print("  Formatting email with Gemini...")
    formatted_body = _format_email_with_gemini(recipient_name, body)
    if not formatted_body:
        formatted_body = f"Hi {recipient_name},\n\n{body}\n\nBest regards"

    # Generate subject
    subject = _generate_subject_with_gemini(formatted_body)

    # Send via SMTP
    print(f"  Sending email to {email_addr}...")
    print(f"  Subject: {subject}")
    print(f"  Body preview: {formatted_body[:100]}...")

    try:
        msg = MIMEMultipart()
        msg["From"] = gmail_user
        msg["To"] = email_addr
        msg["Subject"] = subject
        msg.attach(MIMEText(formatted_body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_user, gmail_pass)
            server.sendmail(gmail_user, email_addr, msg.as_string())

        print("  Email sent successfully!")
        return True
    except Exception as e:
        print(f"  [!] Email error: {e}")
        return False


# ── Tool Dispatch ──────────────────────────────────────────────

def _dispatch(function_calls):
    """Execute the function calls returned by generate_hybrid."""
    for call in function_calls:
        name = call.get("name", "")
        args = call.get("arguments", {})
        print(f"\n  -> Calling: {name}({json.dumps(args)})")

        if name == "send_whatsapp":
            _send_whatsapp(args.get("recipient", ""), args.get("message", ""))
        elif name == "send_email":
            _send_email(args.get("recipient", ""), args.get("body", ""))
        else:
            print(f"  [!] Unknown tool: {name}")


# ── Voice Input ────────────────────────────────────────────────

def _listen_voice():
    """Capture voice input from microphone."""
    try:
        import speech_recognition as sr
    except ImportError:
        print("[!] SpeechRecognition not installed. Run: pip install SpeechRecognition")
        return None

    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("\nListening... (speak now)")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
            print("Processing speech...")
            text = recognizer.recognize_google(audio)
            print(f"You said: \"{text}\"")
            return text
    except Exception as e:
        print(f"[!] Voice input error: {e}")
        return None


# ── Main Loop ──────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Voice Agent — WhatsApp & Email")
    print("=" * 55)
    print("\nCommands:")
    print("  - Press Enter for voice input (microphone)")
    print("  - Type your command directly")
    print("  - Type 'quit' to exit")
    print("  - Type 'contacts' to list contacts")
    print()

    contacts = _load_contacts()
    if contacts:
        print(f"Loaded {len(contacts)} contacts from contacts.json")
    else:
        print("[!] No contacts loaded. Create contacts.json.")

    while True:
        user_input = input("\n> ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if user_input.lower() == "contacts":
            for c in contacts:
                print(f"  {c['name']}: {c.get('phone', 'N/A')} | {c.get('email', 'N/A')}")
            continue

        # Empty input = voice mode
        if not user_input:
            user_input = _listen_voice()
            if not user_input:
                continue

        # Run through generate_hybrid
        messages = [{"role": "user", "content": user_input}]
        print("\nProcessing with hybrid function calling...")
        start = time.time()
        result = generate_hybrid(messages, TOOLS)
        elapsed = (time.time() - start) * 1000

        source = result.get("source", "unknown")
        calls = result.get("function_calls", [])

        print(f"  Source: {source} | Time: {elapsed:.0f}ms | Calls: {len(calls)}")

        if not calls:
            print("  No function calls detected. Try rephrasing.")
            continue

        # Show what was detected
        for call in calls:
            print(f"  Detected: {call['name']}({json.dumps(call.get('arguments', {}))})")

        # Confirm and execute
        confirm = input("\nExecute? [Y/n]: ").strip().lower()
        if confirm in ("", "y", "yes"):
            _dispatch(calls)
        else:
            print("  Cancelled.")


if __name__ == "__main__":
    main()
