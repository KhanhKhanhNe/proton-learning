import argparse
import html
import imaplib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from email import message_from_bytes
from email.header import decode_header
from email.message import Message
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "llama-3.3-70b-versatile"
GMAIL_IMAP_HOST = "imap.gmail.com"
GMAIL_IMAP_PORT = 993
MAX_RETRIES = 3
VALID_PRIORITIES = {"low", "medium", "high"}
DEFAULT_OUTPUT_FILE = "daily_email_summary.json"


class SummaryAppError(Exception):
    """Base error for the email summary script."""


class InputError(SummaryAppError):
    """Raised when the script input is invalid."""


class ApiError(SummaryAppError):
    """Raised when an external API call fails."""


class ValidationError(SummaryAppError):
    """Raised when the model response does not match the schema."""


@dataclass
class SummaryResult:
    summary: str
    action_items: list[str]
    priority: str
    people_mentioned: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SummaryResult":
        required_keys = {"summary", "action_items", "priority", "people_mentioned"}
        missing_keys = required_keys - set(data.keys())
        if missing_keys:
            missing_list = ", ".join(sorted(missing_keys))
            raise ValidationError(f"Missing required fields: {missing_list}")

        summary = data["summary"]
        action_items = data["action_items"]
        priority = data["priority"]
        people_mentioned = data["people_mentioned"]

        if not isinstance(summary, str) or not summary.strip():
            raise ValidationError("Field 'summary' must be a non-empty string")

        if not isinstance(action_items, list) or not all(isinstance(item, str) for item in action_items):
            raise ValidationError("Field 'action_items' must be a list of strings")

        if priority not in VALID_PRIORITIES:
            allowed = ", ".join(sorted(VALID_PRIORITIES))
            raise ValidationError(f"Field 'priority' must be one of: {allowed}")

        if not isinstance(people_mentioned, list) or not all(
            isinstance(item, str) for item in people_mentioned
        ):
            raise ValidationError("Field 'people_mentioned' must be a list of strings")

        return cls(
            summary=summary.strip(),
            action_items=[item.strip() for item in action_items if item.strip()],
            priority=priority,
            people_mentioned=[item.strip() for item in people_mentioned if item.strip()],
        )


@dataclass
class GmailMessage:
    subject: str
    sender: str
    received_at: str
    preview: str
    body_text: str


@dataclass
class EmailSummary:
    subject: str
    sender: str
    received_at: str
    summary: str
    action_items: list[str]
    priority: str
    people_mentioned: list[str]


@dataclass
class DailySummary:
    date: str
    total_emails: int
    high_priority_count: int
    items: list[EmailSummary]


def load_environment() -> None:
    """Load .env from repository root and from the current folder."""
    load_dotenv()

    current_dir_env = Path(__file__).resolve().with_name(".env")
    repo_root_env = Path(__file__).resolve().parents[1] / ".env"

    if repo_root_env.exists():
        load_dotenv(repo_root_env)

    if current_dir_env.exists():
        load_dotenv(current_dir_env)


def get_client() -> OpenAI:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ApiError("Missing GROQ_API_KEY. Add it to your environment or .env file.")

    return OpenAI(base_url=BASE_URL, api_key=api_key)


def get_gmail_credentials() -> tuple[str, str]:
    address = os.getenv("GMAIL_ADDRESS", "").strip()
    app_password = os.getenv("GMAIL_APP_PASSWORD", "").strip()

    if not address:
        raise ApiError("Missing GMAIL_ADDRESS. Add your Gmail address to .env.")

    if not app_password:
        raise ApiError("Missing GMAIL_APP_PASSWORD. Add your Gmail app password to .env.")

    return address, app_password


def parse_date_input(value: str, field_name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise InputError(f"{field_name} must be in YYYY-MM-DD format") from exc


def validate_date_range(start_date: date, end_date: date) -> None:
    if end_date < start_date:
        raise InputError("end-date must be greater than or equal to start-date")


def decode_mime_header(value: str | None, default: str = "") -> str:
    if not value:
        return default

    decoded_parts: list[str] = []
    for chunk, encoding in decode_header(value):
        if isinstance(chunk, bytes):
            decoded_parts.append(chunk.decode(encoding or "utf-8", errors="replace"))
        else:
            decoded_parts.append(chunk)

    result = "".join(decoded_parts).strip()
    return result or default


def clean_text(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def decode_payload(part: Message) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        raw_payload = part.get_payload()
        return raw_payload if isinstance(raw_payload, str) else ""

    charset = part.get_content_charset() or "utf-8"
    return payload.decode(charset, errors="replace")


def extract_message_body(message: Message) -> tuple[str, str]:
    text_parts: list[str] = []
    html_parts: list[str] = []

    if message.is_multipart():
        for part in message.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition") or "")
            if "attachment" in disposition.lower():
                continue
            if content_type == "text/plain":
                text_parts.append(decode_payload(part))
            elif content_type == "text/html":
                html_parts.append(decode_payload(part))
    else:
        content_type = message.get_content_type()
        if content_type == "text/plain":
            text_parts.append(decode_payload(message))
        elif content_type == "text/html":
            html_parts.append(decode_payload(message))

    body_text = clean_text("\n".join(text_parts))
    if body_text:
        preview = body_text[:500]
        return preview, body_text

    html_text = clean_text("\n".join(html_parts))
    preview = html_text[:500]
    return preview, html_text


def parse_received_at(message: Message) -> str:
    raw_date = message.get("Date")
    if not raw_date:
        return datetime.now(timezone.utc).isoformat()

    try:
        parsed = parsedate_to_datetime(raw_date)
    except (TypeError, ValueError, IndexError):
        return datetime.now(timezone.utc).isoformat()

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc).isoformat()


def connect_gmail() -> imaplib.IMAP4_SSL:
    address, app_password = get_gmail_credentials()

    try:
        mailbox = imaplib.IMAP4_SSL(GMAIL_IMAP_HOST, GMAIL_IMAP_PORT)
        mailbox.login(address, app_password)
    except imaplib.IMAP4.error as exc:
        raise ApiError(f"Gmail login failed: {exc}") from exc

    return mailbox


def build_imap_search_query(start_date: date, end_date: date) -> str:
    before_date = end_date + timedelta(days=1)
    return f'(SINCE "{start_date.strftime("%d-%b-%Y")}" BEFORE "{before_date.strftime("%d-%b-%Y")}")'


def fetch_gmail_messages(
    start_date: date,
    end_date: date,
    max_emails: int,
    mailbox_name: str,
) -> list[GmailMessage]:
    validate_date_range(start_date, end_date)
    if max_emails <= 0:
        raise InputError("max-emails must be greater than 0")

    mailbox = connect_gmail()

    try:
        status, _ = mailbox.select(mailbox_name, readonly=True)
        if status != "OK":
            raise ApiError(f"Could not open mailbox '{mailbox_name}'.")

        status, data = mailbox.search(None, build_imap_search_query(start_date, end_date))
        if status != "OK":
            raise ApiError("Gmail search failed.")

        message_ids = data[0].split()
        selected_ids = list(reversed(message_ids))[:max_emails]
        messages: list[GmailMessage] = []

        for message_id in selected_ids:
            fetch_status, payload = mailbox.fetch(message_id, "(RFC822)")
            if fetch_status != "OK" or not payload or payload[0] is None:
                continue

            raw_bytes = payload[0][1]
            if not isinstance(raw_bytes, bytes):
                continue

            parsed_message = message_from_bytes(raw_bytes)
            preview, body_text = extract_message_body(parsed_message)
            messages.append(
                GmailMessage(
                    subject=decode_mime_header(parsed_message.get("Subject"), "(no subject)"),
                    sender=decode_mime_header(parsed_message.get("From"), "unknown"),
                    received_at=parse_received_at(parsed_message),
                    preview=preview,
                    body_text=body_text,
                )
            )

        return messages
    finally:
        try:
            mailbox.close()
        except imaplib.IMAP4.error:
            pass
        mailbox.logout()


def build_messages(email_text: str) -> list[dict[str, str]]:
    system_prompt = (
        "Role: You are an executive assistant that summarizes work emails. "
        "Instruction: Read the email and return a structured JSON object with a concise summary, "
        "clear action items, an overall priority, and the people mentioned. "
        "Constraints: Return JSON only. Do not include markdown, commentary, or extra keys. "
        "If there are no action items, return an empty array. "
        "If there are no people mentioned, return an empty array. "
        "Priority must be exactly one of: low, medium, high. "
        "JSON format: "
        '{"summary":"string","action_items":["string"],"priority":"low|medium|high","people_mentioned":["string"]}'
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": email_text},
    ]


def parse_result(raw_output: str) -> SummaryResult:
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Response is not valid JSON: {exc.msg}") from exc

    if not isinstance(data, dict):
        raise ValidationError("Response must be a JSON object")

    return SummaryResult.from_dict(data)


def call_llm(client: OpenAI, messages: list[dict[str, str]]) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        raise ApiError(f"LLM request failed: {exc}") from exc

    content = response.choices[0].message.content
    if not content:
        raise ValidationError("LLM returned an empty response")

    return content


def summarize_email(email_text: str) -> SummaryResult:
    client = get_client()
    messages = build_messages(email_text)
    last_error: str | None = None

    for _ in range(MAX_RETRIES):
        raw_output = call_llm(client, messages)
        try:
            return parse_result(raw_output)
        except ValidationError as exc:
            last_error = str(exc)
            messages.append({"role": "assistant", "content": raw_output})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response did not match the required schema. "
                        f"Validation error: {last_error}. "
                        "Return only a corrected JSON object with the exact required fields."
                    ),
                }
            )

    raise ValidationError(f"Failed to produce valid JSON after {MAX_RETRIES} attempts: {last_error}")


def build_email_text(message: GmailMessage, use_preview_only: bool) -> str:
    body_text = message.preview if use_preview_only or not message.body_text else message.body_text
    return (
        f"Subject: {message.subject}\n"
        f"From: {message.sender}\n"
        f"Received: {message.received_at}\n\n"
        f"Content:\n{body_text}"
    )


def build_daily_report(messages: list[GmailMessage], use_preview_only: bool) -> list[DailySummary]:
    grouped: dict[str, list[EmailSummary]] = {}

    for message in messages:
        received_at = datetime.fromisoformat(message.received_at)
        day_key = received_at.date().isoformat()
        summary = summarize_email(build_email_text(message, use_preview_only))

        grouped.setdefault(day_key, []).append(
            EmailSummary(
                subject=message.subject,
                sender=message.sender,
                received_at=message.received_at,
                summary=summary.summary,
                action_items=summary.action_items,
                priority=summary.priority,
                people_mentioned=summary.people_mentioned,
            )
        )

    report: list[DailySummary] = []
    for day_key in sorted(grouped.keys()):
        items = grouped[day_key]
        high_priority_count = sum(1 for item in items if item.priority == "high")
        report.append(
            DailySummary(
                date=day_key,
                total_emails=len(items),
                high_priority_count=high_priority_count,
                items=items,
            )
        )

    return report


def write_json_report(report: list[DailySummary], output_path: Path | None) -> None:
    payload = json.dumps([asdict(item) for item in report], ensure_ascii=False, indent=2)
    if output_path is None:
        print(payload)
        return

    output_path.write_text(payload + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output_file": str(output_path)}, ensure_ascii=False, indent=2))


def build_error_payload(error: Exception) -> dict[str, str]:
    error_type = "application_error"
    if isinstance(error, InputError):
        error_type = "input_error"
    elif isinstance(error, ApiError):
        error_type = "api_request_failed"
    elif isinstance(error, ValidationError):
        error_type = "invalid_model_response"

    return {"error": error_type, "message": str(error)}


def parse_args() -> argparse.Namespace:
    today = datetime.now(timezone.utc).date()
    parser = argparse.ArgumentParser(description="Read Gmail emails and build a daily summary report.")
    parser.add_argument("--start-date", default=today.isoformat(), help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", default=today.isoformat(), help="End date in YYYY-MM-DD format")
    parser.add_argument("--max-emails", type=int, default=20, help="Maximum number of emails to summarize")
    parser.add_argument("--mailbox", default="INBOX", help="Mailbox to read from, default is INBOX")
    parser.add_argument(
        "--full-body",
        action="store_true",
        help="Use full email body instead of preview when available",
    )
    parser.add_argument(
        "--output-json",
        default=DEFAULT_OUTPUT_FILE,
        help="Path to output JSON report. Use '-' to print to stdout only.",
    )
    return parser.parse_args()


def main() -> None:
    load_environment()

    try:
        args = parse_args()
        start_date = parse_date_input(args.start_date, "start-date")
        end_date = parse_date_input(args.end_date, "end-date")
        messages = fetch_gmail_messages(
            start_date=start_date,
            end_date=end_date,
            max_emails=args.max_emails,
            mailbox_name=args.mailbox,
        )
        report = build_daily_report(messages, use_preview_only=not args.full_body)

        output_path = None if args.output_json == "-" else Path(args.output_json).expanduser().resolve()
        write_json_report(report, output_path)
    except Exception as exc:
        print(json.dumps(build_error_payload(exc), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
