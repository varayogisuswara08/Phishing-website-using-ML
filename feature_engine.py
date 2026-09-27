import re
import ipaddress
from urllib.parse import urlparse


FEATURE_NAMES = [
    "url_length",
    "domain_length",
    "path_length",
    "number_of_dots",
    "number_of_hyphens",
    "number_of_digits",
    "number_of_special_chars",
    "has_ip",
    "has_at_symbol",
    "has_double_slash",
    "has_https",
    "has_http_token",
    "has_www",
    "number_of_subdomains",
    "has_suspicious_words",
    "has_port",
]


SUSPICIOUS_WORDS = [
    "login",
    "verify",
    "verification",
    "account",
    "update",
    "secure",
    "security",
    "bank",
    "signin",
    "confirm",
    "password",
    "wallet",
    "payment",
    "free",
]


def is_ip_address(hostname):
    if not hostname:
        return 0

    try:
        ipaddress.ip_address(hostname)
        return 1
    except ValueError:
        return 0


def extract_features(url):

    if not url.startswith(("http://", "https://")):
        url_for_parse = "http://" + url
    else:
        url_for_parse = url

    parsed = urlparse(url_for_parse)

    hostname = parsed.hostname or ""
    path = parsed.path or ""

    url_lower = url_for_parse.lower()

    special_chars = re.findall(
        r"[^a-zA-Z0-9]",
        url_for_parse
    )

    subdomains = 0

    if hostname:

        parts = hostname.split(".")

        if len(parts) > 2:
            subdomains = len(parts) - 2

    suspicious_word_found = 0

    for word in SUSPICIOUS_WORDS:

        if word in url_lower:
            suspicious_word_found = 1
            break

    features = {

        "url_length":
            len(url_for_parse),

        "domain_length":
            len(hostname),

        "path_length":
            len(path),

        "number_of_dots":
            url_for_parse.count("."),

        "number_of_hyphens":
            url_for_parse.count("-"),

        "number_of_digits":
            sum(c.isdigit() for c in url_for_parse),

        "number_of_special_chars":
            len(special_chars),

        "has_ip":
            is_ip_address(hostname),

        "has_at_symbol":
            int("@" in url_for_parse),

        "has_double_slash":
            int("//" in path),

        "has_https":
            int(url_lower.startswith("https://")),

        "has_http_token":
            int(
                "http" in hostname.lower()
                or "https" in hostname.lower()
            ),

        "has_www":
            int(hostname.lower().startswith("www.")),

        "number_of_subdomains":
            subdomains,

        "has_suspicious_words":
            suspicious_word_found,

        "has_port":
            int(parsed.port is not None)
            if parsed.port else 0,
    }

    return features