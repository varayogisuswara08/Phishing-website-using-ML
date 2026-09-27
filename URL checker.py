from urllib.parse import urlparse
def extract_features(url):

    parsed = urlparse(url)

    features = {}

    # URL length
    features["URL_Length"] = len(url)

    # Has IP address
    hostname = parsed.hostname or ""

    parts = hostname.split(".")

    has_ip = all(
        part.isdigit()
        for part in parts
        if part
    ) and len(parts) == 4

    features["Has_IP"] = int(has_ip)

    # Has @ symbol
    features["Has_At"] = int("@" in url)

    # Number of dots
    features["Number_of_Dots"] = url.count(".")

    # Number of hyphens
    features["Number_of_Hyphens"] = url.count("-")

    # HTTPS
    features["Uses_HTTPS"] = int(
        parsed.scheme.lower() == "https"
    )

    return features
# Test
url = input("Enter website URL: ")
result = extract_features(url)
print("\nExtracted Features:")
print(result)