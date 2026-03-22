import pandas as pd
import numpy as np
import tldextract
import re
import os
import sys
from urllib.parse import urlparse
from sklearn.preprocessing import MinMaxScaler
import joblib

# ═══════════════════════════════════════════════════════════════════════════════
#  53 standardised lexical features
# ═══════════════════════════════════════════════════════════════════════════════
STANDARD_FEATURES = [
    # Character counts (14)
    'num_dots', 'num_hyphens', 'num_underscores', 'num_slashes',
    'num_question_marks', 'num_equals', 'num_at', 'num_ampersand',
    'num_spaces', 'num_tilde', 'num_comma', 'num_star',
    'num_dollar', 'num_percent',
    # Length (2)
    'url_length', 'hostname_length',
    # Binary indicators (17)
    'is_shortened', 'is_ip',
    'is_brand_in_domain', 'is_brand_in_subdomain', 'is_brand_in_path',
    'has_suspicious', 'has_exe_extension',
    'has_https', 'has_non_std_port',
    'is_tld_suspicious', 'is_tld_in_path', 'is_tld_in_subdomain',
    'has_http_in_path', 'has_punycode',
    'is_subdomain_abnormal', 'has_hyphen', 'is_domain_random',
    # Substring counts (6)
    'num_double_slash', 'num_or', 'num_colon', 'num_semicolon',
    'num_www', 'num_com',
    # Domain / digit (3)
    'num_domain_parts', 'pct_digits_url', 'pct_digits_host',
    # Character-level (2)
    'num_alphanumeric', 'max_consecutive_chars',
    # Token-based (9)
    'shortest_token_url', 'shortest_token_host', 'shortest_token_path',
    'longest_token_url', 'longest_token_host', 'longest_token_path',
    'avg_token_len_url', 'avg_token_len_host', 'avg_token_len_path',
]



# ── Reference lists ───────────────────────────────────────────────────────────
SUSPICIOUS_KEYWORDS = [
    'login', 'password', 'bank', 'secure', 'account', 'update', 'verify',
    'signin', 'paypal', 'ebay', 'amazon', 'facebook', 'google', 'microsoft',
    'apple', 'support', 'help', 'contact', 'free', 'win', 'prize', 'alert',
    'warning', 'confirm', 'validate', 'logon', 'auth', 'credential', 'payment',
]

BRANDS = [
    'paypal', 'ebay', 'amazon', 'facebook', 'google', 'microsoft', 'apple',
    'netflix', 'twitter', 'instagram', 'linkedin', 'yahoo', 'chase',
    'wellsfargo', 'bankofamerica', 'citibank', 'dropbox', 'adobe', 'spotify',
    'whatsapp', 'telegram', 'snapchat',
]

URL_SHORTENERS = [
    'bit.ly', 'goo.gl', 'tinyurl.com', 't.co', 'ow.ly', 'is.gd', 'buff.ly',
    'adf.ly', 'bit.do', 'mcaf.ee', 'su.pr', 'db.tt', 'qr.ae', 'cur.lv',
    'lnkd.in', 'ity.im', 'q.gs', 'po.st', 'bc.vc', 'twitthis.com', 'u.to',
    'j.mp', 'buzurl.com', 'cutt.us', 'u.bb', 'yourls.org', 'x.co',
    'prettylinkpro.com', 'scrnch.me', 'filourl.com', 'vzturl.com', 'qr.net',
    '1url.com', 'tweez.me', 'v.gd', 'tr.im', 'link.zip.net', 'rb.gy',
]

EXE_EXTENSIONS = [
    '.exe', '.bat', '.cmd', '.scr', '.pif', '.vbs', '.js',
    '.wsf', '.msi', '.jar', '.ps1',
]

SUSPICIOUS_TLDS = [
    'zip', 'review', 'country', 'kim', 'cricket', 'science', 'work', 'party',
    'gq', 'link', 'email', 'top', 'ml', 'ga', 'cf', 'tk', 'xyz', 'pw', 'cc',
    'club', 'date', 'bid', 'stream', 'download', 'racing', 'win', 'accountant',
    'faith', 'loan', 'men', 'webcam', 'trade',
]

COMMON_TLDS = [
    'com', 'org', 'net', 'edu', 'gov', 'mil', 'int', 'info', 'biz', 'name',
    'pro', 'aero', 'coop', 'museum', 'co',
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature extraction from a raw URL
# ═══════════════════════════════════════════════════════════════════════════════

def _tokenize(text):
    """Split text on dots, slashes, and hyphens; return non-empty tokens."""
    return [t for t in re.split(r'[./\-]', text) if t]


def extract_features_from_url(url):
    """Extract the 53 lexical features from a raw URL string."""
    url = str(url).strip()
    url_lower = url.lower()

    # ── Parse URL ──
    url_for_parse = url if url.startswith(('http://', 'https://')) else 'http://' + url
    try:
        parsed = urlparse(url_for_parse)
        hostname = parsed.hostname or ''
        path = parsed.path or ''
    except Exception:
        parsed = None
        hostname = ''
        path = ''

    ext = tldextract.extract(url)
    domain = ext.domain or ''
    subdomain = ext.subdomain or ''
    suffix = ext.suffix or ''

    # ── 1-14: Single-character counts ──
    num_dots            = url.count('.')
    num_hyphens         = url.count('-')
    num_underscores     = url.count('_')
    num_slashes         = url.count('/')
    num_question_marks  = url.count('?')
    num_equals          = url.count('=')
    num_at              = url.count('@')
    num_ampersand       = url.count('&')
    num_spaces          = url.count(' ')
    num_tilde           = url.count('~')
    num_comma           = url.count(',')
    num_star            = url.count('*')
    num_dollar          = url.count('$')
    num_percent         = url.count('%')

    # ── 15-16: Length ──
    url_length      = len(url)
    hostname_length = len(hostname)

    # ── 17: URL shortener ──
    is_shortened = 1 if any(s in url_lower for s in URL_SHORTENERS) else 0

    # ── 18: IP address in hostname ──
    is_ip = 1 if re.match(r'^\d+\.\d+\.\d+\.\d+$', hostname) else 0

    # ── 19-21: Brand detection ──
    domain_lower    = domain.lower()
    subdomain_lower = subdomain.lower()
    path_lower      = path.lower()
    is_brand_in_domain    = 1 if any(b in domain_lower for b in BRANDS) else 0
    is_brand_in_subdomain = 1 if any(b in subdomain_lower for b in BRANDS) else 0
    is_brand_in_path      = 1 if any(b in path_lower for b in BRANDS) else 0

    # ── 22: Suspicious keywords ──
    has_suspicious = 1 if any(kw in url_lower for kw in SUSPICIOUS_KEYWORDS) else 0

    # ── 23: Executable file extension in path ──
    has_exe_extension = 1 if any(path_lower.endswith(e) for e in EXE_EXTENSIONS) else 0

    # ── 24: HTTPS ──
    has_https = 1 if url_lower.startswith('https') else 0

    # ── 25: Non-standard port ──
    try:
        port = parsed.port if parsed else None
        has_non_std_port = 1 if port and port not in (80, 443) else 0
    except (ValueError, AttributeError):
        has_non_std_port = 0

    # ── 26: Suspicious TLD ──
    is_tld_suspicious = 1 if suffix.lower() in SUSPICIOUS_TLDS else 0

    # ── 27-28: TLD presence ──
    tlds_to_check = set(COMMON_TLDS)
    if suffix:
        tlds_to_check.add(suffix.lower())
    is_tld_in_path = 1 if any(f'.{t}' in path_lower for t in tlds_to_check) else 0
    sub_parts = [p for p in subdomain_lower.split('.') if p]
    is_tld_in_subdomain = 1 if any(p in tlds_to_check for p in sub_parts) else 0

    # ── 29-33: Substring counts ──
    num_double_slash = url.count('//')
    num_or           = url_lower.count('or')
    num_colon        = url.count(':')
    num_semicolon    = url.count(';')
    num_www          = url_lower.count('www')
    num_com          = url_lower.count('com')

    # ── 34: HTTP in path ──
    has_http_in_path = 1 if 'http' in path_lower else 0

    # ── 35: Punycode (internationalized domain) ──
    has_punycode = 1 if 'xn--' in url_lower else 0

    # ── 36: Domain parts ──
    parts = [p for p in (subdomain.split('.') + [domain, suffix]) if p]
    num_domain_parts = len(parts)

    # ── 37-38: Digit ratios ──
    pct_digits_url  = sum(c.isdigit() for c in url) / max(url_length, 1)
    pct_digits_host = sum(c.isdigit() for c in hostname) / max(hostname_length, 1)

    # ── 39: Abnormal subdomain (multi-level and not just "www") ──
    is_subdomain_abnormal = 1 if (subdomain and subdomain != 'www'
                                  and '.' in subdomain) else 0

    # ── 40: Hyphen presence (binary) ──
    has_hyphen = 1 if '-' in url else 0

    # ── 41: Random domain (Shannon entropy > 3.5) ──
    if domain:
        freq = {}
        for c in domain:
            freq[c] = freq.get(c, 0) + 1
        n = len(domain)
        entropy = -sum((f / n) * np.log2(f / n) for f in freq.values())
        is_domain_random = 1 if entropy > 3.5 else 0
    else:
        is_domain_random = 0

    # ── 42: Alphanumeric character count ──
    num_alphanumeric = sum(c.isalnum() for c in url)

    # ── 43: Max consecutive repeated characters ──
    max_consecutive_chars = 0
    if url:
        run = 1
        for i in range(1, len(url)):
            if url[i] == url[i - 1]:
                run += 1
            else:
                run = 1
            if run > max_consecutive_chars:
                max_consecutive_chars = run
        max_consecutive_chars = max(max_consecutive_chars, 1)

    # ── 44-52: Token-based features ──
    url_tokens  = _tokenize(url)
    host_tokens = _tokenize(hostname)
    path_tokens = _tokenize(path)

    def _shortest(tokens):
        return min((len(t) for t in tokens), default=0)
    def _longest(tokens):
        return max((len(t) for t in tokens), default=0)
    def _average(tokens):
        return float(np.mean([len(t) for t in tokens])) if tokens else 0.0

    shortest_token_url  = _shortest(url_tokens)
    shortest_token_host = _shortest(host_tokens)
    shortest_token_path = _shortest(path_tokens)
    longest_token_url   = _longest(url_tokens)
    longest_token_host  = _longest(host_tokens)
    longest_token_path  = _longest(path_tokens)
    avg_token_len_url   = _average(url_tokens)
    avg_token_len_host  = _average(host_tokens)
    avg_token_len_path  = _average(path_tokens)

    return {
        'num_dots': num_dots,
        'num_hyphens': num_hyphens,
        'num_underscores': num_underscores,
        'num_slashes': num_slashes,
        'num_question_marks': num_question_marks,
        'num_equals': num_equals,
        'num_at': num_at,
        'num_ampersand': num_ampersand,
        'num_spaces': num_spaces,
        'num_tilde': num_tilde,
        'num_comma': num_comma,
        'num_star': num_star,
        'num_dollar': num_dollar,
        'num_percent': num_percent,
        'url_length': url_length,
        'hostname_length': hostname_length,
        'is_shortened': is_shortened,
        'is_ip': is_ip,
        'is_brand_in_domain': is_brand_in_domain,
        'is_brand_in_subdomain': is_brand_in_subdomain,
        'is_brand_in_path': is_brand_in_path,
        'has_suspicious': has_suspicious,
        'has_exe_extension': has_exe_extension,
        'has_https': has_https,
        'has_non_std_port': has_non_std_port,
        'is_tld_suspicious': is_tld_suspicious,
        'is_tld_in_path': is_tld_in_path,
        'is_tld_in_subdomain': is_tld_in_subdomain,
        'has_http_in_path': has_http_in_path,
        'has_punycode': has_punycode,
        'is_subdomain_abnormal': is_subdomain_abnormal,
        'has_hyphen': has_hyphen,
        'is_domain_random': is_domain_random,
        'num_double_slash': num_double_slash,
        'num_or': num_or,
        'num_colon': num_colon,
        'num_semicolon': num_semicolon,
        'num_www': num_www,
        'num_com': num_com,
        'num_domain_parts': num_domain_parts,
        'pct_digits_url': pct_digits_url,
        'pct_digits_host': pct_digits_host,
        'num_alphanumeric': num_alphanumeric,
        'max_consecutive_chars': max_consecutive_chars,
        'shortest_token_url': shortest_token_url,
        'shortest_token_host': shortest_token_host,
        'shortest_token_path': shortest_token_path,
        'longest_token_url': longest_token_url,
        'longest_token_host': longest_token_host,
        'longest_token_path': longest_token_path,
        'avg_token_len_url': avg_token_len_url,
        'avg_token_len_host': avg_token_len_host,
        'avg_token_len_path': avg_token_len_path,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset processors
# ═══════════════════════════════════════════════════════════════════════════════

def process_raw_csv(file_path, url_col='URL', label_col='Label'):
    """Process raw CSV with URL and label columns."""
    df = pd.read_csv(file_path)
    features_list = []
    for idx, row in df.iterrows():
        features = extract_features_from_url(row[url_col])
        features['id'] = idx + 1
        label = str(row[label_col]).lower()
        if label == 'bad':
            features['label'] = 'Phishing'
        elif label == 'good':
            features['label'] = 'Not Phishing'
        else:
            features['label'] = 'Phishing'  # default
        features_list.append(features)
    return pd.DataFrame(features_list)


def process_index_csv(file_path):
    """Process index.csv with rec_id, url, result columns."""
    df = pd.read_csv(file_path)
    features_list = []
    for idx, row in df.iterrows():
        features = extract_features_from_url(row['url'])
        features['id'] = row['rec_id']
        features['label'] = 'Phishing' if row['result'] == 1 else 'Not Phishing'
        features_list.append(features)
    return pd.DataFrame(features_list)


def process_defined_csv(file_path):
    """
    Process defined CSV datasets [3] and [6] (Hannousse / Shashwat Tiwari style).
    Maps their column names to the 53 standard features.
    """
    df = pd.read_csv(file_path)

    # Source column → standard feature name
    column_mapping = {
        'nb_dots':            'num_dots',
        'nb_hyphens':         'num_hyphens',
        'nb_underscore':      'num_underscores',
        'nb_slash':           'num_slashes',
        'nb_qm':              'num_question_marks',
        'nb_eq':              'num_equals',
        'nb_at':              'num_at',
        'nb_and':             'num_ampersand',
        'nb_space':           'num_spaces',
        'nb_tilde':           'num_tilde',
        'nb_comma':           'num_comma',
        'nb_star':            'num_star',
        'nb_dollar':          'num_dollar',
        'nb_percent':         'num_percent',
        'length_url':         'url_length',
        'length_hostname':    'hostname_length',
        'shortening_service': 'is_shortened',
        'ip':                 'is_ip',
        'domain_in_brand':    'is_brand_in_domain',
        'brand_in_subdomain': 'is_brand_in_subdomain',
        'brand_in_path':      'is_brand_in_path',
        'phish_hints':        'has_suspicious',
        'path_extension':     'has_exe_extension',
        'https_token':        'has_https',          # needs inversion
        'port':               'has_non_std_port',
        'suspecious_tld':     'is_tld_suspicious',
        'tld_in_path':        'is_tld_in_path',
        'tld_in_subdomain':   'is_tld_in_subdomain',
        'http_in_path':       'has_http_in_path',
        'punycode':           'has_punycode',
        'abnormal_subdomain': 'is_subdomain_abnormal',
        'prefix_suffix':      'has_hyphen',
        'random_domain':      'is_domain_random',
        'nb_dslash':          'num_double_slash',
        'nb_or':              'num_or',
        'nb_colon':           'num_colon',
        'nb_semicolumn':      'num_semicolon',
        'nb_www':             'num_www',
        'nb_com':             'num_com',
        'nb_subdomains':      'num_domain_parts',
        'ratio_digits_url':   'pct_digits_url',
        'ratio_digits_host':  'pct_digits_host',
        'char_repeat':        'max_consecutive_chars',
        'shortest_words_raw': 'shortest_token_url',
        'shortest_word_host': 'shortest_token_host',
        'shortest_word_path': 'shortest_token_path',
        'longest_words_raw':  'longest_token_url',
        'longest_word_host':  'longest_token_host',
        'longest_word_path':  'longest_token_path',
        'avg_words_raw':      'avg_token_len_url',
        'avg_word_host':      'avg_token_len_host',
        'avg_word_path':      'avg_token_len_path',
        'status':             'label',
    }

    available = [c for c in df.columns if c in column_mapping]
    result = df[available].copy()
    result.rename(columns=column_mapping, inplace=True)

    # Compute num_alphanumeric from the URL column (not in source columns)
    result['num_alphanumeric'] = df['url'].apply(
        lambda u: sum(c.isalnum() for c in str(u))
    )

    # Invert https_token: original 1 = http only → has_https = 0
    if 'has_https' in result.columns:
        result['has_https'] = 1 - result['has_https']

    # Binarize has_suspicious (phish_hints count → binary)
    if 'has_suspicious' in result.columns:
        result['has_suspicious'] = (result['has_suspicious'] > 0).astype(int)

    # Add sequential id
    result['id'] = result.index + 1

    # Standardize label
    if result['label'].dtype == 'object':
        result['label'] = result['label'].map({
            'legitimate': 'Not Phishing', 'phishing': 'Phishing'
        })
    else:
        result['label'] = result['label'].map({
            0: 'Not Phishing', 1: 'Phishing'
        })

    return result[['id'] + STANDARD_FEATURES + ['label']]


def _process_fadil_tan(df):
    """Shared processing for Fadil / Tan style datasets ([4] and [b3]).
    Maps the available columns to the 53 standard features.
    Columns that have no counterpart will be zero-filled by load_dataset.
    """
    column_mapping = {
        'NumDots':            'num_dots',
        'NumDash':            'num_hyphens',
        'NumUnderscore':      'num_underscores',
        'AtSymbol':           'num_at',
        'NumAmpersand':       'num_ampersand',
        'TildeSymbol':        'num_tilde',
        'NumPercent':         'num_percent',
        'UrlLength':          'url_length',
        'HostnameLength':     'hostname_length',
        'IpAddress':          'is_ip',
        'DomainInSubdomains': 'is_brand_in_subdomain',
        'DomainInPaths':      'is_brand_in_path',
        'NumSensitiveWords':  'has_suspicious',
        'NoHttps':            'has_https',
        'DoubleSlashInPath':  'num_double_slash',
        'EmbeddedBrandName':  'is_brand_in_domain',
        'SubdomainLevel':     'num_domain_parts',
        'RandomString':       'is_domain_random',
        'NumNumericChars':    'num_alphanumeric',
        'CLASS_LABEL':        'label',
    }

    available = [c for c in df.columns if c in column_mapping]
    result = df[available].copy()
    result.rename(columns=column_mapping, inplace=True)

    # Invert NoHttps: original 1 = no https -> has_https = 0
    if 'has_https' in result.columns:
        result['has_https'] = 1 - result['has_https']

    # Binarize has_suspicious (count -> binary)
    if 'has_suspicious' in result.columns:
        result['has_suspicious'] = (result['has_suspicious'] > 0).astype(int)

    # Derive has_hyphen from num_hyphens
    if 'num_hyphens' in result.columns:
        result['has_hyphen'] = (result['num_hyphens'] > 0).astype(int)

    # Standardize label
    if result['label'].dtype == 'object':
        result['label'] = result['label'].map(
            {'0': 'Not Phishing', '1': 'Phishing'}
        )
    else:
        result['label'] = result['label'].map(
            {0: 'Not Phishing', 1: 'Phishing'}
        )

    return result


def process_fadil_tan_csv(file_path):
    """Process [4] Mohamad Fadil (Kaggle, Defined) CSV."""
    df = pd.read_csv(file_path)
    return _process_fadil_tan(df)


def process_fadil_tan_arff(file_path):
    """Process [b3] Tan (Mendeley, Defined) ARFF."""
    attrs = []
    data_started = False
    rows = []
    with open(file_path, 'r') as f:
        for line in f:
            stripped = line.strip()
            if data_started:
                if stripped and not stripped.startswith('%'):
                    rows.append(stripped.split(','))
            elif stripped.lower().startswith('@attribute'):
                parts = stripped.split()
                attrs.append(parts[1])
            elif stripped.lower() == '@data':
                data_started = True
    df = pd.DataFrame(rows, columns=attrs)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return _process_fadil_tan(df)


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset auto-discovery
# ═══════════════════════════════════════════════════════════════════════════════

# Maps bracketed file prefix → processor key.
# 'defined' sub-folder files use column-based dispatch (see _detect_processor).
# 'raw' sub-folder files extract all 53 features from the URL column.

# Processor key → (callable, description)
PROCESSOR_MAP = {
    'defined_csv':    process_defined_csv,
    'raw_csv':        lambda f: process_raw_csv(f, 'URL', 'Label'),
    'index_csv':      process_index_csv,
    'fadil_tan_csv':  process_fadil_tan_csv,
    'fadil_tan_arff': process_fadil_tan_arff,
}

# ── Label mapping tables for raw datasets ─────────────────────────────────────
# Each entry: (url_col, label_col, label_map)
# label_map converts raw label values → 'Phishing' / 'Not Phishing'
_RAW_LABEL_SPECS = {
    # col_name → {value: standardised_label}
    'Label':      {'bad': 'Phishing', 'good': 'Not Phishing'},
    'label':      {'phishing': 'Phishing', 'benign': 'Not Phishing',
                   1: 'Phishing', 0: 'Not Phishing'},
    'result':     {1: 'Phishing', 0: 'Not Phishing'},
    'ClassLabel': {1: 'Phishing', 1.0: 'Phishing',
                   0: 'Not Phishing', 0.0: 'Not Phishing'},
    'status':     {1: 'Phishing', 0: 'Not Phishing'},
    'type':       {'phishing': 'Phishing', 'legitimate': 'Not Phishing'},
}

# Columns that identify the Hannousse / Shashwat Tiwari defined format
_DEFINED_MARKER_COLS = {'nb_dots', 'length_url', 'nb_hyphens'}


def _detect_processor_and_build(subfolder, file_path):
    """Return (processor_key, extra_kwargs) for a discovered file."""
    fname = os.path.basename(file_path)

    # ARFF files → fadil_tan_arff
    if fname.lower().endswith('.arff'):
        return 'fadil_tan_arff', {}

    if subfolder == 'defined':
        # Peek at columns to decide
        cols = set(pd.read_csv(file_path, nrows=0).columns)
        if cols & _DEFINED_MARKER_COLS:
            return 'defined_csv', {}
        else:
            return 'fadil_tan_csv', {}

    # subfolder == 'raw'  →  auto-detect URL + label columns
    cols = list(pd.read_csv(file_path, nrows=0).columns)
    # Find URL column (case-insensitive)
    url_col = next((c for c in cols if c.lower() == 'url'), None)
    if url_col is None:
        return None, {}  # can't process without a URL column

    # Special case: index.csv style (rec_id, url, result)
    cols_set = set(cols)
    if {'rec_id', 'url', 'result'} <= cols_set:
        return 'index_csv', {}

    # Find label column
    for candidate in _RAW_LABEL_SPECS:
        if candidate in cols:
            return 'raw_auto', {'url_col': url_col, 'label_col': candidate}

    return None, {}


def _process_raw_auto(file_path, url_col, label_col):
    """Generic raw processor: extract 53 features from URL, map labels."""
    df = pd.read_csv(file_path)
    label_map = _RAW_LABEL_SPECS[label_col]

    features_list = []
    for idx, row in df.iterrows():
        features = extract_features_from_url(row[url_col])
        features['id'] = idx + 1
        raw_label = row[label_col]
        # Try exact match first, then lowercase string match
        if raw_label in label_map:
            features['label'] = label_map[raw_label]
        else:
            str_label = str(raw_label).strip().lower()
            mapped = None
            for k, v in label_map.items():
                if str(k).strip().lower() == str_label:
                    mapped = v
                    break
            features['label'] = mapped if mapped else 'Phishing'
        features_list.append(features)
    return pd.DataFrame(features_list)


# Register the auto processor
PROCESSOR_MAP['raw_auto'] = None  # placeholder; called with kwargs


def discover_datasets(datasets_dir='datasets'):
    """Scan datasets/ for files matching the [ID] prefix pattern.

    Returns a list of dataset entry dicts sorted by ID, e.g.:
        {'id': '3', 'name': '...', 'path': 'defined/...', 'processor': '...',
         'proc_kwargs': {...}}
    """
    entries = []
    pattern = re.compile(r'^\[([^\]]+)\]\s*(.+)$')

    for subfolder in ('defined', 'raw'):
        folder = os.path.join(datasets_dir, subfolder)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            m = pattern.match(fname)
            if not m:
                continue
            file_id = m.group(1)        # e.g. '3', 'c1', 'b3'
            remainder = m.group(2)      # rest of filename
            full_path = os.path.join(folder, fname)

            proc_key, kwargs = _detect_processor_and_build(subfolder, full_path)
            if proc_key is None:
                continue

            # Build a readable name from filename (strip extension + pct info)
            name_clean = re.sub(r'\s*\([^)]*\)\s*', ' ', remainder)
            name_clean = re.sub(r'\.(csv|arff)$', '', name_clean, flags=re.I).strip()
            display_name = f'[{file_id}] {name_clean} ({subfolder.title()})'

            entries.append({
                'id': file_id,
                'name': display_name,
                'path': f'{subfolder}/{fname}',
                'processor': proc_key,
                'proc_kwargs': kwargs,
            })

    # Sort: numeric IDs first (ascending), then alphanumeric IDs
    def _sort_key(e):
        try:
            return (0, int(e['id']), '')
        except ValueError:
            return (1, 0, e['id'])

    entries.sort(key=_sort_key)

    # Assign sequential display numbers
    for i, entry in enumerate(entries, 1):
        entry['display_id'] = i

    return entries


# ═══════════════════════════════════════════════════════════════════════════════
#  Pipeline helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset(entry, datasets_dir='datasets'):
    """Load a single dataset, extract features, zero-fill missing columns."""
    file_path = os.path.join(datasets_dir, entry['path'])
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    proc_key = entry['processor']
    kwargs = entry.get('proc_kwargs', {})

    if proc_key == 'raw_auto':
        df = _process_raw_auto(file_path, **kwargs)
    else:
        processor = PROCESSOR_MAP[proc_key]
        df = processor(file_path)

    # Zero-fill any missing standard features
    for col in STANDARD_FEATURES:
        if col not in df.columns:
            df[col] = 0

    df = df[STANDARD_FEATURES + ['label']]

    before = len(df)
    df = df.dropna(subset=STANDARD_FEATURES + ['label']).reset_index(drop=True)
    dropped = before - len(df)

    print(f"  [{entry['display_id']}] {entry['name']}: {len(df)} rows", end='')
    if dropped:
        print(f" ({dropped} missing dropped)", end='')
    print()

    return df


def load_and_merge(entries, role_name, datasets_dir='datasets'):
    """Load multiple datasets for a role and merge them into one DataFrame."""
    print(f"\nLoading {role_name} datasets:")
    dfs = []
    for entry in entries:
        df = load_dataset(entry, datasets_dir)
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)
    print(f"  Merged {role_name} set: {len(merged)} rows")
    return merged


def save_split(split_df, split_name, path_dir):
    """Save a split CSV with re-numbered ids."""
    out = split_df.copy()
    out.insert(0, 'id', range(1, len(out) + 1))
    out = out[['id'] + STANDARD_FEATURES + ['label']]
    out.to_csv(os.path.join(path_dir, f'{split_name}.csv'), index=False)
    return len(out)


def run_pipeline(train_entries, val_entries, test_entries,
                 datasets_dir='datasets', output_dir='processed_datasets'):
    """
    Cross-dataset preprocessing pipeline (53 features).

    1. Load and merge each role (train / val / test), zero-filling missing
       columns so every dataset has the same 53 features.
    2. Remove duplicates from training set.
    3. Fit a MinMaxScaler on the 53 training features and save it.
    4. Scale all splits with that scaler.
    5. Save the scaled CSVs and scaler — both Path A (Chi-square + RF) and
       Path B (CNN-BiLSTM) consume the same scaled representation.
    """

    # ── Step 1: Load and merge each role ──
    train_df = load_and_merge(train_entries, 'TRAINING', datasets_dir)
    val_df   = load_and_merge(val_entries,   'VALIDATION', datasets_dir)
    test_df  = load_and_merge(test_entries,  'TESTING', datasets_dir)

    # ── Step 2: Remove duplicates from training set ──
    before = len(train_df)
    train_df = train_df.drop_duplicates(
        subset=STANDARD_FEATURES + ['label']
    ).reset_index(drop=True)
    removed = before - len(train_df)
    if removed:
        print(f"\nDuplicate removal: removed {removed} rows from training set "
              f"({len(train_df)} remaining)")

    # ── Step 3: Build output folder name from dataset IDs ──
    train_ids = ','.join(str(e['display_id']) for e in train_entries)
    val_ids   = ','.join(str(e['display_id']) for e in val_entries)
    test_ids  = ','.join(str(e['display_id']) for e in test_entries)
    run_folder = f'train[{train_ids}]_val[{val_ids}]_test[{test_ids}]'
    run_dir = os.path.join(output_dir, run_folder)
    os.makedirs(run_dir, exist_ok=True)

    # ── Step 4: Fit MinMaxScaler on training, apply to all splits ──
    scaler = MinMaxScaler()
    train_scaled = train_df.copy()
    val_scaled   = val_df.copy()
    test_scaled  = test_df.copy()

    train_scaled[STANDARD_FEATURES] = scaler.fit_transform(
        train_df[STANDARD_FEATURES])
    val_scaled[STANDARD_FEATURES] = scaler.transform(
        val_df[STANDARD_FEATURES])
    test_scaled[STANDARD_FEATURES] = scaler.transform(
        test_df[STANDARD_FEATURES])
    print(f"\nMinMaxScaler fit on training set, applied to all splits")

    # Save scaler for reuse in training / inference
    scaler_path = os.path.join(run_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # Report class distribution
    class_counts = train_scaled['label'].value_counts()
    print(f"Training class distribution: {dict(class_counts)}")

    # ── Step 5: Save scaled CSVs ──
    n_train = save_split(train_scaled, 'train', run_dir)
    n_val   = save_split(val_scaled,   'val',   run_dir)
    n_test  = save_split(test_scaled,  'test',  run_dir)
    print(f"\nSaved to {run_dir}")
    print(f"  train={n_train}, val={n_val}, test={n_test}")


def parse_ids(text, available):
    """Parse comma-separated display IDs into a list of available entries."""
    ids = [int(x.strip()) for x in text.split(',')]
    available_ids = {ds['display_id'] for ds in available}
    entries = [ds for ds in available if ds['display_id'] in ids]
    bad = [i for i in ids if i not in available_ids]
    if bad:
        print(f"  Warning: ID(s) {bad} not in the available list, skipped.")
    return entries


def get_input(prompt, arg_value=None):
    """Get input from CLI arg or interactive prompt."""
    if arg_value is not None:
        print(f"{prompt}{arg_value}")
        return arg_value
    return input(prompt).strip()


def main():
    print("\n===== Cross-Dataset Preprocessing Pipeline =====\n")
    available = discover_datasets()
    if not available:
        print("No datasets found in the datasets/ directory.")
        return
    print("Available datasets:")
    for ds in available:
        print(f"  [{ds['display_id']}] {ds['name']}")
    print()

    # CLI usage: python preprocess.py <train_ids> <val_ids> <test_ids>
    # Example:   python preprocess.py 1,3 2 4
    # Interactive: prompts for each role
    cli_train = sys.argv[1].strip() if len(sys.argv) > 1 else None
    cli_val   = sys.argv[2].strip() if len(sys.argv) > 2 else None
    cli_test  = sys.argv[3].strip() if len(sys.argv) > 3 else None

    try:
        train_text = get_input("Enter TRAINING dataset IDs (comma-separated): ", cli_train)
        val_text   = get_input("Enter VALIDATION dataset IDs (comma-separated): ", cli_val)
        test_text  = get_input("Enter TESTING dataset IDs (comma-separated): ", cli_test)

        train_entries = parse_ids(train_text, available)
        val_entries   = parse_ids(val_text, available)
        test_entries  = parse_ids(test_text, available)
    except ValueError:
        print("Invalid input. Enter numbers separated by commas.")
        return

    # Validate: each role must have at least one dataset
    if not train_entries:
        print("No valid training dataset IDs.")
        return
    if not val_entries:
        print("No valid validation dataset IDs.")
        return
    if not test_entries:
        print("No valid testing dataset IDs.")
        return

    # Validate: no dataset used in more than one role
    train_ids = {e['display_id'] for e in train_entries}
    val_ids   = {e['display_id'] for e in val_entries}
    test_ids  = {e['display_id'] for e in test_entries}
    overlap = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
    if overlap:
        print(f"Error: Dataset(s) {overlap} assigned to multiple roles. Each dataset must be in exactly one role.")
        return

    print(f"\nConfiguration:")
    print(f"  Training:   {[e['name'] for e in train_entries]}")
    print(f"  Validation: {[e['name'] for e in val_entries]}")
    print(f"  Testing:    {[e['name'] for e in test_entries]}")

    try:
        run_pipeline(train_entries, val_entries, test_entries)
    except Exception as e:
        print(f"\nError: {e}")
        raise

    print("\nDone.")


if __name__ == '__main__':
    main()
