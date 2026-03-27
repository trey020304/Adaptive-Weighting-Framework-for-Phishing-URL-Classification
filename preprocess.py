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


def run_homogeneous_pipeline(entries, train_ratio=0.70, val_ratio=0.15,
                             datasets_dir='datasets',
                             output_dir='processed_datasets',
                             seed=42):
    """
    Homogeneous splitting pipeline: each dataset is independently split
    into train/val/test (stratified by label), then all train portions are
    merged, all val portions are merged, all test portions are merged.

    This ensures every split contains samples from every dataset, so
    feature-label relationships remain consistent across splits.
    """
    from sklearn.model_selection import train_test_split

    train_dfs, val_dfs, test_dfs = [], [], []

    print("\nLoading and splitting each dataset (stratified per-dataset):")
    for entry in entries:
        df = load_dataset(entry, datasets_dir)
        labels = df['label']

        # First split: train vs (val+test)
        val_test_ratio = 1.0 - train_ratio
        df_train, df_valtest = train_test_split(
            df, test_size=val_test_ratio, stratify=labels, random_state=seed
        )

        # Second split: val vs test from the remainder
        test_frac_of_remainder = (1.0 - train_ratio - val_ratio) / val_test_ratio
        df_val, df_test = train_test_split(
            df_valtest, test_size=test_frac_of_remainder,
            stratify=df_valtest['label'], random_state=seed
        )

        print(f"    [{entry['display_id']}] train={len(df_train)}, "
              f"val={len(df_val)}, test={len(df_test)}")

        train_dfs.append(df_train)
        val_dfs.append(df_val)
        test_dfs.append(df_test)

    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df   = pd.concat(val_dfs,   ignore_index=True)
    test_df  = pd.concat(test_dfs,  ignore_index=True)

    # Shuffle each split
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df   = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df  = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"\n  Merged: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Remove duplicates from training set
    before = len(train_df)
    train_df = train_df.drop_duplicates(
        subset=STANDARD_FEATURES + ['label']
    ).reset_index(drop=True)
    removed = before - len(train_df)
    if removed:
        print(f"  Duplicate removal: removed {removed} rows from training set "
              f"({len(train_df)} remaining)")

    # Build output folder
    ds_ids = ','.join(str(e['display_id']) for e in entries)
    run_folder = f'homogeneous[{ds_ids}]'
    run_dir = os.path.join(output_dir, run_folder)
    os.makedirs(run_dir, exist_ok=True)

    # Fit MinMaxScaler on training, apply to all splits
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
    print(f"\n  MinMaxScaler fit on training set, applied to all splits")

    scaler_path = os.path.join(run_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved to {scaler_path}")

    # Report class distribution
    for name, split in [('Train', train_scaled), ('Val', val_scaled), ('Test', test_scaled)]:
        counts = split['label'].value_counts()
        total = len(split)
        phish = counts.get('Phishing', 0)
        print(f"  {name}: {total} rows — "
              f"Phishing={phish} ({100*phish/total:.1f}%), "
              f"Not Phishing={total-phish} ({100*(total-phish)/total:.1f}%)")

    # Save scaled CSVs
    n_train = save_split(train_scaled, 'train', run_dir)
    n_val   = save_split(val_scaled,   'val',   run_dir)
    n_test  = save_split(test_scaled,  'test',  run_dir)
    print(f"\n  Saved to {run_dir}")
    print(f"  train={n_train}, val={n_val}, test={n_test}")

    return run_folder


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

    # ── Homogeneous mode ──
    # CLI: python preprocess.py --homogeneous 3,5,7,8,9
    if len(sys.argv) > 1 and sys.argv[1] == '--homogeneous':
        ids_text = sys.argv[2].strip() if len(sys.argv) > 2 else None
        if ids_text is None:
            ids_text = input("Enter dataset IDs to include (comma-separated): ").strip()
        entries = parse_ids(ids_text, available)
        if not entries:
            print("No valid dataset IDs.")
            return
        print(f"Homogeneous mode — datasets: {[e['name'] for e in entries]}")
        try:
            run_folder = run_homogeneous_pipeline(entries)
            print(f"\nDone. Set split_dir: \"{run_folder}\" in config.yaml")
        except Exception as e:
            print(f"\nError: {e}")
            raise
        return

    # ── Cross-dataset mode (original) ──
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Preprocessing functions for external pipelines (Aqilla / Princeton)
# ═══════════════════════════════════════════════════════════════════════════════

def load_config(config_path="config.yaml"):
    """Load the YAML configuration file."""
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Unified dataset loader ────────────────────────────────────────────────────

def load_any_dataset(dataset_id, cfg):
    """Load any dataset by ID from the registry in config.yaml.

    For 'defined' datasets: reads CSV as-is, normalises URL/label columns.
    For 'raw' datasets: reads URL + label, extracts 53 standard features.

    Returns a DataFrame with columns: url, <53 standard features>, status
    where status is 'phishing' or 'legitimate'.
    """
    ds_cfg = cfg["datasets"][str(dataset_id)]
    data_path = ds_cfg["path"]
    ds_type = ds_cfg["type"]
    url_col = ds_cfg["url_col"]
    label_col = ds_cfg["label_col"]
    phishing_values = ds_cfg["phishing_values"]

    print(f"Loading dataset [{dataset_id}] from {data_path} ...")
    df = pd.read_csv(data_path)
    print(f"  Shape: {df.shape}")

    # Drop rows with missing URL or label
    df = df.dropna(subset=[url_col, label_col])

    # Normalise label → 'phishing' / 'legitimate'
    raw_labels = df[label_col]
    # Convert to comparable types (handle int/float/string)
    phishing_set = set()
    for v in phishing_values:
        phishing_set.add(v)
        if isinstance(v, (int, float)):
            phishing_set.add(int(v))
            phishing_set.add(float(v))
            phishing_set.add(str(int(v)))
        else:
            phishing_set.add(str(v).lower())

    def _map_label(val):
        if val in phishing_set:
            return "phishing"
        if isinstance(val, str) and val.lower() in phishing_set:
            return "phishing"
        return "legitimate"

    df["status"] = raw_labels.apply(_map_label)

    # Normalise URL column name
    if url_col != "url":
        df = df.rename(columns={url_col: "url"})

    # Drop duplicates
    dupes = df.duplicated().sum()
    print(f"  Duplicates dropped: {dupes}")
    df = df.drop_duplicates().reset_index(drop=True)

    print(f"  Label distribution:\n{df['status'].value_counts().to_string()}")

    if ds_type == "defined":
        # Defined datasets already have feature columns — return as-is
        print(f"  Type: defined (pre-extracted features)")
        return df

    # Raw datasets: extract 53 standard features from URLs
    print(f"  Type: raw — extracting 53 lexical features from URLs ...")
    features_list = []
    urls = df["url"].astype(str).values
    total = len(urls)
    for i, url in enumerate(urls):
        if (i + 1) % 10000 == 0 or (i + 1) == total:
            print(f"    Processed {i + 1}/{total} URLs ...", end="\r")
        features_list.append(extract_features_from_url(url))
    print()  # newline after progress

    feat_df = pd.DataFrame(features_list)
    feat_df["url"] = df["url"].values
    feat_df["status"] = df["status"].values

    # Reorder: url first, then 53 features, then status
    cols = ["url"] + STANDARD_FEATURES + ["status"]
    feat_df = feat_df[cols]

    print(f"  Final shape: {feat_df.shape}")
    return feat_df


# ── Aqilla pipeline preprocessing ─────────────────────────────────────────────

def aqilla_load_and_clean(cfg):
    """Load dataset via registry, check missing/duplicates, encode labels.
    Returns (df, feature_cols, X, y, label_encoder).
    """
    dataset_id = cfg["aqilla"]["dataset_id"]
    df = load_any_dataset(dataset_id, cfg)

    # Encode target label
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["status"])  # legitimate=0, phishing=1
    print(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Missing values
    missing = df.isnull().sum().sum()
    print(f"Missing values: {missing}")
    if missing > 0:
        df = df.fillna(0)

    # Separate features and target
    drop_cols = ["url", "status", "label"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    y = df["label"].copy()

    return df, feature_cols, X, y, le


def aqilla_feature_engineering(X):
    """Create augmented features: domain_age_adequate, is_trusted_domain,
    similar_to_trusted. Returns modified X."""
    X = X.copy()
    if "domain_age" in X.columns:
        X["domain_age_adequate"] = (X["domain_age"] > 365).astype(int)
    if {"google_index", "dns_record"}.issubset(X.columns):
        X["is_trusted_domain"] = (
            (X["google_index"] == 1) & (X["dns_record"] == 1)
        ).astype(int)
    brand_cols = [c for c in ["domain_in_brand", "brand_in_subdomain", "brand_in_path"]
                  if c in X.columns]
    if brand_cols:
        X["similar_to_trusted"] = (X[brand_cols].sum(axis=1) > 0).astype(int)
    return X


def aqilla_feature_selection(X, y, cfg):
    """Correlation removal + chi-square selection. Returns (X_selected, chi2_df)."""
    corr_thresh = cfg["aqilla"]["correlation_threshold"]
    p_thresh = cfg["aqilla"]["chi2_p_threshold"]

    # Correlation analysis
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    high_corr_cols = [col for col in upper_tri.columns
                      if any(upper_tri[col] > corr_thresh)]
    print(f"\nHighly correlated features removed ({len(high_corr_cols)}): {high_corr_cols}")
    X = X.drop(columns=high_corr_cols)

    # Chi-square test
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.feature_selection import chi2
    scaler_chi = MinMaxScaler()
    X_chi = pd.DataFrame(scaler_chi.fit_transform(X), columns=X.columns)
    chi2_scores, p_values = chi2(X_chi, y)
    chi2_df = pd.DataFrame({
        "feature": X.columns, "chi2": chi2_scores, "p_value": p_values
    }).sort_values("chi2", ascending=False)

    sig_features = chi2_df[chi2_df["p_value"] < p_thresh]["feature"].tolist()
    print(f"Features after chi-square selection (p<{p_thresh}): "
          f"{len(sig_features)} of {X.shape[1]}")
    X = X[sig_features]
    return X, chi2_df


def aqilla_normalize(X):
    """Min-Max normalization then Z-score standardization. Returns (X_final, mm_scaler, z_scaler)."""
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    mm_scaler = MinMaxScaler()
    X_scaled = mm_scaler.fit_transform(X)
    z_scaler = StandardScaler()
    X_scaled = z_scaler.fit_transform(X_scaled)
    X_final = pd.DataFrame(X_scaled, columns=X.columns)
    return X_final, mm_scaler, z_scaler


def aqilla_pca(X_final, cfg):
    """PCA dimensionality reduction. Returns (X_pca, pca_model)."""
    from sklearn.decomposition import PCA
    seed = cfg["random_seed"]
    variance = cfg["aqilla"]["pca_variance_retained"]
    pca = PCA(n_components=variance, random_state=seed)
    X_pca = pca.fit_transform(X_final)
    print(f"PCA components ({variance*100:.0f}% variance): {pca.n_components_}")
    return X_pca, pca


def aqilla_preprocess(cfg):
    """Full Aqilla preprocessing pipeline. Returns dict with all needed objects."""
    df, feature_cols, X, y, le = aqilla_load_and_clean(cfg)

    print(f"\nFeature matrix shape before selection: {X.shape}")
    X = aqilla_feature_engineering(X)
    print(f"Feature matrix shape after augmentation: {X.shape}")

    X, chi2_df = aqilla_feature_selection(X, y, cfg)
    print(f"Feature matrix shape after selection: {X.shape}")

    X_final, mm_scaler, z_scaler = aqilla_normalize(X)
    print(f"\nFinal feature matrix shape: {X_final.shape}")

    return {
        "X_final": X_final,
        "y": y,
        "le": le,
        "chi2_df": chi2_df,
        "feature_names": X.columns.tolist(),
        "mm_scaler": mm_scaler,
        "z_scaler": z_scaler,
    }


# ── Princeton pipeline preprocessing (char-level URL) ─────────────────────────

def _build_char_vocab(urls_clean):
    """Build character vocabulary from cleaned URL list."""
    chars = sorted(set("".join(urls_clean)))
    char_to_idx = {c: i + 1 for i, c in enumerate(chars)}  # 0 = padding
    vocab_size = len(char_to_idx) + 1
    return char_to_idx, vocab_size


def _encode_url_chars(url, mapping, maxlen):
    """Encode a single URL string to a fixed-length integer sequence."""
    encoded = [mapping.get(c, 0) for c in url[:maxlen]]
    if len(encoded) < maxlen:
        encoded += [0] * (maxlen - len(encoded))
    return encoded


def princeton_preprocess(cfg):
    """Full Princeton (original) preprocessing. Returns dict."""
    pcfg = cfg["princeton"]
    seed = cfg["random_seed"]
    data_path = pcfg["dataset_path"]
    max_len = pcfg["char_max_len"]

    df = pd.read_csv(data_path)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Loading dataset from {data_path} ...")
    print(f"  Shape: {df.shape}")
    print(f"  Label distribution:\n{df['status'].value_counts().to_string()}\n")

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    labels = le.fit_transform(df["status"])
    print(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Char-level encoding
    print("\nPreprocessing URLs (character-level tokenization) ...")
    urls = df["url"].astype(str).values
    urls_clean = [u.lower().strip() for u in urls]

    char_to_idx, vocab_size = _build_char_vocab(urls_clean)
    print(f"  Vocabulary size (unique chars): {vocab_size}")
    print(f"  Max sequence length: {max_len}")

    X_encoded = np.array([_encode_url_chars(u, char_to_idx, max_len)
                          for u in urls_clean])
    y = np.array(labels, dtype=np.float32)
    print(f"  Encoded shape: {X_encoded.shape}")

    # Train / val / test split
    from sklearn.model_selection import train_test_split
    test_size = pcfg["test_size"]
    val_size = pcfg["val_size"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=seed, stratify=y_train
    )

    print(f"\n  Train : {X_train.shape[0]}")
    print(f"  Val   : {X_val.shape[0]}")
    print(f"  Test  : {X_test.shape[0]}")

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "le": le,
        "vocab_size": vocab_size,
        "char_to_idx": char_to_idx,
    }


# ── Princeton improved preprocessing (char + token + tabular) ─────────────────

def _build_token_vocab(urls_clean, min_freq):
    """Tokenize URLs by delimiters, build vocab with min frequency."""
    import re
    all_tokens_list = [re.split(r'[/:?=&@.\-_~#%]+', u) for u in urls_clean]

    token_freq = {}
    for tokens in all_tokens_list:
        for t in tokens:
            if t:
                token_freq[t] = token_freq.get(t, 0) + 1

    token_vocab = {t: i + 1 for i, (t, freq) in enumerate(
        sorted(token_freq.items(), key=lambda x: -x[1])
    ) if freq >= min_freq}
    token_vocab_size = len(token_vocab) + 1
    return all_tokens_list, token_vocab, token_vocab_size


def _encode_url_tokens(tokens, mapping, maxlen):
    """Encode token list to fixed-length integer sequence."""
    encoded = [mapping.get(t, 0) for t in tokens[:maxlen] if t]
    if len(encoded) < maxlen:
        encoded += [0] * (maxlen - len(encoded))
    return encoded[:maxlen]


def princeton_improved_preprocess(cfg):
    """Full Princeton improved preprocessing. Returns dict with 3 input branches."""
    pcfg = cfg["princeton_improved"]
    seed = cfg["random_seed"]
    dataset_id = pcfg["dataset_id"]

    df = load_any_dataset(dataset_id, cfg)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    labels = le.fit_transform(df["status"])
    print(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    urls = df["url"].astype(str).values
    urls_clean = [u.lower().strip() for u in urls]

    # ── Char-level ──
    char_max_len = pcfg["char_max_len"]
    print("\n── Preparing char-level URL input ──")
    char_to_idx, char_vocab_size = _build_char_vocab(urls_clean)
    print(f"  Vocab size: {char_vocab_size}, Max seq len: {char_max_len}")
    X_char = np.array([_encode_url_chars(u, char_to_idx, char_max_len)
                        for u in urls_clean])
    print(f"  Char-encoded shape: {X_char.shape}")

    # ── Token-level ──
    token_max_len = pcfg["token_max_len"]
    token_min_freq = pcfg["token_min_freq"]
    print("\n── Preparing token-level URL input ──")
    all_tokens_list, token_vocab, token_vocab_size = _build_token_vocab(
        urls_clean, token_min_freq
    )
    print(f"  Token vocab size: {token_vocab_size}, Max tokens: {token_max_len}")
    X_token = np.array([_encode_url_tokens(toks, token_vocab, token_max_len)
                         for toks in all_tokens_list])
    print(f"  Token-encoded shape: {X_token.shape}")

    # ── Tabular numeric features ──
    print("\n── Preparing tabular numeric features ──")
    drop_cols = ["url", "status"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X_tab = df[feature_cols].values.astype(np.float32)
    X_tab = np.nan_to_num(X_tab, nan=0.0)
    from sklearn.preprocessing import MinMaxScaler
    tab_scaler = MinMaxScaler()
    X_tab = tab_scaler.fit_transform(X_tab)
    num_features = X_tab.shape[1]
    print(f"  Tabular features: {num_features}")

    y = np.array(labels, dtype=np.float32)

    # ── Split using indices (keeps all arrays aligned) ──
    from sklearn.model_selection import train_test_split
    test_size = pcfg["test_size"]
    val_size = pcfg["val_size"]

    idx = np.arange(len(y))
    idx_train, idx_test, y_train, y_test = train_test_split(
        idx, y, test_size=test_size, random_state=seed, stratify=y
    )
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_train, y_train, test_size=val_size, random_state=seed, stratify=y_train
    )

    print(f"\n  Train: {len(idx_train)}  |  Val: {len(idx_val)}  |  Test: {len(idx_test)}")

    return {
        "X_char_train": X_char[idx_train], "X_char_val": X_char[idx_val],
        "X_char_test": X_char[idx_test],
        "X_token_train": X_token[idx_train], "X_token_val": X_token[idx_val],
        "X_token_test": X_token[idx_test],
        "X_tab_train": X_tab[idx_train], "X_tab_val": X_tab[idx_val],
        "X_tab_test": X_tab[idx_test],
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "le": le,
        "char_vocab_size": char_vocab_size,
        "token_vocab_size": token_vocab_size,
        "num_tab_features": num_features,
        "char_to_idx": char_to_idx,
        "token_vocab": token_vocab,
        "tab_scaler": tab_scaler,
    }


if __name__ == '__main__':
    main()
