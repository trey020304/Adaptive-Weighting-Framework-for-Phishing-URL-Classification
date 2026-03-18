import pandas as pd
import numpy as np
import tldextract
import re
import os
import sys
from urllib.parse import urlparse
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

# The 14 standardized feature names
STANDARD_FEATURES = [
    'num_dots', 'url_length', 'num_domain_parts', 'num_hyphens',
    'num_underscores', 'num_at', 'num_ampersand', 'num_tilde',
    'num_percent', 'hostname_length', 'is_ip', 'num_double_slash',
    'has_https', 'has_suspicious'
]

# Continuous features where IQR outlier removal applies
# Excludes binary features (is_ip, has_https, has_suspicious) and
# sparse/zero-inflated counts (num_underscores, num_at, num_ampersand,
# num_tilde, num_percent, num_double_slash) — IQR collapses to [0,0]
# for those, wiping all non-zero values.
CONTINUOUS_FEATURES = [
    'num_dots', 'url_length', 'num_domain_parts', 'num_hyphens',
    'hostname_length'
]

# Suspicious keywords used for raw URL feature extraction
SUSPICIOUS_KEYWORDS = [
    'login', 'password', 'bank', 'secure', 'account', 'update', 'verify', 'signin',
    'paypal', 'ebay', 'amazon', 'facebook', 'google', 'microsoft', 'apple',
    'support', 'help', 'contact', 'free', 'win', 'prize', 'alert', 'warning',
    'confirm', 'validate', 'logon', 'auth', 'credential', 'payment'
]

def extract_features_from_url(url):
    """
    Extract the 14 lexical features from a URL.
    """
    url = str(url).lower()
    
    # 1. Number of "." in URL
    num_dots = url.count('.')
    
    # 2. Number of characters in URL
    url_length = len(url)
    
    # 3. Number of parts in the domain
    try:
        ext = tldextract.extract(url)
        domain_parts = ext.subdomain.split('.') + [ext.domain, ext.suffix]
        domain_parts = [p for p in domain_parts if p]
        num_domain_parts = len(domain_parts)
    except:
        num_domain_parts = 0
    
    # 4. Number of "-" in URL
    num_hyphens = url.count('-')
    
    # 5. Number of "_" in URL
    num_underscores = url.count('_')
    
    # 6. Number of "@" in URL
    num_at = url.count('@')
    
    # 7. Number of "&" in URL
    num_ampersand = url.count('&')
    
    # 8. Number of "~" in URL
    num_tilde = url.count('~')
    
    # 9. Number of "%" in URL
    num_percent = url.count('%')
    
    # 10. Number of characters in hostname
    parsed = urlparse(url)
    hostname = parsed.hostname or ''
    hostname_length = len(hostname)
    
    # 11. Is IP address present in URL
    is_ip = 1 if re.match(r'^\d+\.\d+\.\d+\.\d+$', hostname) else 0
    
    # 12. Number of "//" in URL
    num_double_slash = url.count('//')
    
    # 13. Is "https" present in URL
    has_https = 1 if 'https' in url else 0
    
    # 14. Is suspicious keyword present in URL (binary: 0 or 1)
    has_suspicious = 1 if any(kw in url for kw in SUSPICIOUS_KEYWORDS) else 0
    
    return {
        'num_dots': num_dots,
        'url_length': url_length,
        'num_domain_parts': num_domain_parts,
        'num_hyphens': num_hyphens,
        'num_underscores': num_underscores,
        'num_at': num_at,
        'num_ampersand': num_ampersand,
        'num_tilde': num_tilde,
        'num_percent': num_percent,
        'hostname_length': hostname_length,
        'is_ip': is_ip,
        'num_double_slash': num_double_slash,
        'has_https': has_https,
        'has_suspicious': has_suspicious
    }

def process_raw_csv(file_path, url_col='URL', label_col='Label'):
    """
    Process raw CSV with URLs and labels.
    """
    df = pd.read_csv(file_path)
    features_list = []
    for idx, row in df.iterrows():
        url = row[url_col]
        features = extract_features_from_url(url)
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
    """
    Process index.csv with rec_id, url, result.
    """
    df = pd.read_csv(file_path)
    features_list = []
    for idx, row in df.iterrows():
        url = row['url']
        features = extract_features_from_url(url)
        features['id'] = row['rec_id']
        result = row['result']
        features['label'] = 'Phishing' if result == 1 else 'Not Phishing'
        features_list.append(features)
    return pd.DataFrame(features_list)

def process_defined_csv(file_path):
    """
    Process defined CSV with features.
    Handles two naming conventions:
      - Datasets [3],[6]: nb_dots, length_url, nb_hyphens, etc.
      - Dataset  [4]:     NumDots, UrlLength, NumDash, etc.
    """
    df = pd.read_csv(file_path)

    # Standardized feature names
    STANDARD_FEATURES = [
        'num_dots', 'url_length', 'num_domain_parts', 'num_hyphens',
        'num_underscores', 'num_at', 'num_ampersand', 'num_tilde',
        'num_percent', 'hostname_length', 'is_ip', 'num_double_slash',
        'has_https', 'has_suspicious'
    ]

    # Map every known source column name → standard name
    # Multiple source names can map to the same standard name.
    column_mapping = {
        # --- datasets [3],[6] style ---
        'nb_dots':            'num_dots',
        'length_url':         'url_length',
        'nb_subdomains':      'num_domain_parts',
        'nb_hyphens':         'num_hyphens',
        'nb_underscore':      'num_underscores',
        'nb_at':              'num_at',
        'nb_and':             'num_ampersand',
        'nb_tilde':           'num_tilde',
        'nb_percent':         'num_percent',
        'length_hostname':    'hostname_length',
        'ip':                 'is_ip',
        'nb_dslash':          'num_double_slash',
        'https_token':        'has_https',
        'phish_hints':        'has_suspicious',
        'status':             'label',
        # --- dataset [4] style ---
        'NumDots':            'num_dots',
        'UrlLength':          'url_length',
        'SubdomainLevel':     'num_domain_parts',
        'NumDash':            'num_hyphens',
        'NumUnderscore':      'num_underscores',
        'AtSymbol':           'num_at',
        'NumAmpersand':       'num_ampersand',
        'TildeSymbol':        'num_tilde',
        'NumPercent':         'num_percent',
        'HostnameLength':     'hostname_length',
        'IpAddress':          'is_ip',
        'DoubleSlashInPath':  'num_double_slash',
        'NoHttps':            'no_https',
        'NumSensitiveWords':  'has_suspicious',
        'CLASS_LABEL':        'label',
    }

    # Pick only columns we know how to map
    available_cols = [col for col in df.columns if col in column_mapping]
    df_selected = df[available_cols].copy()
    df_selected.rename(columns=column_mapping, inplace=True)

    # Add id
    if 'id' in df.columns:
        df_selected['id'] = df['id'].astype(int)
    else:
        df_selected['id'] = df_selected.index + 1

    # Handle https: two conventions
    if 'no_https' in df_selected.columns:
        # NoHttps: 1 means no https → invert
        df_selected['has_https'] = (df_selected['no_https'] == 0).astype(int)
        df_selected.drop(columns=['no_https'], inplace=True)
    elif 'has_https' in df_selected.columns:
        # https_token: 1 means http (no https) → invert
        df_selected['has_https'] = 1 - df_selected['has_https']

    # Binarize has_suspicious: any value > 0 becomes 1
    if 'has_suspicious' in df_selected.columns:
        df_selected['has_suspicious'] = (df_selected['has_suspicious'] > 0).astype(int)
    else:
        df_selected['has_suspicious'] = 0

    # Standardize label
    if df_selected['label'].dtype == 'object':
        df_selected['label'] = df_selected['label'].map({'legitimate': 'Not Phishing', 'phishing': 'Phishing'})
    else:
        df_selected['label'] = df_selected['label'].map({0: 'Not Phishing', 1: 'Phishing'})

    # Keep only id + 14 standard features + label
    df_selected = df_selected[['id'] + STANDARD_FEATURES + ['label']]

    return df_selected

def process_arff(file_path):
    """
    Process .arff file.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find @data
    data_start = None
    for i, line in enumerate(lines):
        if line.strip().lower() == '@data':
            data_start = i + 1
            break
    
    if data_start is None:
        raise ValueError("No @data section found")
    
    # Attributes
    attributes = []
    for line in lines:
        if line.startswith('@attribute'):
            attr = line.split()[1]
            attributes.append(attr)
    
    # Data
    data_lines = lines[data_start:]
    data = []
    for line in data_lines:
        line = line.strip()
        if line:
            values = line.split(',')
            data.append(values)
    
    df = pd.DataFrame(data, columns=attributes)
    
    # Convert to numeric
    for col in df.columns:
        if col != 'CLASS_LABEL':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Map to our features
    feature_mapping = {
        'NumDots': 'num_dots',
        'UrlLength': 'url_length',
        'SubdomainLevel': 'num_domain_parts',
        'NumDash': 'num_hyphens',
        'NumUnderscore': 'num_underscores',
        'AtSymbol': 'num_at',
        'NumAmpersand': 'num_ampersand',
        'TildeSymbol': 'num_tilde',
        'NumPercent': 'num_percent',
        'HostnameLength': 'hostname_length',
        'IpAddress': 'is_ip',
        'DoubleSlashInPath': 'num_double_slash',
        'NoHttps': 'no_https',
        'NumSensitiveWords': 'num_sensitive',
        'CLASS_LABEL': 'label'
    }
    
    df.rename(columns=feature_mapping, inplace=True)
    
    # has_https = 1 if no_https == 0
    df['has_https'] = (df['no_https'] == 0).astype(int)
    
    # has_suspicious = 1 if num_sensitive > 0 (binarize)
    df['has_suspicious'] = (df['num_sensitive'] > 0).astype(int)
    
    # Label — ARFF class values are strings ('0'/'1')
    df['label'] = df['label'].map({'0': 'Not Phishing', '1': 'Phishing',
                                    0: 'Not Phishing',  1: 'Phishing'})
    
    # Add id
    df['id'] = df.index + 1
    
    # Keep only id + 14 standard features + label
    STANDARD_FEATURES = [
        'num_dots', 'url_length', 'num_domain_parts', 'num_hyphens',
        'num_underscores', 'num_at', 'num_ampersand', 'num_tilde',
        'num_percent', 'hostname_length', 'is_ip', 'num_double_slash',
        'has_https', 'has_suspicious'
    ]
    df = df[['id'] + STANDARD_FEATURES + ['label']]
    
    return df

DATASETS = [
    {'id': 1, 'name': '[3] Shashwat Tiwari (Kaggle, Defined)',       'path': 'defined/[3] dataset_phishing_kaggle_Shashwat Tiwari.csv',        'processor': 'defined_csv'},
    {'id': 2, 'name': '[4] Mohamad Fadil (Kaggle, Defined)',          'path': 'defined/[4] Phishing_Legitimate_full_Kaggle_Mohamad Fadil.csv',  'processor': 'defined_csv'},
    {'id': 3, 'name': '[6] Hannousse (Mendeley, Defined)',            'path': 'defined/[6] dataset_B_05_2020_Mendeley_Hannousse.csv',           'processor': 'defined_csv'},
    {'id': 4, 'name': '[b3] Tan (Mendeley, Defined, ARFF)',           'path': 'defined/[b3] Phishing_Legitimate_full_Mendeley_Tan.arff',        'processor': 'arff'},
    {'id': 5, 'name': '[7] Phishing Site URLs (Raw)',                 'path': 'raw/[7] phishing_site_urls.csv',                                 'processor': 'raw_csv'},
    {'id': 6, 'name': '[c1] Index (Raw)',                             'path': 'raw/[c1] index.csv',                                             'processor': 'index_csv'},
]

PROCESSOR_MAP = {
    'defined_csv': process_defined_csv,
    'arff':        process_arff,
    'raw_csv':     lambda f: process_raw_csv(f, 'URL', 'Label'),
    'index_csv':   process_index_csv,
}


def load_dataset(entry, datasets_dir='datasets'):
    """Load a single dataset, extract features, drop missing values."""
    file_path = os.path.join(datasets_dir, entry['path'])
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    processor = PROCESSOR_MAP[entry['processor']]
    df = processor(file_path)
    df = df[STANDARD_FEATURES + ['label']]

    before = len(df)
    df = df.dropna(subset=STANDARD_FEATURES + ['label']).reset_index(drop=True)
    dropped = before - len(df)

    print(f"  [{entry['id']}] {entry['name']}: {len(df)} rows", end='')
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
    Cross-dataset preprocessing pipeline.
    - Training datasets   → IQR outlier removal + SMOTE (both paths), MinMaxScaler (Path B)
    - Validation datasets → untouched (Path A), scaled with train scaler (Path B)
    - Testing datasets    → untouched (Path A), scaled with train scaler (Path B)
    """

    # ── Step 1: Load and merge each role ──
    train_df = load_and_merge(train_entries, 'TRAINING', datasets_dir)
    val_df   = load_and_merge(val_entries,   'VALIDATION', datasets_dir)
    test_df  = load_and_merge(test_entries,  'TESTING', datasets_dir)

    # ── Step 2: IQR outlier removal on training set only ──
    before = len(train_df)
    for col in CONTINUOUS_FEATURES:
        Q1 = train_df[col].quantile(0.25)
        Q3 = train_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        train_df = train_df[(train_df[col] >= lower) & (train_df[col] <= upper)]
    train_df = train_df.reset_index(drop=True)
    removed = before - len(train_df)
    if removed:
        print(f"\nIQR outlier removal: removed {removed} rows from training set ({len(train_df)} remaining)")

    # ── Step 3: Build output folder name from dataset IDs ──
    train_ids = ','.join(str(e['id']) for e in train_entries)
    val_ids   = ','.join(str(e['id']) for e in val_entries)
    test_ids  = ','.join(str(e['id']) for e in test_entries)
    run_folder = f'train[{train_ids}]_val[{val_ids}]_test[{test_ids}]'

    # ==================== PATH A: Chi-square + Random Forest ====================
    print(f"\n{'='*60}")
    print(f"PATH A — Chi-square + Random Forest (raw features)")
    print(f"{'='*60}")
    path_a_dir = os.path.join(output_dir, 'path_a', run_folder)
    os.makedirs(path_a_dir, exist_ok=True)

    # SMOTE on training set only
    X_train = train_df[STANDARD_FEATURES]
    y_train = train_df['label']
    class_counts = y_train.value_counts()
    print(f"  Train class distribution before SMOTE: {dict(class_counts)}")

    if len(class_counts) == 2 and class_counts.min() < class_counts.max():
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        train_a = pd.DataFrame(X_res, columns=STANDARD_FEATURES)
        train_a['label'] = y_res
        print(f"  Applied SMOTE: {len(train_a)} training rows (balanced)")
    else:
        train_a = train_df[STANDARD_FEATURES + ['label']].copy()
        print(f"  SMOTE skipped (classes already balanced)")

    n_train = save_split(train_a, 'train', path_a_dir)
    n_val   = save_split(val_df,  'val',   path_a_dir)
    n_test  = save_split(test_df, 'test',  path_a_dir)
    print(f"  Saved to {path_a_dir}")
    print(f"    train={n_train}, val={n_val}, test={n_test}")

    # ==================== PATH B: CNN-BiLSTM ====================
    print(f"\n{'='*60}")
    print(f"PATH B — CNN-BiLSTM (MinMaxScaler)")
    print(f"{'='*60}")
    path_b_dir = os.path.join(output_dir, 'path_b', run_folder)
    os.makedirs(path_b_dir, exist_ok=True)

    # Fit scaler on training set, transform all splits
    scaler = MinMaxScaler()
    train_b = train_df[STANDARD_FEATURES + ['label']].copy()
    val_b   = val_df[STANDARD_FEATURES + ['label']].copy()
    test_b  = test_df[STANDARD_FEATURES + ['label']].copy()

    train_b[STANDARD_FEATURES] = scaler.fit_transform(train_b[STANDARD_FEATURES])
    val_b[STANDARD_FEATURES]   = scaler.transform(val_b[STANDARD_FEATURES])
    test_b[STANDARD_FEATURES]  = scaler.transform(test_b[STANDARD_FEATURES])
    print(f"  MinMaxScaler fit on training set, applied to all splits")

    # SMOTE on scaled training set only
    X_train_b = train_b[STANDARD_FEATURES]
    y_train_b = train_b['label']
    class_counts_b = y_train_b.value_counts()
    print(f"  Train class distribution before SMOTE: {dict(class_counts_b)}")

    if len(class_counts_b) == 2 and class_counts_b.min() < class_counts_b.max():
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train_b, y_train_b)
        train_b = pd.DataFrame(X_res, columns=STANDARD_FEATURES)
        train_b['label'] = y_res
        print(f"  Applied SMOTE: {len(train_b)} training rows (balanced)")
    else:
        train_b = train_b[STANDARD_FEATURES + ['label']].copy()
        print(f"  SMOTE skipped (classes already balanced)")

    n_train = save_split(train_b, 'train', path_b_dir)
    n_val   = save_split(val_b,   'val',   path_b_dir)
    n_test  = save_split(test_b,  'test',  path_b_dir)
    print(f"  Saved to {path_b_dir}")
    print(f"    train={n_train}, val={n_val}, test={n_test}")


def parse_ids(text):
    """Parse comma-separated dataset IDs into a list of dataset entries."""
    ids = [int(x.strip()) for x in text.split(',')]
    entries = [ds for ds in DATASETS if ds['id'] in ids]
    return entries


def get_input(prompt, arg_value=None):
    """Get input from CLI arg or interactive prompt."""
    if arg_value is not None:
        print(f"{prompt}{arg_value}")
        return arg_value
    return input(prompt).strip()


def main():
    print("\n===== Cross-Dataset Preprocessing Pipeline =====\n")
    print("Available datasets:")
    for ds in DATASETS:
        print(f"  [{ds['id']}] {ds['name']}")
    print()

    # CLI usage: python preprocess.py <train_ids> <val_ids> <test_ids>
    # Example:   python preprocess.py 1,2,5 3 4,6
    # Interactive: prompts for each role
    cli_train = sys.argv[1].strip() if len(sys.argv) > 1 else None
    cli_val   = sys.argv[2].strip() if len(sys.argv) > 2 else None
    cli_test  = sys.argv[3].strip() if len(sys.argv) > 3 else None

    try:
        train_text = get_input("Enter TRAINING dataset IDs (comma-separated): ", cli_train)
        val_text   = get_input("Enter VALIDATION dataset IDs (comma-separated): ", cli_val)
        test_text  = get_input("Enter TESTING dataset IDs (comma-separated): ", cli_test)

        train_entries = parse_ids(train_text)
        val_entries   = parse_ids(val_text)
        test_entries  = parse_ids(test_text)
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
    train_ids = {e['id'] for e in train_entries}
    val_ids   = {e['id'] for e in val_entries}
    test_ids  = {e['id'] for e in test_entries}
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