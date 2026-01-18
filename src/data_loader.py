import pandas as pd
import numpy as np
import re

import os
import glob
import json
from pathlib import Path
from datetime import datetime

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

_DATA_ROOT = Path(os.environ.get('MCSA_DATA_DIR', str(_PROJECT_ROOT.joinpath('data')))).expanduser().resolve()


def get_data_path(*parts):
    return str(_DATA_ROOT.joinpath(*parts))


def audit_mcsa_dataframe(df, sample_n=25):
    if df is None:
        return {
            'rows': 0,
            'missing_columns': [],
            'invalid_date_count': 0,
            'duplicate_key_count': 0,
            'missing_equipment_count': 0,
            'missing_parameter_count': 0,
            'unknown_unit_count': 0,
            'unknown_voltage_count': 0,
            'invalid_date_sample': pd.DataFrame(),
            'duplicate_key_sample': pd.DataFrame(),
        }

    required = ['Equipment', 'Parameter', 'Date', 'Raw_Value']
    missing_columns = [c for c in required if c not in df.columns]

    rows = int(len(df))
    invalid_date_sample = pd.DataFrame()
    invalid_date_count = 0
    if 'Date' in df.columns:
        dt = pd.to_datetime(df['Date'], errors='coerce')
        invalid_mask = dt.isna()
        invalid_date_count = int(invalid_mask.sum())
        if invalid_date_count:
            cols = [c for c in ['Equipment', 'Parameter', 'Date', 'Raw_Value'] if c in df.columns]
            invalid_date_sample = df.loc[invalid_mask, cols].head(sample_n)

    duplicate_key_sample = pd.DataFrame()
    duplicate_key_count = 0
    if all(c in df.columns for c in ['Equipment', 'Parameter', 'Date']):
        dt = pd.to_datetime(df['Date'], errors='coerce')
        key_df = pd.DataFrame({
            'Equipment': df['Equipment'].astype(str),
            'Parameter': df['Parameter'].astype(str),
            'Date': dt
        })
        key_df = key_df.dropna(subset=['Date'])
        if not key_df.empty:
            grp = key_df.groupby(['Equipment', 'Parameter', 'Date']).size().reset_index(name='Count')
            dup = grp[grp['Count'] > 1].sort_values('Count', ascending=False)
            duplicate_key_count = int(len(dup))
            if duplicate_key_count:
                duplicate_key_sample = dup.head(sample_n)

    missing_equipment_count = int((df.get('Equipment', pd.Series([], dtype=object)).astype(str).str.strip() == '').sum()) if 'Equipment' in df.columns else 0
    missing_parameter_count = int((df.get('Parameter', pd.Series([], dtype=object)).astype(str).str.strip() == '').sum()) if 'Parameter' in df.columns else 0

    def _unknown_mask(series):
        s = series.astype(str).str.strip().str.lower()
        return s.isin(['', 'unknown', 'nan', 'none'])

    unknown_unit_count = int(_unknown_mask(df['Unit_Name']).sum()) if 'Unit_Name' in df.columns else 0
    unknown_voltage_count = int(_unknown_mask(df['Voltage_Level']).sum()) if 'Voltage_Level' in df.columns else 0

    return {
        'rows': rows,
        'missing_columns': missing_columns,
        'invalid_date_count': invalid_date_count,
        'duplicate_key_count': duplicate_key_count,
        'missing_equipment_count': missing_equipment_count,
        'missing_parameter_count': missing_parameter_count,
        'unknown_unit_count': unknown_unit_count,
        'unknown_voltage_count': unknown_voltage_count,
        'invalid_date_sample': invalid_date_sample,
        'duplicate_key_sample': duplicate_key_sample,
    }


def fix_mcsa_dataframe(df):
    if df is None:
        return pd.DataFrame(), {
            'rows_before': 0,
            'rows_after': 0,
            'dropped_empty_keys': 0,
            'repaired_dates': 0,
            'invalid_dates_remaining': 0,
            'duplicates_removed': 0,
            'filled_unit_name': 0,
            'filled_voltage_level': 0,
            'filled_full_name': 0,
        }

    work = df.copy()
    rows_before = int(len(work))

    for col, default in [('Unit_Name', 'Unknown'), ('Voltage_Level', 'Unknown')]:
        if col not in work.columns:
            work[col] = default
        s = work[col].astype(str).str.strip()
        mask = s.eq('') | s.str.lower().isin(['unknown', 'nan', 'none'])
        if col == 'Unit_Name':
            filled_unit_name = int(mask.sum())
        else:
            filled_voltage_level = int(mask.sum())
        work.loc[mask, col] = default

    if 'Full_Name' not in work.columns:
        work['Full_Name'] = work.get('Equipment', '').astype(str)
        filled_full_name = int(len(work))
    else:
        s = work['Full_Name'].astype(str).str.strip()
        mask = s.eq('') | s.str.lower().isin(['unknown', 'nan', 'none'])
        filled_full_name = int(mask.sum())
        work.loc[mask, 'Full_Name'] = work.get('Equipment', '').astype(str)

    dropped_empty_keys = 0
    if 'Equipment' in work.columns and 'Parameter' in work.columns:
        eq_ok = work['Equipment'].astype(str).str.strip().ne('')
        p_ok = work['Parameter'].astype(str).str.strip().ne('')
        dropped_empty_keys = int((~(eq_ok & p_ok)).sum())
        work = work[eq_ok & p_ok].copy()

    repaired_dates = 0
    if 'Date' in work.columns:
        dt = pd.to_datetime(work['Date'], errors='coerce')
        needs_repair = dt.isna()
        if needs_repair.any() and 'Year' in work.columns and ('Month_Name' in work.columns or 'Month' in work.columns):
            month_src = work['Month_Name'] if 'Month_Name' in work.columns else work['Month']
            month_s = month_src.astype(str).str.strip().str.upper()
            month_order = {
                'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
            }
            month_num = month_s.map(month_order)
            if month_num.isna().any():
                try_num = pd.to_numeric(month_s, errors='coerce')
                month_num = month_num.fillna(try_num)
            y = pd.to_numeric(work['Year'], errors='coerce')
            can = needs_repair & y.notna() & month_num.notna()
            if can.any():
                repaired_dates = int(can.sum())
                work.loc[can, 'Date'] = pd.to_datetime(
                    y.loc[can].astype(int).astype(str)
                    + '-' + month_num.loc[can].astype(int).astype(str).str.zfill(2)
                    + '-01',
                    errors='coerce'
                )

        work['Date'] = pd.to_datetime(work['Date'], errors='coerce')
        invalid_dates_remaining = int(work['Date'].isna().sum())
    else:
        invalid_dates_remaining = 0

    duplicates_removed = 0
    if all(c in work.columns for c in ['Equipment', 'Parameter', 'Date']):
        before = int(len(work))
        work = work.sort_values('Date').drop_duplicates(subset=['Equipment', 'Parameter', 'Date'], keep='last')
        duplicates_removed = int(before - len(work))

    rows_after = int(len(work))

    return work, {
        'rows_before': rows_before,
        'rows_after': rows_after,
        'dropped_empty_keys': dropped_empty_keys,
        'repaired_dates': repaired_dates,
        'invalid_dates_remaining': invalid_dates_remaining,
        'duplicates_removed': duplicates_removed,
        'filled_unit_name': filled_unit_name,
        'filled_voltage_level': filled_voltage_level,
        'filled_full_name': filled_full_name,
    }


def _normalize_date_columns(df):
    if 'Year' not in df.columns:
        df['Year'] = datetime.now().year
    if 'Date' not in df.columns:
        month_order = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
            'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }
        month_num = df['Month'].map(month_order).fillna(1)
        df['Date'] = pd.to_datetime(
            df['Year'].astype(int).astype(str)
            + '-' + month_num.astype(int).astype(str).str.zfill(2)
            + '-01',
            errors='coerce'
        )
    else:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    if 'Month_Name' not in df.columns:
        df['Month_Name'] = df['Date'].dt.strftime('%b').str.upper()
        df.loc[df['Month_Name'].isna(), 'Month_Name'] = df['Month'].astype(str)

    return df


def _normalize_status_columns(df):
    if df is None or df.empty:
        return df
    if 'Parameter' not in df.columns or 'Raw_Value' not in df.columns:
        return df
    if 'Status' not in df.columns:
        df['Status'] = 'Normal'

    p = df['Parameter'].astype(str).str.strip().str.lower()
    mask = p.isin(['kondisi', 'bearing', 'rotorbar'])
    rv = df['Raw_Value']
    rv_ok = rv.notna() & rv.astype(str).str.strip().ne('')
    df.loc[mask & rv_ok, 'Status'] = rv.loc[mask & rv_ok]

    ps_mask = df['Parameter'].astype(str).str.startswith('Ringkasan Kinerja')
    df.loc[ps_mask, 'Status'] = 'Info'
    return df


def _normalize_unit_voltage_columns(df):
    if df is None or df.empty:
        return df

    def _canon_unit(v):
        s = str(v).strip().upper()
        s = re.sub(r'[_\-]+', ' ', s)
        s = re.sub(r'\s+', ' ', s)
        if s in {'NAN', 'NONE', ''}:
            return 'Unknown'
        if re.search(r'\b(COMMON|COMM|COM|CMN)\b', s):
            return 'UNIT COMMON'
        m = re.search(r'\bUNIT\b\s*([1-3]|I|II|III)\b', s)
        if not m:
            m = re.search(r'\bU\s*([1-3])\b', s)
        if m:
            roman = {'I': '1', 'II': '2', 'III': '3'}
            u = roman.get(m.group(1), m.group(1))
            return f'UNIT {u}'
        return 'Unknown'

    def _canon_volt(v):
        s = str(v).strip().upper().replace(' ', '')
        if s in {'NAN', 'NONE', ''}:
            return 'Unknown'
        if '6.3' in s or '63KV' in s or '6KV' in s:
            return '6.3 KV'
        if '400' in s or '380' in s:
            return '380/400 V'
        return 'Unknown'

    if 'Unit_Name' in df.columns:
        df['Unit_Name'] = df['Unit_Name'].apply(_canon_unit)
    if 'Voltage_Level' in df.columns:
        df['Voltage_Level'] = df['Voltage_Level'].apply(_canon_volt)
    return df


def _add_status_category(df):
    if df is None or df.empty:
        return df
    if 'Parameter' not in df.columns:
        return df

    if 'Status' not in df.columns and 'Raw_Value' in df.columns:
        df['Status'] = df['Raw_Value']
    if 'Status' not in df.columns:
        df['Status'] = 'Normal'

    p = df['Parameter'].astype(str).str.strip().str.lower()
    src = df['Status']
    if 'Raw_Value' in df.columns:
        src = src.fillna(df['Raw_Value'])
    src = src.astype(str)
    s = src.str.strip().str.lower()

    cat = pd.Series('Unknown', index=df.index)
    cat[s.str.contains('standby', na=False)] = 'Standby'
    cat[s.str.contains('normal|\bok\b|good', na=False)] = 'Normal'
    cat[s.str.contains('alarm|warning', na=False)] = 'Alarm'
    cat[s.str.contains('high|bad|critical|rusak|damage', na=False)] = 'High'

    mask = p.isin(['kondisi', 'bearing'])
    df['Status_Category'] = df.get('Status_Category', 'Unknown')
    df.loc[mask, 'Status_Category'] = cat.loc[mask]

    level_map = {'Standby': 0, 'Normal': 1, 'Alarm': 2, 'High': 3, 'Unknown': -1}
    df['Status_Level'] = df.get('Status_Level', -1)
    df.loc[mask, 'Status_Level'] = df.loc[mask, 'Status_Category'].map(level_map).fillna(-1).astype(int)
    return df

def load_nameplate_csv():
    path = Path(get_data_path('master', 'motor_nameplate.csv'))
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_csv(path, encoding='utf-8')
        except Exception:
            df = pd.read_csv(path, encoding='latin1')
    cols = {
        'Equipment': 'Equipment',
        'Voltage_Nominal': 'Voltage_Nominal',
        'FLA': 'FLA',
        'PF_Nominal': 'PF_Nominal',
        'RPM': 'RPM',
        'VFD_Flag': 'VFD_Flag'
    }
    for k, v in cols.items():
        if k not in df.columns and v in df.columns:
            df[k] = df[v]
    for c in ['Equipment']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df
def load_equipment_mapping():
    """Loads equipment acronym mapping."""
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        map_path = os.path.join(base_dir, 'src', 'equipment_mapping.json')
        with open(map_path, 'r') as f:
            return json.load(f)
    except:
        return {}

def get_full_name(code, mapping):
    """
    Converts code (e.g. C3WP1A) to full name (e.g. Closed Cooling Water Pump 1A).
    """
    code_upper = code.upper()
    
    # Sort keys by length desc to match longest prefix first (e.g. IDFF vs IDF)
    sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
    
    for prefix in sorted_keys:
        if code_upper.startswith(prefix):
            # Extract suffix (e.g. 1A)
            suffix = code_upper[len(prefix):]
            # Handle special cases like BC102 -> Belt Conveyor 102
            return f"{mapping[prefix]} {suffix}"
            
    return code # Return original if no match

def get_folder_metadata(base_path):
    """
    Walks through data/Laporan to build a mapping of Equipment Code -> (Unit, Voltage, FullName).
    Returns a dict: {'C3WP1A': {'Unit': 'UNIT 1', 'Voltage': '400 V', 'Full_Name': '...'}, ...}
    """
    mapping = {}
    eq_name_map = load_equipment_mapping()
    
    # Structure: data/Laporan/{Unit}/{Voltage}/{Filename}
    # We assume base_path is d:/Project/MCSA/data/Laporan
    
    if not os.path.exists(base_path):
        return mapping

    def _normalize_unit_name(name):
        s = str(name).strip().upper()
        s = re.sub(r'[_\-]+', ' ', s)
        s = re.sub(r'\s+', ' ', s)
        if re.search(r'\b(COMMON|COMM|COM|CMN)\b', s):
            return 'UNIT COMMON'
        m = re.search(r'\bUNIT\b\s*([1-3]|I|II|III)\b', s)
        if not m:
            m = re.search(r'\bU\s*([1-3])\b', s)
        if m:
            v = m.group(1)
            roman = {'I': '1', 'II': '2', 'III': '3'}
            v = roman.get(v, v)
            return f'UNIT {v}'
        return name

    for unit_folder in os.listdir(base_path):
        unit_path = os.path.join(base_path, unit_folder)
        if os.path.isdir(unit_path):
            # Check if it is a Unit folder
            if "UNIT" in unit_folder.upper() or re.search(r'\bU\s*[1-3]\b', unit_folder.upper()) or re.search(r'\b(COMMON|COMM|COM|CMN)\b', unit_folder.upper()):
                unit_canon = _normalize_unit_name(unit_folder)
                # Look for Voltage folders
                for volt_folder in os.listdir(unit_path):
                    volt_path = os.path.join(unit_path, volt_folder)
                    if os.path.isdir(volt_path):
                        # Determine voltage string
                        volt_str = "Unknown"
                        if "400" in volt_folder or "380" in volt_folder:
                            volt_str = "380/400 V"
                        elif "6.3" in volt_folder:
                            volt_str = "6.3 KV"
                        
                        # Look for files
                        for fname in os.listdir(volt_path):
                            if fname.startswith('~$'):
                                continue
                            ext = os.path.splitext(fname)[1].lower()
                            if ext not in {'.docx', '.docm'}:
                                continue
                            # Extract code: C3WP1A_000.docx -> C3WP1A
                            # Split by _ or .
                            code = fname.split('_')[0].split('.')[0]
                            full_name = get_full_name(code, eq_name_map)
                            mapping[code.upper()] = {
                                'Unit': unit_canon, 
                                'Voltage': volt_str,
                                'Full_Name': full_name
                            }
                            
    return mapping

def load_mcsa_data(file_path, force_excel=False):
    """
    Loads MCSA data. Prioritizes 'mcsa_updated.csv' if it exists.
    Otherwise loads from the specified Excel file.
    If force_excel=True, ignores CSV and loads from Excel.
    """
    csv_path = os.path.join(os.path.dirname(file_path), 'mcsa_updated.csv')
    if os.path.exists(csv_path) and not force_excel:
        try:
            df = pd.read_csv(csv_path)
            # Ensure new columns exist if loading from old CSV
            if 'Unit_Name' not in df.columns:
                df['Unit_Name'] = 'Unknown'
            if 'Voltage_Level' not in df.columns:
                df['Voltage_Level'] = 'Unknown'
            if 'Full_Name' not in df.columns:
                df['Full_Name'] = df['Equipment']
            df = _normalize_date_columns(df)
            df = _normalize_status_columns(df)
            df = _normalize_unit_voltage_columns(df)
            df = _add_status_category(df)

            try:
                df = _recalculate_latest_per_equipment(df)
                df = _normalize_date_columns(df)
                df = _normalize_status_columns(df)
                df = _normalize_unit_voltage_columns(df)
                df = _add_status_category(df)
            except:
                pass

            laporan_path = os.path.join(os.path.dirname(file_path), 'Laporan')
            mapping = get_folder_metadata(laporan_path)
            try:
                from src.docx_parser import parse_all_reports
                df_word = parse_all_reports(laporan_path, mapping)
            except:
                df_word = pd.DataFrame()

            if not df_word.empty:
                for col in ['Unit_Name', 'Voltage_Level', 'Full_Name']:
                    if col not in df_word.columns:
                        df_word[col] = 'Unknown'
                for col in ['Unit_Name', 'Voltage_Level', 'Full_Name']:
                    if col not in df.columns:
                        df[col] = 'Unknown'

                merged = pd.concat([df_word, df], ignore_index=True)
                merged['Date'] = pd.to_datetime(merged.get('Date', pd.NaT), errors='coerce')
                merged = merged.drop_duplicates(subset=['Equipment', 'Parameter', 'Date'], keep='last')
                merged = _normalize_date_columns(merged)
                merged = _normalize_status_columns(merged)
                merged = _normalize_unit_voltage_columns(merged)
                merged = _add_status_category(merged)

                try:
                    merged = _recalculate_latest_per_equipment(merged)
                    merged = _normalize_date_columns(merged)
                    merged = _normalize_status_columns(merged)
                    merged = _normalize_unit_voltage_columns(merged)
                    merged = _add_status_category(merged)
                except:
                    pass
                return merged

            return df
        except:
            pass
    try:
        # Read the file, skipping initial rows to align with the header structure we found
        df = pd.read_excel(file_path, header=None, skiprows=11)
        
        # Define column names based on our inspection
        # [nan, 1, 'WATER JET PUMP A', 'WATER JET PUMP A', 'Month', nan, nan, 'JAN', ...]
        # We'll assign meaningful names
        col_names = [
            'raw_0', 'No', 'Eq_Temp', 'Equipment', 'Parameter', 'Unit', 'Limit',
            'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
            'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'
        ]
        
        # Assign columns. Handle cases where there might be extra columns
        if len(df.columns) > len(col_names):
            df = df.iloc[:, :len(col_names)]
        df.columns = col_names[:len(df.columns)]
        
        # Forward fill Equipment name
        # The 'Equipment' column (index 3) seems to contain the name.
        # Sometimes it might be in 'Eq_Temp' (index 2). Let's coalesce if needed.
        df['Equipment'] = df['Equipment'].fillna(method='ffill')
        
        # Drop rows where Parameter is NaN (these are likely spacer rows)
        df = df.dropna(subset=['Parameter'])
        
        # Filter out header rows that might have been caught (e.g. if "Month" appears in Parameter column)
        df = df[df['Parameter'] != 'Month']
        df = df[df['Parameter'] != 'Parameter Trending']
        
        # Melt the DataFrame to long format for easier analysis/plotting
        value_vars = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        
        # Ensure only existing columns are used (in case file doesn't have all months)
        existing_value_vars = [c for c in value_vars if c in df.columns]
        
        df_long = df.melt(
            id_vars=['Equipment', 'Parameter', 'Unit', 'Limit'],
            value_vars=existing_value_vars,
            var_name='Month',
            value_name='Raw_Value'
        )
        
        # Create a Numeric Value column
        df_long['Value'] = pd.to_numeric(df_long['Raw_Value'], errors='coerce')
        
        # Status Logic (Simple Rule-based)
        # We can try to parse the 'Limit' column, but it varies (<= 25%, -, etc.)
        # For now, we will add a placeholder 'Condition' based on existing data if possible,
        # or just leave it for the visualizer to handle.
        # Let's add a 'Status' column initialized to 'Normal'
        df_long['Status'] = 'Normal'
        
        # Example logic: If Value is NaN but Raw_Value is not NaN (e.g. 'SE'), mark as 'Info'
        df_long.loc[df_long['Value'].isna() & df_long['Raw_Value'].notna(), 'Status'] = 'Info'
        
        # Clean up
        df_long = df_long.sort_values(by=['Equipment', 'Month'])

        df_long['Year'] = datetime.now().year
        month_order = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
            'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }
        month_num = df_long['Month'].map(month_order)
        month_num = month_num.fillna(1)
        df_long['Date'] = pd.to_datetime(
            df_long['Year'].astype(int).astype(str)
            + '-' + month_num.astype(int).astype(str).str.zfill(2)
            + '-01',
            errors='coerce'
        )
        df_long['Month_Name'] = df_long['Month']
        
        # --- NEW: Add Unit and Voltage Columns ---
        df_long['Unit_Name'] = 'Unknown'
        df_long['Voltage_Level'] = 'Unknown'
        df_long['Full_Name'] = df_long['Equipment'] # Default to Code
        
        # Try to map from folders
        laporan_path = os.path.join(os.path.dirname(file_path), 'Laporan')
        mapping = get_folder_metadata(laporan_path)
        
        # Apply mapping
        # We need to match df['Equipment'] with mapping keys.
        # Since exact match is unlikely (e.g. "WATER JET PUMP A" vs "WJP3B"), 
        # we might just map what we can, and leave others for manual update.
        # However, for demo, let's try basic containment.
        
        for eq in df_long['Equipment'].unique():
            # 1. Try exact match (normalized)
            eq_norm = str(eq).strip().upper()
            found = False
            
            # Direct lookup
            if eq_norm in mapping:
                df_long.loc[df_long['Equipment'] == eq, 'Unit_Name'] = mapping[eq_norm]['Unit']
                df_long.loc[df_long['Equipment'] == eq, 'Voltage_Level'] = mapping[eq_norm]['Voltage']
                df_long.loc[df_long['Equipment'] == eq, 'Full_Name'] = mapping[eq_norm]['Full_Name']
                found = True
            
            # Fuzzy / Partial lookup
            if not found:
                for code, meta in mapping.items():
                    # Reverse logic: Does the code (C3WP1A) appear in Equipment Name (MOTOR C3WP1A)?
                    # OR does Equipment Name (WJP3B) appear in Code?
                    
                    # Logic A: Code in Equipment Name
                    # e.g. code="C3WP1A", eq="C3WP1A" -> Match
                    if code in eq_norm: 
                        df_long.loc[df_long['Equipment'] == eq, 'Unit_Name'] = meta['Unit']
                        df_long.loc[df_long['Equipment'] == eq, 'Voltage_Level'] = meta['Voltage']
                        df_long.loc[df_long['Equipment'] == eq, 'Full_Name'] = meta['Full_Name']
                        break

        df_long = _normalize_date_columns(df_long)
        df_long = _normalize_status_columns(df_long)
        df_long = _normalize_unit_voltage_columns(df_long)
        df_long = _add_status_category(df_long)

        df_word = pd.DataFrame()
        try:
            from src.docx_parser import parse_all_reports
            df_word = parse_all_reports(laporan_path, mapping)
        except:
            df_word = pd.DataFrame()

        if not df_word.empty:
            merged = pd.concat([df_word, df_long], ignore_index=True)
            merged['Date'] = pd.to_datetime(merged.get('Date', pd.NaT), errors='coerce')
            merged = merged.drop_duplicates(subset=['Equipment', 'Parameter', 'Date'], keep='last')
            merged = _normalize_date_columns(merged)
            merged = _normalize_status_columns(merged)
            merged = _normalize_unit_voltage_columns(merged)
            merged = _add_status_category(merged)
            return merged

        return df_long

    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()


def _recalculate_latest_per_equipment(df):
    if df is None or df.empty:
        return df
    if 'Equipment' not in df.columns or 'Parameter' not in df.columns:
        return df

    from src.standards import calculate_condition, calculate_rotorbar_severity

    work = df.copy()
    if 'Value' not in work.columns:
        work['Value'] = np.nan
    else:
        work['Value'] = pd.to_numeric(work['Value'], errors='coerce')
    work['Date'] = pd.to_datetime(work.get('Date', pd.NaT), errors='coerce')
    valid = work.dropna(subset=['Date'])
    if valid.empty:
        return work

    latest_dates = valid.groupby('Equipment', dropna=False)['Date'].max()

    rows_to_add = []
    idx_updates = []
    upd_values = []
    upd_raw_values = []
    upd_status = []

    def _safe_float(v):
        try:
            if v is None:
                return None
            if pd.isna(v):
                return None
            if isinstance(v, str) and v.strip() == '':
                return None
            return float(v)
        except Exception:
            return None

    def _get_param_value(g, name):
        r = g[g['Parameter'] == name]
        if r.empty:
            return None
        v = r.iloc[0].get('Value', None)
        v2 = _safe_float(v)
        if v2 is not None:
            return v2
        return r.iloc[0].get('Raw_Value', None)

    for eq, dt in latest_dates.items():
        g = valid[(valid['Equipment'] == eq) & (valid['Date'] == dt)]
        if g.empty:
            continue

        meta_unit = g['Unit_Name'].iloc[0] if 'Unit_Name' in g.columns else 'Unknown'
        meta_volt = g['Voltage_Level'].iloc[0] if 'Voltage_Level' in g.columns else 'Unknown'
        meta_full = g['Full_Name'].iloc[0] if 'Full_Name' in g.columns else eq

        params = {
            'Dev Voltage': _get_param_value(g, 'Dev Voltage'),
            'Dev Current': _get_param_value(g, 'Dev Current'),
            'THD Voltage %': _get_param_value(g, 'THD Voltage %'),
            'Upper Sideband': _get_param_value(g, 'Upper Sideband'),
            'Lower Sideband': _get_param_value(g, 'Lower Sideband'),
            'Rotorbar Health': _get_param_value(g, 'Rotorbar Health'),
            'Se Fund': _get_param_value(g, 'Se Fund'),
            'Se Harm': _get_param_value(g, 'Se Harm'),
            'Rotorbar Level %': _get_param_value(g, 'Rotorbar Level %'),
            'Bearing': _get_param_value(g, 'Bearing'),
        }

        cond = calculate_condition(params)
        rb = calculate_rotorbar_severity(params)

        to_upsert = []
        to_upsert.append(('Kondisi', cond.get('Overall', 'Normal'), np.nan, cond.get('Overall', 'Normal')))
        if cond.get('Bearing') is not None:
            to_upsert.append(('Bearing', cond.get('Bearing'), np.nan, cond.get('Bearing')))

        rb_status = rb.get('Status', 'Normal')
        rb_level = rb.get('Level', None)
        to_upsert.append(('Rotorbar', rb_status, np.nan, rb_status))
        if rb_level is not None:
            to_upsert.append(('Rotorbar Severity Level', int(rb_level), float(rb_level), 'Normal'))

        month_name = dt.strftime('%b').upper()
        year_val = int(dt.year)

        for param_name, raw_val, num_val, stat in to_upsert:
            mask = (work['Equipment'] == eq) & (work['Parameter'] == param_name) & (work['Date'] == dt)
            nv = np.nan
            if num_val is not None and not (isinstance(num_val, float) and np.isnan(num_val)):
                try:
                    nv = float(num_val)
                except Exception:
                    nv = np.nan
            if mask.any():
                row_idx = work[mask].index[0]
                idx_updates.append(row_idx)
                upd_raw_values.append(str(raw_val) if raw_val is not None else '')
                upd_values.append(nv)
                upd_status.append(str(stat) if stat is not None else 'Normal')
            else:
                rows_to_add.append({
                    'Equipment': eq,
                    'Parameter': param_name,
                    'Month': month_name,
                    'Year': year_val,
                    'Month_Name': month_name,
                    'Date': str(dt.date()),
                    'Raw_Value': str(raw_val) if raw_val is not None else '',
                    'Value': nv,
                    'Unit': '',
                    'Limit': '',
                    'Status': str(stat) if stat is not None else 'Normal',
                    'Unit_Name': meta_unit,
                    'Voltage_Level': meta_volt,
                    'Full_Name': meta_full,
                })

    if idx_updates:
        work.loc[idx_updates, 'Raw_Value'] = upd_raw_values
        work.loc[idx_updates, 'Value'] = upd_values
        work.loc[idx_updates, 'Status'] = upd_status

    if rows_to_add:
        work = pd.concat([work, pd.DataFrame(rows_to_add)], ignore_index=True)
    return work

def save_mcsa_data(df, original_file_path):
    """
    Saves the DataFrame to a CSV file to persist changes.
    """
    csv_path = os.path.join(os.path.dirname(original_file_path), 'mcsa_updated.csv')
    df = _normalize_date_columns(df)
    df = _normalize_status_columns(df)
    df = _normalize_unit_voltage_columns(df)
    df = _add_status_category(df)
    if 'Full_Name' not in df.columns:
        df['Full_Name'] = df['Equipment']
    df.to_csv(csv_path, index=False)
    return csv_path

def get_latest_data(df):
    """
    Returns the latest non-null data for each equipment and parameter.
    """
    if df.empty:
        return df

    df = _normalize_date_columns(df)

    df_sorted = df.sort_values(by=['Equipment', 'Parameter', 'Date'], ascending=[True, True, False])
    df_valid = df_sorted.dropna(subset=['Raw_Value'])
    df_latest = df_valid.drop_duplicates(subset=['Equipment', 'Parameter'])
    return df_latest

if __name__ == "__main__":
    # Test
    path = get_data_path('Report MCSA.xls')
    data = load_mcsa_data(path)
    print(data.head())
    print("Unique Parameters:", data['Parameter'].unique())
