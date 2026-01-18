import re
import os
import json

import pandas as pd
from src.data_loader import get_data_path


def _safe_float(v):
    try:
        if v is None:
            return None
        if isinstance(v, str) and v.strip() == '':
            return None
        if isinstance(v, str):
            s = v.strip().replace(',', '')
            s = re.sub(r'[^0-9eE+\-\.]', '', s)
            if s in {'', '+', '-', '.', '+.', '-.'}:
                return None
            v = s
        return float(v)
    except Exception:
        return None


def calculate_rotorbar_severity(params):
    upper = _safe_float(params.get('Upper Sideband'))
    lower = _safe_float(params.get('Lower Sideband'))
    rb_idx = _safe_float(params.get('Rotorbar Health'))

    se_fund = _safe_float(params.get('Se Fund'))
    se_harm = _safe_float(params.get('Se Harm'))
    level_pct = _safe_float(params.get('Rotorbar Level %'))

    sb_values = [v for v in [upper, lower] if v is not None]
    max_sb = max(sb_values) if sb_values else None

    level = None
    if max_sb is not None:
        if max_sb >= -45:
            level = 4
        elif max_sb >= -54:
            level = 3
        elif max_sb >= -60:
            level = 2
        else:
            level = 1

    if rb_idx is not None:
        if rb_idx >= 3.0:
            level = 4 if level is None else max(level, 4)
        elif rb_idx >= 1.0:
            level = 3 if level is None else max(level, 3)
        elif rb_idx >= 0.1:
            level = 2 if level is None else max(level, 2)

    if level_pct is None and se_fund is not None and se_fund != 0 and se_harm is not None:
        try:
            level_pct = abs(se_harm / se_fund) * 100.0
        except Exception:
            level_pct = None

    if level_pct is not None:
        if level_pct >= 5.0:
            level = 4 if level is None else max(level, 4)
        elif level_pct >= 2.0:
            level = 3 if level is None else max(level, 3)
        elif level_pct >= 0.5:
            level = 2 if level is None else max(level, 2)

    if level is None:
        status = 'Normal'
        assessment = 'Unknown'
    else:
        # Mapping 1-6 severity based on user request
        if level >= 6:
            status = 'CRITICAL / TRIP RISK'
            assessment = 'Very Poor / Failed Imminent'
        elif level == 5:
            status = 'Alarm'
            assessment = 'Poor'
        elif level == 4:
            status = 'Alarm'
            assessment = 'Fair'
        elif level == 3:
            status = 'Monitoring'
            assessment = 'Good'
        elif level == 2:
            status = 'Normal'
            assessment = 'Very Good'
        else: # level <= 1
            status = 'Normal'
            assessment = 'Excellent'

    return {
        'Level': level,
        'Status': status,
        'Assessment': assessment,
        'Max Sideband': max_sb,
        'RB Index': rb_idx,
        'Level %': level_pct,
    }


def calculate_condition(params):
    results = {
        'Rotorbar': 'Normal',
        'Unbalance_Voltage': 'Normal',
        'Unbalance_Current': 'Normal',
        'THD': 'Normal',
        'Bearing': 'Normal', # Default if no data
        'Overall': 'Normal'
    }
    
    rb = calculate_rotorbar_severity(params)
    results['Rotorbar'] = rb.get('Status', 'Normal')

    # --- 2. Unbalance Analysis ---
    # Voltage Unbalance (NEMA MG-1)
    # > 1% Warning, > 2% Alarm
    if 'Dev Voltage' in params and params['Dev Voltage'] is not None:
        val = _safe_float(params.get('Dev Voltage'))
        if val is not None:
            if val > 2.0:
                results['Unbalance_Voltage'] = 'High'
            elif val > 1.0:
                results['Unbalance_Voltage'] = 'Alarm'

    # Current Unbalance
    # > 10% Alarm, > 5% Warning
    if 'Dev Current' in params and params['Dev Current'] is not None:
        val = _safe_float(params.get('Dev Current'))
        if val is not None:
            if val > 10.0:
                results['Unbalance_Current'] = 'High'
            elif val > 5.0:
                results['Unbalance_Current'] = 'Alarm'

    # --- 3. Power Quality (THD) ---
    # IEEE 519: Voltage THD > 5% is bad
    if 'THD Voltage %' in params and params['THD Voltage %'] is not None:
        val = _safe_float(params.get('THD Voltage %'))
        if val is not None:
            if val > 8.0:
                results['THD'] = 'High'
            elif val > 5.0:
                results['THD'] = 'Alarm'

    # --- 4. Bearing ---
    # User Request: Normal or Bad.
    # Mapping: Bad -> High (Red)
    if 'Bearing' in params and isinstance(params['Bearing'], str):
        val = params['Bearing'].lower()
        if 'bad' in val or 'damage' in val or 'rusak' in val or 'high' in val:
            results['Bearing'] = 'High'
        elif 'alarm' in val or 'warning' in val:
            results['Bearing'] = 'Alarm'
        elif 'ok' in val or 'normal' in val:
            results['Bearing'] = 'Normal'

    # --- 5. Overall Calculation ---
    # Priority: High > Alarm > Normal
    severity_map = {
        'Normal': 1, 
        'Monitoring': 1,
        'Standby': 0,
        'Alarm': 2, 
        'Warning': 2, 
        'High': 3, 
        'Bad': 3,
        'High / Warning': 2, # Kept for backward compatibility if old data exists
        'CRITICAL / TRIP RISK': 3,
        'Critical': 3
    }
    
    current_max = 1
    for k, v in results.items():
        if k != 'Overall':
            score = severity_map.get(v, 1)
            if score > current_max:
                current_max = score
    
    final_status = 'Normal'
    if current_max == 2:
        final_status = 'Alarm' # User: Alarm = Kuning
    elif current_max == 3:
        final_status = 'High'  # User: High = Merah
        
    results['Overall'] = final_status
    return results


def generate_initial_analysis(history_df):
    if history_df is None or len(history_df) == 0:
        return {
            'indicators': [],
            'recommendations': [
                'Tidak ada data history untuk dianalisis.'
            ],
            'references': []
        }

    df = history_df.copy()
    if 'Parameter' not in df.columns:
        return {
            'indicators': [],
            'recommendations': [
                'Format data tidak memiliki kolom Parameter.'
            ],
            'references': []
        }

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    else:
        df['Date'] = pd.NaT

    df = df.dropna(subset=['Parameter'])
    df['Parameter'] = df['Parameter'].astype(str)

    df_sorted = df.sort_values(by=['Date'], ascending=True)
    latest_rows = df_sorted.groupby('Parameter', as_index=False).tail(1)
    latest_map = {row['Parameter']: row for _, row in latest_rows.iterrows()}

    def _row_val(param_name):
        row = latest_map.get(param_name)
        if row is None:
            return None
        v = row.get('Value') if 'Value' in row else None
        if v is None or (isinstance(v, float) and pd.isna(v)):
            v = row.get('Raw_Value') if 'Raw_Value' in row else None
        return v

    condition_params = {
        'Dev Voltage': _row_val('Dev Voltage'),
        'Dev Current': _row_val('Dev Current'),
        'THD Voltage %': _row_val('THD Voltage %'),
        'Bearing': _row_val('Bearing'),
        'Upper Sideband': _row_val('Upper Sideband'),
        'Lower Sideband': _row_val('Lower Sideband'),
        'Rotorbar Health': _row_val('Rotorbar Health'),
        'Se Fund': _row_val('Se Fund'),
        'Se Harm': _row_val('Se Harm'),
        'Rotorbar Level %': _row_val('Rotorbar Level %'),
    }
    status = calculate_condition(condition_params)

    def _trend(param_name):
        s = df_sorted[df_sorted['Parameter'] == param_name].copy()
        if s.empty:
            return None

        if 'Value' in s.columns:
            nums = s['Value'].map(_safe_float)
        else:
            nums = pd.Series([None] * len(s), index=s.index)
        if 'Raw_Value' in s.columns:
            nums = nums.fillna(s['Raw_Value'].map(_safe_float))
        s = s.assign(_num=nums).dropna(subset=['_num'])
        if len(s) < 2:
            return 'Stabil'

        last = float(s['_num'].iloc[-1])
        prev = float(s['_num'].iloc[-2])
        delta = last - prev
        eps = max(0.01, abs(prev) * 0.01)
        if abs(delta) <= eps:
            return 'Stabil'
        return 'Naik' if delta > 0 else 'Turun'

    def _last_num(param_name):
        v = _row_val(param_name)
        return _safe_float(v)

    def _last_date(param_name):
        row = latest_map.get(param_name)
        if row is None:
            return None
        dt = row.get('Date')
        if isinstance(dt, pd.Timestamp) and not pd.isna(dt):
            return dt.date().isoformat()
        return None

    indicator_specs = [
        ('Dev Voltage', 'Unbalance_Voltage'),
        ('Dev Current', 'Unbalance_Current'),
        ('THD Voltage %', 'THD'),
    ]

    indicators = []
    for p_name, status_key in indicator_specs:
        if p_name in latest_map:
            indicators.append({
                'Parameter': p_name,
                'Nilai Terakhir': _last_num(p_name),
                'Trend': _trend(p_name),
                'Status': status.get(status_key, 'Normal'),
                'Tanggal': _last_date(p_name)
            })

    if 'Bearing' in latest_map:
        indicators.append({
            'Parameter': 'Bearing',
            'Nilai Terakhir': _row_val('Bearing'),
            'Trend': None,
            'Status': status.get('Bearing', 'Normal'),
            'Tanggal': _last_date('Bearing')
        })

    if any(k in latest_map for k in ['Upper Sideband', 'Lower Sideband', 'Rotorbar Health', 'Rotorbar Level %']):
        rb = calculate_rotorbar_severity(condition_params)
        indicators.append({
            'Parameter': 'Rotorbar',
            'Nilai Terakhir': rb.get('Assessment'),
            'Trend': None,
            'Status': status.get('Rotorbar', 'Normal'),
            'Tanggal': _last_date('Rotorbar Health') or _last_date('Upper Sideband') or _last_date('Lower Sideband')
        })

    references = []
    recommendations = []

    overall = status.get('Overall', 'Normal')
    if overall == 'High':
        recommendations.append('Prioritaskan inspeksi dan verifikasi ulang hasil pengukuran; pertimbangkan pembatasan operasi hingga penyebab jelas.')
    elif overall == 'Alarm':
        recommendations.append('Jadwalkan pemeriksaan lanjutan dan lakukan trending lebih sering untuk parameter yang abnormal.')

    uv = status.get('Unbalance_Voltage', 'Normal')
    if uv in {'Alarm', 'High'}:
        recommendations.append('Cek ketidakseimbangan tegangan (suplai/terminal/koneksi), ketidakseimbangan beban satu fasa, dan kualitas sambungan; lakukan ukur ulang di titik yang sama.')
        references.append('NEMA MG 1 (Voltage unbalance guidance)')

    uc = status.get('Unbalance_Current', 'Normal')
    if uc in {'Alarm', 'High'}:
        recommendations.append('Jika arus tidak seimbang, periksa indikasi unbalance tegangan, koneksi, dan kondisi beban; pastikan konfigurasi CT dan pembacaan benar.')
        if 'NEMA MG 1 (Voltage unbalance guidance)' not in references:
            references.append('NEMA MG 1 (Voltage unbalance guidance)')

    thd = status.get('THD', 'Normal')
    if thd in {'Alarm', 'High'}:
        recommendations.append('Jika THD tegangan tinggi, identifikasi sumber harmonisa (mis. VFD/rectifier), evaluasi filter/reaktor, dan pastikan pengukuran di titik PCC sesuai praktik yang benar.')
        references.append('IEEE Std 519 (Harmonic control in power systems)')

    br = status.get('Bearing', 'Normal')
    if br in {'Alarm', 'High'}:
        recommendations.append('Tindak lanjuti indikasi bearing: cek pelumasan, alignment, looseness, dan lakukan inspeksi lanjutan (vibrasi/termografi) sesuai prosedur maintenance.')
        references.append('ISO 15243 (Rolling bearings — Damage and failures)')

    if not recommendations:
        recommendations.append('Tidak ada anomali utama terdeteksi pada indikator yang tersedia; lanjutkan monitoring berkala dan konsistenkan metode pengukuran.')

    references = list(dict.fromkeys([r for r in references if r]))

    return {
        'indicators': indicators,
        'recommendations': recommendations,
        'references': references
    }

def load_thresholds_config():
    base = get_data_path('config')
    paths = [
        os.path.join(base, 'thresholds_default.json'),
        os.path.join(base, 'thresholds_site_override.json'),
        os.path.join(base, 'thresholds_motor_override.json'),
    ]
    cfg = {}
    for p in paths:
        try:
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                for k, v in (data or {}).items():
                    cfg[k] = v
        except:
            pass
    return cfg
