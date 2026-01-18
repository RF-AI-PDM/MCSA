import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_loader import load_mcsa_data, get_latest_data, save_mcsa_data, get_folder_metadata, get_data_path, audit_mcsa_dataframe, fix_mcsa_dataframe, load_nameplate_csv
from src.docx_parser import parse_all_reports_with_report
from src.ppt_generator import create_ppt
from src.chatbot import MCSAChatbot
from src.standards import calculate_condition, generate_initial_analysis
import os
import json
from typing import Optional
from datetime import datetime

# Page Config
st.set_page_config(page_title="MCSA Dashboard & Chatbot", layout="wide")

# Title
st.title("⚡ RBot MCSA Visual Dashbot")

# Data Loading
# Remove cache_data to allow updates to reflect immediately
def load_data():
    file_path = get_data_path('Report MCSA.xls')
    # Ensure we load the latest available data (including updates)
    return load_mcsa_data(file_path)

if 'data_changed' not in st.session_state:
    st.session_state.data_changed = False

df = load_data()

if df.empty:
    st.error("Gagal memuat data atau file tidak ditemukan.")
    st.stop()

df['Date'] = pd.to_datetime(df.get('Date', pd.NaT), errors='coerce')

min_date = df['Date'].min()
max_date = df['Date'].max()

if pd.isna(min_date) or pd.isna(max_date):
    min_date = datetime.now().date()
    max_date = datetime.now().date()
else:
    min_date = min_date.date()
    max_date = max_date.date()

df_latest_all = get_latest_data(df)

# Sidebar
st.sidebar.header("Navigasi")

NAV_OPTIONS = ["Dashboard", "Manajemen Data", "Sync Laporan Word", "Materi Training", "Chatbot", "Laporan PPT"]

if 'page' not in st.session_state:
    st.session_state.page = NAV_OPTIONS[0]

# Ensure session state page is valid
if st.session_state.page not in NAV_OPTIONS:
    st.session_state.page = NAV_OPTIONS[0]

def on_nav_change():
    st.session_state.page = st.session_state.nav_selection

current_index = NAV_OPTIONS.index(st.session_state.page)
st.sidebar.radio(
    "Pilih Halaman",
    NAV_OPTIONS,
    index=current_index,
    key="nav_selection",
    on_change=on_nav_change
)

if 'edit_mode' not in st.session_state:
    st.session_state.edit_mode = False
st.sidebar.checkbox('Mode Edit (izinkan perubahan data)', key='edit_mode')

page = st.session_state.page

# --- FILTERING (Global) ---
st.sidebar.markdown("---")
st.sidebar.header("Filter Data")

sel_date_range = st.sidebar.date_input(
    "Periode (Tanggal Mulai - Akhir)",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(sel_date_range, (list, tuple)) and len(sel_date_range) == 2:
    date_start, date_end = sel_date_range
else:
    date_start, date_end = min_date, max_date

df_period = df.copy()
df_period['Date'] = pd.to_datetime(df_period.get('Date', pd.NaT), errors='coerce')
df_period = df_period[(df_period['Date'].dt.date >= date_start) & (df_period['Date'].dt.date <= date_end)]
df_latest = get_latest_data(df_period)
df_latest_for_filters = df_latest if not df_latest.empty else df_latest_all

# Get unique values for filters
unit_choices = ['All', 'UNIT 1', 'UNIT 2', 'UNIT 3', 'UNIT COMMON', 'Unknown']
units_present = set(str(x) for x in df_latest_for_filters.get('Unit_Name', pd.Series(dtype=str)).unique())
all_units = [u for u in unit_choices if u == 'All' or u in units_present] + sorted([u for u in units_present if u not in set(unit_choices)])
sel_unit = st.sidebar.selectbox("Pilih Unit:", all_units)

volt_choices = ['All', '380/400 V', '6.3 KV', 'Unknown']
volts_present = set(str(x) for x in df_latest_for_filters.get('Voltage_Level', pd.Series(dtype=str)).unique())
all_volts = [v for v in volt_choices if v == 'All' or v in volts_present] + sorted([v for v in volts_present if v not in set(volt_choices)])
sel_volt = st.sidebar.selectbox("Pilih Voltage:", all_volts)

standby_enabled = st.sidebar.checkbox('Standby otomatis jika tidak ada data bulan ini', value=True)
standby_scope = None
if standby_enabled:
    standby_scope = st.sidebar.selectbox('Cakupan Standby', ['Per Unit (mengikuti filter Unit/Voltage)', 'Semua Unit (abaikan filter Unit)'])

update_param_candidates = ['Kondisi', 'Load', 'Dev Voltage', 'Dev Current', 'THD Voltage %', 'Rotorbar Health', 'Upper Sideband', 'Lower Sideband', 'Bearing']
present_params = set(df.get('Parameter', pd.Series(dtype=str)).astype(str).unique())
update_param_options = [p for p in update_param_candidates if p in present_params] + sorted([p for p in present_params if p not in set(update_param_candidates)])
_wajib_params = ['Dev Voltage', 'Dev Current', 'THD Voltage %', 'Upper Sideband', 'Lower Sideband', 'Load']
default_required = [p for p in _wajib_params if p in update_param_options]
if not default_required:
    default_required = [p for p in ['Kondisi'] if p in update_param_options]
required_month_params = st.sidebar.multiselect('Parameter wajib update bulanan', update_param_options, default=default_required)
if not required_month_params:
    required_month_params = ['Kondisi']

df_latest_augmented = df_latest.copy()
standby_report = None
if standby_enabled:
    ref_dt = pd.Timestamp(date_end)
    month_start = ref_dt.replace(day=1)
    month_end = month_start + pd.offsets.MonthEnd(0)

    df_month = df.copy()
    df_month['Date'] = pd.to_datetime(df_month.get('Date', pd.NaT), errors='coerce')
    df_month = df_month[(df_month['Date'] >= month_start) & (df_month['Date'] <= month_end)]

    meta_df = df_latest_all.copy()
    if standby_scope and standby_scope.startswith('Per Unit'):
        if sel_unit != 'All' and 'Unit_Name' in meta_df.columns:
            meta_df = meta_df[meta_df['Unit_Name'] == sel_unit]
            if 'Unit_Name' in df_month.columns:
                df_month = df_month[df_month['Unit_Name'] == sel_unit]
        if sel_volt != 'All' and 'Voltage_Level' in meta_df.columns:
            meta_df = meta_df[meta_df['Voltage_Level'] == sel_volt]
            if 'Voltage_Level' in df_month.columns:
                df_month = df_month[df_month['Voltage_Level'] == sel_volt]
    else:
        if sel_volt != 'All' and 'Voltage_Level' in meta_df.columns:
            meta_df = meta_df[meta_df['Voltage_Level'] == sel_volt]
            if 'Voltage_Level' in df_month.columns:
                df_month = df_month[df_month['Voltage_Level'] == sel_volt]

    eq_universe = set(meta_df.get('Equipment', pd.Series(dtype=str)).astype(str).unique())

    df_month_req = df_month.copy()
    df_month_req = df_month_req[df_month_req.get('Parameter', pd.Series(dtype=str)).astype(str).isin(required_month_params)]
    if 'Raw_Value' in df_month_req.columns:
        rv = df_month_req['Raw_Value']
        df_month_req = df_month_req[rv.notna() & rv.astype(str).str.strip().ne('')]
    eq_present = set(df_month_req.get('Equipment', pd.Series(dtype=str)).astype(str).unique())
    standby_eq = [e for e in sorted(eq_universe - eq_present) if e and e.lower() not in {'nan', 'none'}]

    # Load standby reasons config
    reasons_path = get_data_path('config', 'standby_reasons.json')
    reasons_data = {}
    try:
        if os.path.exists(reasons_path):
            with open(reasons_path, 'r', encoding='utf-8') as fp:
                reasons_data = json.load(fp)
    except:
        reasons_data = {}

    month_key = month_start.strftime('%Y-%m')
    reasons_for_month = reasons_data.get(month_key, {})

    standby_report = {
        'month_start': month_start,
        'month_end': month_end,
        'required_params': required_month_params,
        'scope': standby_scope,
        'sel_unit': sel_unit,
        'sel_volt': sel_volt,
        'eq_universe': sorted([e for e in eq_universe if e and e.lower() not in {'nan', 'none'}]),
        'eq_present': sorted([e for e in eq_present if e and e.lower() not in {'nan', 'none'}]),
        'eq_missing': standby_eq,
        'standby_reasons': reasons_for_month,
    }

    if standby_eq:
        meta_map = meta_df.drop_duplicates(subset=['Equipment']).set_index('Equipment') if 'Equipment' in meta_df.columns else pd.DataFrame()
        month_name = month_start.strftime('%b').upper()
        year_val = int(month_start.year)
        standby_rows = []
        for eq in standby_eq:
            unit_name = 'Unknown'
            volt_name = 'Unknown'
            full_name = eq
            if not meta_map.empty and eq in meta_map.index:
                r = meta_map.loc[eq]
                if isinstance(r, pd.DataFrame):
                    r = r.iloc[0]
                unit_name = r.get('Unit_Name', unit_name)
                volt_name = r.get('Voltage_Level', volt_name)
                full_name = r.get('Full_Name', full_name)
            standby_rows.append({
                'Equipment': eq,
                'Parameter': 'Kondisi',
                'Month': month_name,
                'Year': year_val,
                'Month_Name': month_name,
                'Date': str(month_end.date()),
                'Raw_Value': 'Standby',
                'Value': None,
                'Unit': '',
                'Limit': '',
                'Status': 'Standby',
                'Status_Category': 'Standby',
                'Status_Level': 0,
                'Unit_Name': unit_name,
                'Voltage_Level': volt_name,
                'Full_Name': full_name
            })
        df_latest_augmented = pd.concat([df_latest_augmented, pd.DataFrame(standby_rows)], ignore_index=True)

# Apply filters
filtered_df = df_latest_augmented.copy()
if sel_unit != 'All':
    filtered_df = filtered_df[filtered_df['Unit_Name'] == sel_unit]
if sel_volt != 'All':
    filtered_df = filtered_df[filtered_df['Voltage_Level'] == sel_volt]

# Use filtered_df for Dashboard, but keep full df for management if needed (or filter there too)

# --- DASHBOARD PAGE ---
if page == "Dashboard":
    st.header("Overview Kondisi Equipment")

    st.caption(f"Periode: {date_start} s/d {date_end}")

    unit_label = sel_unit if sel_unit != 'All' else 'Semua Unit'
    volt_label = sel_volt if sel_volt != 'All' else 'Semua Voltage'
    st.caption(f"{unit_label} | {volt_label}")

    if standby_report is not None:
        with st.expander('Kepatuhan Sampling Bulanan', expanded=False):
            month_label = standby_report['month_start'].strftime('%Y-%m')
            required_label = ', '.join(standby_report.get('required_params') or [])
            st.caption(f"Bulan: {month_label} | Parameter wajib: {required_label}")

            universe = standby_report.get('eq_universe') or []
            present = set(standby_report.get('eq_present') or [])
            missing = standby_report.get('eq_missing') or []
            reasons_for_month = standby_report.get('standby_reasons') or {}
            excluded = [e for e in missing if e in reasons_for_month and str(reasons_for_month[e].get('reason','')).strip() != '']
            missing_visible = [e for e in missing if e not in excluded]

            total_expected = len(universe)
            updated = len(present)
            missing_cnt = len(missing_visible)
            compliance_pct = 0.0 if total_expected == 0 else (updated / total_expected) * 100.0

            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric('Expected', total_expected)
            cc2.metric('Updated', updated)
            cc3.metric('Belum Update', missing_cnt)
            cc4.metric('Compliance %', round(compliance_pct, 1))

            # Load quality classification based on Load rule
            if present:
                dfm = df_month.copy()
                dfm['Parameter'] = dfm['Parameter'].astype(str)
                load_rows = dfm[dfm['Parameter'] == 'Load']
                load_vals = pd.to_numeric(load_rows.get('Value', pd.NA), errors='coerce')
                if 'Raw_Value' in load_rows.columns:
                    load_vals = load_vals.fillna(pd.to_numeric(load_rows['Raw_Value'], errors='coerce'))
                load_rows = load_rows.assign(_load=load_vals)
                eq_load_last = load_rows.sort_values('Date').dropna(subset=['_load']).drop_duplicates(subset=['Equipment'], keep='last')
                class_map = {}
                for _, r in eq_load_last.iterrows():
                    lv = float(r['_load'])
                    if lv < 20.0:
                        class_map[r['Equipment']] = 'Invalid (<20%)'
                    elif lv < 40.0:
                        class_map[r['Equipment']] = 'Monitoring Only (20–40%)'
                    else:
                        class_map[r['Equipment']] = 'Valid Diagnosis (≥40%)'
                # Fallback using Current and FLA from nameplate
                name_df = load_nameplate_csv()
                if not name_df.empty:
                    fla_map = name_df.set_index('Equipment')['FLA'].to_dict() if 'FLA' in name_df.columns else {}
                    for eq in present:
                        if eq not in class_map and eq in fla_map and fla_map[eq] not in {None, '', 'nan'}:
                            curr_rows = dfm[(dfm['Equipment'] == eq) & (dfm['Parameter'].isin(['Current 1','Current 2','Current 3']))]
                            curr_vals = pd.to_numeric(curr_rows.get('Value', pd.NA), errors='coerce')
                            if 'Raw_Value' in curr_rows.columns:
                                curr_vals = curr_vals.fillna(pd.to_numeric(curr_rows['Raw_Value'], errors='coerce'))
                            if curr_vals.notna().any():
                                avg_i = float(curr_vals.dropna().mean())
                                try:
                                    fla = float(fla_map[eq])
                                    if fla > 0:
                                        lv = (avg_i / fla) * 100.0
                                        if lv < 20.0:
                                            class_map[eq] = 'Invalid (<20%)'
                                        elif lv < 40.0:
                                            class_map[eq] = 'Monitoring Only (20–40%)'
                                        else:
                                            class_map[eq] = 'Valid Diagnosis (≥40%)'
                                except Exception:
                                    pass
                valid_cnt = sum(1 for e in present if class_map.get(e) == 'Valid Diagnosis (≥40%)')
                mon_cnt = sum(1 for e in present if class_map.get(e) == 'Monitoring Only (20–40%)')
                inv_cnt = sum(1 for e in present if class_map.get(e) == 'Invalid (<20%)')
                cv1, cv2, cv3 = st.columns(3)
                cv1.metric('Valid Diagnosis', valid_cnt)
                cv2.metric('Monitoring Only', mon_cnt)
                cv3.metric('Invalid (Load<20%)', inv_cnt)

            if missing_cnt:
                miss_df = pd.DataFrame({'Equipment': missing_visible})
                st.dataframe(miss_df, use_container_width=True, hide_index=True)
                st.download_button(
                    'Download CSV (Belum Update)',
                    data=miss_df.to_csv(index=False).encode('utf-8'),
                    file_name=f"belum_update_{month_label}.csv",
                    mime='text/csv'
                )

            st.markdown('---')
            st.subheader('Alasan Standby (pengecualian kepatuhan)')
            reason_enum = ['PLANNED_SHUTDOWN','UNPLANNED_OUTAGE','MAINTENANCE','UNIT_OFF','VFD_BYPASS','STARTUP_TEST','OTHER']
            new_reasons = {}
            for eq in missing:
                c1, c2 = st.columns([2, 3])
                with c1:
                    sel_reason = st.selectbox(f"{eq}", reason_enum, index=reason_enum.index(reasons_for_month.get(eq, {}).get('reason','OTHER')) if reasons_for_month.get(eq) else reason_enum.index('OTHER'))
                note_val = ''
                with c2:
                    note_val = st.text_input(f"Catatan ({eq})", value=reasons_for_month.get(eq, {}).get('note',''))
                new_reasons[eq] = {'reason': sel_reason, 'note': note_val}
            if st.button('Simpan Alasan Standby'):
                reasons_data[month_key] = new_reasons
                try:
                    os.makedirs(os.path.dirname(reasons_path), exist_ok=True)
                    with open(reasons_path, 'w', encoding='utf-8') as fp:
                        json.dump(reasons_data, fp, ensure_ascii=False, indent=2)
                    st.success('Alasan standby disimpan.')
                except Exception as e:
                    st.error(f'Gagal menyimpan: {e}')

    # Metrics based on Filtered Data
    total_eq = filtered_df['Equipment'].nunique()
    
    # Count Conditions
    cond_rows = filtered_df[filtered_df['Parameter'] == 'Kondisi']
    if 'Status_Category' in cond_rows.columns:
        status_series = cond_rows['Status_Category'].astype(str)
    else:
        status_series = cond_rows['Raw_Value'].astype(str)

    status_lower = status_series.str.strip().str.lower()
    normal_count = (status_lower == 'normal').sum()
    alarm_count = (status_lower == 'alarm').sum()
    high_count = (status_lower == 'high').sum()
    standby_count = (status_lower == 'standby').sum()
    unknown_count = (~status_lower.isin(['normal', 'alarm', 'high', 'standby'])).sum()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Equipment", total_eq)
    col2.metric("Normal (Hijau)", int(normal_count))
    col3.metric("Standby", int(standby_count))
    col4.metric("Alarm (Kuning)", int(alarm_count), delta_color="inverse")
    col5.metric("High (Merah)", int(high_count), delta_color="inverse")
    
    # Charts
    st.subheader("Distribusi Kondisi")
    if not cond_rows.empty:
        chart_df = cond_rows.copy()
        if 'Status_Category' in chart_df.columns:
            chart_df['Status_Display'] = chart_df['Status_Category']
        else:
            chart_df['Status_Display'] = chart_df['Raw_Value']

        color_map = {
            'Normal': 'green',
            'Standby': 'gray',
            'Alarm': 'yellow',
            'High': 'red',
            'normal': 'green',
            'standby': 'gray',
            'alarm': 'yellow',
            'high': 'red'
        }
        fig = px.pie(chart_df, names='Status_Display', title='Status Equipment', color='Status_Display', color_discrete_map=color_map)
        st.plotly_chart(fig)
    else:
        st.info("Tidak ada data untuk filter ini.")
        
    # Detail Table
    st.subheader("Data Detail")
    # Only show equipments in the filtered list
    eq_list = filtered_df['Equipment'].unique()
    if len(eq_list) > 0:
        # Create a display map for the selectbox
        # "Code (Full Name)"
        display_map = {}
        for eq in eq_list:
            row = filtered_df[filtered_df['Equipment'] == eq].iloc[0]
            full_name = row['Full_Name'] if 'Full_Name' in row and pd.notna(row['Full_Name']) else eq
            if full_name != eq:
                display_map[f"{eq} ({full_name})"] = eq
            else:
                display_map[eq] = eq
                
        selected_label = st.selectbox("Pilih Equipment:", list(display_map.keys()), key="dashboard_eq")
        selected_eq = display_map[selected_label]
        
        eq_data = filtered_df[filtered_df['Equipment'] == selected_eq]
        
        # Show Unit info
        u_name = eq_data['Unit_Name'].iloc[0] if 'Unit_Name' in eq_data.columns else '-'
        u_volt = eq_data['Voltage_Level'].iloc[0] if 'Voltage_Level' in eq_data.columns else '-'
        f_name = eq_data['Full_Name'].iloc[0] if 'Full_Name' in eq_data.columns else '-'
        
        st.markdown(f"### {f_name}")
        st.caption(f"**Code:** {selected_eq} | **Unit:** {u_name} | **Voltage:** {u_volt}")
        
        # Show calculated specific conditions if available
        # Add color highlighting for status
        def highlight_status(val):
            val_lower = str(val).lower()
            if 'normal' in val_lower:
                return 'background-color: #90EE90; color: black' # Light Green
            elif 'standby' in val_lower:
                return 'background-color: #E0E0E0; color: black' # Light Gray
            elif 'alarm' in val_lower:
                return 'background-color: #FFFFE0; color: black' # Light Yellow
            elif 'monitoring' in val_lower:
                 return 'background-color: #ADD8E6; color: black' # Light Blue
            elif 'high' in val_lower or 'warning' in val_lower:
                return 'background-color: #FFB6C1; color: black' # Light Red
            elif 'critical' in val_lower or 'trip' in val_lower:
                return 'background-color: #FF0000; color: white' # Red
            elif 'bad' in val_lower:
                return 'background-color: #FFB6C1; color: black' # Light Red
            return ''

        st.dataframe(eq_data[['Parameter', 'Raw_Value', 'Unit', 'Status']].style.applymap(highlight_status, subset=['Status']))

        st.subheader('Analisa & Rekomendasi Awal')
        eq_history = df[df['Equipment'] == selected_eq].copy()
        analysis = generate_initial_analysis(eq_history)
        indicators = analysis.get('indicators') or []
        if indicators:
            st.dataframe(pd.DataFrame(indicators), use_container_width=True, hide_index=True)

        def _to_num(v):
            if v is None:
                return None
            if isinstance(v, (int, float)):
                if pd.isna(v):
                    return None
                return float(v)
            s = str(v).strip().replace(',', '')
            if s == '':
                return None
            s = ''.join(ch for ch in s if (ch.isdigit() or ch in {'.', '-', '+', 'e', 'E'}))
            if s in {'', '+', '-', '.', '+.', '-.'}:
                return None
            try:
                return float(s)
            except Exception:
                return None

        key_params = [
            'Load',
            'Dev Voltage',
            'Dev Current',
            'THD Voltage %',
            'Upper Sideband',
            'Lower Sideband',
            'Rotorbar Health',
            'Rotorbar Level %',
        ]

        trend_src = eq_history.copy()
        if 'Date' in trend_src.columns:
            trend_src['Date'] = pd.to_datetime(trend_src['Date'], errors='coerce')
        else:
            trend_src['Date'] = pd.NaT
        trend_src = trend_src.dropna(subset=['Date'])
        trend_src = trend_src[trend_src['Parameter'].astype(str).isin(key_params)].copy()
        if not trend_src.empty:
            trend_src['Trend_Value'] = trend_src.get('Value', pd.Series([None] * len(trend_src), index=trend_src.index)).map(_to_num)
            if 'Raw_Value' in trend_src.columns:
                trend_src['Trend_Value'] = trend_src['Trend_Value'].fillna(trend_src['Raw_Value'].map(_to_num))
            trend_src = trend_src.dropna(subset=['Trend_Value'])
            if not trend_src.empty:
                trend_src = trend_src.sort_values('Date')
                fig_trend = px.line(
                    trend_src,
                    x='Date',
                    y='Trend_Value',
                    color='Parameter',
                    markers=True,
                    title='Trend Parameter Kunci'
                )
                st.plotly_chart(fig_trend, use_container_width=True)
        recs = analysis.get('recommendations') or []
        if recs:
            st.markdown('\n'.join([f"- {r}" for r in recs]))
        refs = analysis.get('references') or []
        if refs:
            st.caption('Referensi: ' + ' | '.join(refs))

        def _classify_status(text: str) -> str:
            s = str(text or '').strip().lower()
            if 'high' in s or 'bad' in s or 'critical' in s or 'rusak' in s or 'damage' in s:
                return 'high'
            if 'alarm' in s or 'warning' in s:
                return 'alarm'
            if 'standby' in s:
                return 'standby'
            if 'normal' in s or s == 'ok' or 'good' in s:
                return 'normal'
            return 'unknown'

        def _open_materi(query_text: str = '', prefer_name_contains: Optional[str] = None):
            st.session_state['materi_query'] = query_text
            st.session_state['materi_prefer_name_contains'] = prefer_name_contains
            st.session_state['page'] = 'Materi Training'
            st.rerun()

        cond_now = ''
        cond_row = eq_data[eq_data['Parameter'] == 'Kondisi']
        if not cond_row.empty:
            r0 = cond_row.iloc[0]
            if 'Status_Category' in cond_row.columns and pd.notna(r0.get('Status_Category')):
                cond_now = r0.get('Status_Category')
            else:
                cond_now = r0.get('Raw_Value')
        cond_class = _classify_status(cond_now)

        st.subheader('Materi Terkait')
        colm1, colm2, colm3, colm4, colm5 = st.columns(5)
        if colm1.button('SOP Pengukuran', use_container_width=True):
            _open_materi('sop', 'sop')
        if colm2.button('Rotor Bar', use_container_width=True):
            _open_materi('rotor bar', 'mcsa')
        if colm3.button('Bearing', use_container_width=True):
            _open_materi('bearing', 'mcsa')
        if colm4.button('Power Quality', use_container_width=True):
            _open_materi('power quality', 'power')
        if colm5.button('Pattern Recognition', use_container_width=True):
            _open_materi('pattern', 'pattern')

        if cond_class in {'alarm', 'high'}:
            label = 'Tindak Lanjut (Alarm/High)'
            if st.button(label, use_container_width=True):
                _open_materi('tindak lanjut', 'sop')

        perf_rows = eq_data[eq_data['Parameter'].astype(str).str.startswith('Ringkasan Kinerja')].copy()
        if perf_rows.empty:
            fallback_perf = df_latest_all[df_latest_all['Equipment'] == selected_eq]
            perf_rows = fallback_perf[fallback_perf['Parameter'].astype(str).str.startswith('Ringkasan Kinerja')].copy()

        if not perf_rows.empty:
            st.subheader('Ringkasan Performance')
            perf_rows['Bagian'] = perf_rows['Parameter'].astype(str).str.replace('Ringkasan Kinerja -', '', regex=False).str.strip()
            show_perf = perf_rows[['Bagian', 'Raw_Value']].rename(columns={'Raw_Value': 'Ringkasan'}).drop_duplicates(subset=['Bagian'], keep='last')
            show_perf = show_perf.sort_values('Bagian')
            st.dataframe(show_perf, use_container_width=True, hide_index=True)

        # --- Spectrum Analysis Section ---
        st.subheader("Analisis Spektrum")
        
        # Image directory (adjust path as needed)
        root_dir = os.path.dirname(os.path.abspath(__file__))
        img_dir = os.path.join(root_dir, 'data', 'images')
        
        spectrum_images = []
        if os.path.exists(img_dir):
            for f in os.listdir(img_dir):
                # Simple matching: starts with Equipment Name
                if f.upper().startswith(f"{selected_eq.upper()}_") and f.lower().endswith(".png"):
                    spectrum_images.append(f)
        
        spectrum_images.sort(reverse=True) # Newest date first
        
        if spectrum_images:
            c_img1, c_img2 = st.columns([1, 2])
            with c_img1:
                sel_img = st.selectbox("Pilih Gambar Spektrum", spectrum_images)
            with c_img2:
                if sel_img:
                    st.image(os.path.join(img_dir, sel_img), caption=sel_img, use_container_width=True)
        else:
            st.info("Tidak ada gambar spektrum yang tersedia untuk equipment ini.")

        # Historical Trend (using full df for history)
        st.subheader("Trend Parameter")
        # Controls for trend range and aggregation
        trend_params = sorted(df['Parameter'].unique())
        param_trend = st.selectbox("Pilih Parameter untuk Trend:", trend_params, index=trend_params.index('Load') if 'Load' in trend_params else 0)

        trend_range = st.radio("Rentang Waktu", ["3 Bulan", "6 Bulan", "12 Bulan", "Semua"], horizontal=True)
        agg_choice = st.radio("Agregasi", ["Harian", "Bulanan", "Tahunan"], horizontal=True)

        # Base data for selected equipment & parameter across full df (not just filtered period)
        base_all = df[(df['Equipment'] == selected_eq) & (df['Parameter'] == param_trend)].copy()
        base_all['Date'] = pd.to_datetime(base_all.get('Date', pd.NaT), errors='coerce')
        base_all = base_all.dropna(subset=['Date']).sort_values('Date')

        base = base_all.copy()

        # Limit range by selected window
        if not base.empty:
            last_date = base['Date'].max()
            if trend_range == "3 Bulan":
                start_date = last_date - pd.DateOffset(months=3)
            elif trend_range == "6 Bulan":
                start_date = last_date - pd.DateOffset(months=6)
            elif trend_range == "12 Bulan":
                start_date = last_date - pd.DateOffset(months=12)
            else:
                start_date = base['Date'].min()
            base = base[base['Date'] >= start_date]

        if param_trend in ['Kondisi', 'Bearing']:
            show = base.copy()
            if 'Status_Category' not in show.columns:
                show['Status_Category'] = show.get('Raw_Value', '')
            if 'Status_Level' not in show.columns:
                s = show.get('Status_Category', '').astype(str).str.strip().str.lower()
                lvl = pd.Series(-1, index=show.index)
                lvl[s.str.contains('standby', na=False)] = 0
                lvl[s.str.contains('normal|\bok\b|good', na=False)] = 1
                lvl[s.str.contains('alarm|warning', na=False)] = 2
                lvl[s.str.contains('high|bad|critical|rusak|damage', na=False)] = 3
                show['Status_Level'] = lvl

            if agg_choice == "Bulanan":
                show['YearMonth'] = show['Date'].dt.to_period('M').astype(str)
                show = show.sort_values('Date').drop_duplicates(subset=['YearMonth'], keep='last')
                x_col, title = 'YearMonth', f"Trend Bulanan {param_trend} - {selected_eq}"
            elif agg_choice == "Tahunan":
                show['Year'] = show['Date'].dt.year
                show = show.sort_values('Date').drop_duplicates(subset=['Year'], keep='last')
                x_col, title = 'Year', f"Trend Tahunan {param_trend} - {selected_eq}"
            else:
                x_col, title = 'Date', f"Trend Harian {param_trend} - {selected_eq}"

            if not show.empty:
                fig_trend = px.line(
                    show,
                    x=x_col,
                    y='Status_Level',
                    title=title,
                    markers=True,
                    hover_data={'Status_Category': True, 'Raw_Value': True}
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("Belum ada data untuk menampilkan trend parameter ini.")
        else:
            show = base.copy()
            show['Value_num'] = pd.to_numeric(show.get('Value', pd.NA), errors='coerce')
            if 'Raw_Value' in show.columns:
                show['Value_num'] = show['Value_num'].fillna(pd.to_numeric(show['Raw_Value'], errors='coerce'))
            if agg_choice == "Bulanan":
                show['YearMonth'] = show['Date'].dt.to_period('M').astype(str)
                show = show.groupby('YearMonth', as_index=False)['Value_num'].mean()
                x_col, y_col, title = 'YearMonth', 'Value_num', f"Trend Bulanan {param_trend} - {selected_eq}"
            elif agg_choice == "Tahunan":
                show['Year'] = show['Date'].dt.year
                show = show.groupby('Year', as_index=False)['Value_num'].mean()
                x_col, y_col, title = 'Year', 'Value_num', f"Trend Tahunan {param_trend} - {selected_eq}"
            else:
                x_col, y_col, title = 'Date', 'Value_num', f"Trend Harian {param_trend} - {selected_eq}"

            if not show.empty and show[y_col].notna().any():
                fig_trend = px.line(show, x=x_col, y=y_col, title=title, markers=True)
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("Belum ada data numerik untuk menampilkan trend parameter ini.")

        with st.expander("Perbandingan Bulanan/Tahunan", expanded=True):
            monthly = base_all.copy()
            monthly['YearMonth'] = monthly['Date'].dt.to_period('M').astype(str)

            if param_trend in ['Kondisi', 'Bearing']:
                if 'Status_Category' not in monthly.columns:
                    monthly['Status_Category'] = monthly.get('Raw_Value', '')
                rv = monthly.get('Raw_Value', pd.Series('', index=monthly.index)).astype(str).str.strip().str.lower()
                has_any = monthly['Status_Category'].notna() & rv.ne('') & rv.ne('nan')
                monthly = monthly[has_any].sort_values('Date').drop_duplicates(subset=['YearMonth'], keep='last')
                month_options = monthly['YearMonth'].tolist()

                if not month_options:
                    st.info("Belum ada data untuk perbandingan bulanan/tahunan.")
                else:
                    sel_month = st.selectbox("Pilih Bulan (YYYY-MM)", month_options, index=len(month_options) - 1, key=f"month_cmp_{selected_eq}_{param_trend}")
                    sel_idx = month_options.index(sel_month)
                    prev_month = month_options[sel_idx - 1] if sel_idx > 0 else None

                    this_row = monthly[monthly['YearMonth'] == sel_month].iloc[0]
                    this_cat = str(this_row.get('Status_Category', this_row.get('Raw_Value', '')))

                    prev_cat = None
                    if prev_month:
                        prev_row = monthly[monthly['YearMonth'] == prev_month].iloc[0]
                        prev_cat = str(prev_row.get('Status_Category', prev_row.get('Raw_Value', '')))

                    year, month = sel_month.split('-')
                    same_month_last_year = f"{int(year) - 1:04d}-{month}"
                    yoy_cat = None
                    if same_month_last_year in month_options:
                        yoy_row = monthly[monthly['YearMonth'] == same_month_last_year].iloc[0]
                        yoy_cat = str(yoy_row.get('Status_Category', yoy_row.get('Raw_Value', '')))

                    c1, c2, c3 = st.columns(3)
                    c1.metric(f"Status {sel_month}", this_cat)
                    c2.metric("Status Bulan Sebelumnya", prev_cat if prev_cat is not None else "-")
                    c3.metric("Status Bulan Sama Tahun Lalu", yoy_cat if yoy_cat is not None else "-")

                    st.dataframe(monthly[['YearMonth', 'Status_Category', 'Raw_Value']].sort_values('YearMonth', ascending=False).head(24), use_container_width=True)
            else:
                monthly['Value_num'] = pd.to_numeric(monthly.get('Value', pd.NA), errors='coerce')
                if 'Raw_Value' in monthly.columns:
                    monthly['Value_num'] = monthly['Value_num'].fillna(pd.to_numeric(monthly['Raw_Value'], errors='coerce'))
                if 'Raw_Value' in monthly.columns:
                    rv = monthly['Raw_Value'].astype(str).str.strip().str.lower()
                    has_any = monthly['Value_num'].notna() | (monthly['Raw_Value'].notna() & rv.ne('') & rv.ne('nan'))
                else:
                    has_any = monthly['Value_num'].notna()

                monthly = monthly[has_any].sort_values('Date').drop_duplicates(subset=['YearMonth'], keep='last')
                month_options = monthly['YearMonth'].tolist()

                if not month_options:
                    st.info("Belum ada data untuk perbandingan bulanan/tahunan.")
                else:
                    sel_month = st.selectbox("Pilih Bulan (YYYY-MM)", month_options, index=len(month_options) - 1, key=f"month_cmp_{selected_eq}_{param_trend}")
                    sel_idx = month_options.index(sel_month)
                    prev_month = month_options[sel_idx - 1] if sel_idx > 0 else None

                    this_row = monthly[monthly['YearMonth'] == sel_month].iloc[0]
                    this_num = this_row['Value_num'] if pd.notna(this_row['Value_num']) else None
                    this_raw = this_row.get('Raw_Value', None)

                    prev_num = None
                    prev_raw = None
                    if prev_month:
                        prev_row = monthly[monthly['YearMonth'] == prev_month].iloc[0]
                        prev_num = prev_row['Value_num'] if pd.notna(prev_row['Value_num']) else None
                        prev_raw = prev_row.get('Raw_Value', None)

                    year, month = sel_month.split('-')
                    same_month_last_year = f"{int(year) - 1:04d}-{month}"
                    yoy_num = None
                    yoy_raw = None
                    if same_month_last_year in month_options:
                        yoy_row = monthly[monthly['YearMonth'] == same_month_last_year].iloc[0]
                        yoy_num = yoy_row['Value_num'] if pd.notna(yoy_row['Value_num']) else None
                        yoy_raw = yoy_row.get('Raw_Value', None)

                    c1, c2, c3 = st.columns(3)
                    if this_num is not None:
                        c1.metric(f"Nilai {sel_month}", float(this_num))
                    else:
                        c1.metric(f"Nilai {sel_month}", str(this_raw))

                    if prev_month:
                        if prev_num is not None:
                            delta_val = None if this_num is None else float(this_num - prev_num)
                            c2.metric(f"Nilai {prev_month}", float(prev_num), delta=delta_val)
                        else:
                            c2.metric(f"Nilai {prev_month}", str(prev_raw))
                    else:
                        c2.metric("Nilai Bulan Sebelumnya", "-")

                    if same_month_last_year in month_options:
                        if yoy_num is not None:
                            delta_yoy = None if this_num is None else float(this_num - yoy_num)
                            c3.metric(f"Nilai {same_month_last_year}", float(yoy_num), delta=delta_yoy)
                        else:
                            c3.metric(f"Nilai {same_month_last_year}", str(yoy_raw))
                    else:
                        c3.metric("Nilai Bulan Sama Tahun Lalu", "-")

                    table_cols = ['YearMonth']
                    if monthly['Value_num'].notna().any():
                        table_cols.append('Value_num')
                    if 'Raw_Value' in monthly.columns:
                        table_cols.append('Raw_Value')
                    st.dataframe(monthly[table_cols].sort_values('YearMonth', ascending=False).head(24), use_container_width=True)
    else:
        st.warning("Tidak ada equipment yang sesuai filter.")

# --- MANAJEMEN DATA (NEW) ---
elif page == "Manajemen Data":
    st.header("📝 Manajemen Data MCSA")
    st.info("Update nilai parameter, info Unit, atau hapus data equipment.")

    edit_mode = bool(st.session_state.get('edit_mode', False))
    if not edit_mode:
        st.warning('Mode Edit nonaktif. Aksi yang mengubah data dinonaktifkan.')

    with st.expander('Data Health Check', expanded=False):
        file_path = get_data_path('Report MCSA.xls')
        st.caption(f"Sumber data: {file_path}")
        audit = audit_mcsa_dataframe(df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Rows', int(audit.get('rows', 0)))
        c2.metric('Invalid Date', int(audit.get('invalid_date_count', 0)))
        c3.metric('Duplicate Key', int(audit.get('duplicate_key_count', 0)))
        c4.metric('Missing Cols', len(audit.get('missing_columns') or []))

        missing_cols = audit.get('missing_columns') or []
        if missing_cols:
            st.error('Kolom wajib hilang: ' + ', '.join(missing_cols))

        bad_dates = audit.get('invalid_date_sample')
        if isinstance(bad_dates, pd.DataFrame) and not bad_dates.empty:
            st.subheader('Contoh Date tidak valid')
            st.dataframe(bad_dates, use_container_width=True, hide_index=True)

        dup = audit.get('duplicate_key_sample')
        if isinstance(dup, pd.DataFrame) and not dup.empty:
            st.subheader('Contoh duplikat (Equipment, Parameter, Date)')
            st.dataframe(dup, use_container_width=True, hide_index=True)

        if st.button('Perbaiki Otomatis (ringan)', disabled=not edit_mode):
            fixed_df, rep = fix_mcsa_dataframe(df)
            save_mcsa_data(fixed_df, file_path)
            st.success(f"Selesai. Rows: {rep.get('rows_before')} → {rep.get('rows_after')}. Duplikat dihapus: {rep.get('duplicates_removed')}. Date diperbaiki: {rep.get('repaired_dates')}.")
            st.rerun()
    
    action = st.radio("Aksi:", ["Edit/Update Data", "Hapus Data Equipment", "Reload dari Excel (Reset)"])
    
    # Allow searching from ALL equipment, or filtered? Let's use filtered for convenience, but maybe full list is better.
    # User might want to edit something not in current filter. Let's use full list but sorted.
    all_eq_list = sorted(df_latest_all['Equipment'].unique())
    selected_eq_manage = st.selectbox("Pilih Equipment:", all_eq_list, key="manage_eq")
    
    if action == "Edit/Update Data":
        st.subheader(f"Edit Data: {selected_eq_manage}")
        
        # Get current values to pre-fill form
        current_data = df_latest_all[df_latest_all['Equipment'] == selected_eq_manage]
        
        # Helper to get value
        def get_val(param):
            row = current_data[current_data['Parameter'] == param]
            if not row.empty:
                return row['Raw_Value'].values[0]
            return ""

        # Current Unit/Volt
        curr_unit = current_data['Unit_Name'].iloc[0] if not current_data.empty else "Unknown"
        curr_volt = current_data['Voltage_Level'].iloc[0] if not current_data.empty else "Unknown"

        with st.form("edit_form"):
            # --- Metadata Section ---
            st.markdown("### Info Equipment")
            c1, c2 = st.columns(2)
            with c1:
                new_unit = st.selectbox("Unit Name", ["UNIT 1", "UNIT 2", "UNIT 3", "UNIT COMMON", "Unknown"], index=0 if curr_unit not in ["UNIT 1", "UNIT 2", "UNIT 3", "UNIT COMMON"] else ["UNIT 1", "UNIT 2", "UNIT 3", "UNIT COMMON", "Unknown"].index(curr_unit))
            with c2:
                new_volt = st.selectbox("Voltage Level", ["380/400 V", "6.3 KV", "Unknown"], index=0 if curr_volt == "380/400 V" else (1 if curr_volt == "6.3 KV" else 2))

            # --- Parameter Section ---
            upd_date = st.date_input("Tanggal Update", value=datetime.now().date())

            st.markdown("### Parameter MCSA")
            col1, col2 = st.columns(2)
            
            # List of editable parameters
            params_to_edit = [
                'Load', 'Current 1', 'Current 2', 'Current 3', 'Dev Current',
                'Voltage 1', 'Voltage 2', 'Voltage 3', 'Dev Voltage',
                'power factor', 'THD Current %', 'THD Voltage %', 'Real Power',
                'Rotorbar Health', 'Upper Sideband', 'Lower Sideband',
                'Se Fund', 'Se Harm', 'Rotorbar Level %',
                'Rotorbar', 'Rotorbar Severity Level',
                'Bearing'
            ]
            
            new_values = {}
            for i, param in enumerate(params_to_edit):
                with col1 if i % 2 == 0 else col2:
                    val = st.text_input(f"{param}", value=str(get_val(param)))
                    new_values[param] = val
            
            st.markdown("---")
            st.markdown("**Hasil Perhitungan Otomatis akan memperbarui 'Kondisi' dan 'Bearing'**")
            
            submitted = st.form_submit_button("Hitung & Simpan", disabled=not edit_mode)
            
            if submitted:
                # 1. Calculate Conditions
                calc_results = calculate_condition(new_values)
                st.success(f"Hasil Perhitungan: {calc_results}")
                
                # 2. Update Data
                # We append new rows for 'UPDATED' month
                upd_dt = pd.to_datetime(upd_date)
                m_name = upd_dt.strftime('%b').upper()
                y_val = int(upd_dt.year)
                new_rows = []
                
                # Add explicit parameters
                for p, v in new_values.items():
                    if v and v != 'nan':
                        new_rows.append({
                            'Equipment': selected_eq_manage,
                            'Parameter': p,
                            'Month': m_name,
                            'Year': y_val,
                            'Month_Name': m_name,
                            'Date': str(upd_date),
                            'Raw_Value': v,
                            'Value': float(v) if v.replace('.','',1).replace('-','',1).isdigit() else None,
                            'Unit': '', # Keep simple
                            'Limit': '',
                            'Status': 'Normal',
                            'Unit_Name': new_unit,       # Save metadata
                            'Voltage_Level': new_volt    # Save metadata
                        })
                
                # Add Calculated Parameters (Kondisi, Bearing status if updated)
                new_rows.append({
                    'Equipment': selected_eq_manage,
                    'Parameter': 'Kondisi',
                    'Month': m_name,
                    'Year': y_val,
                    'Month_Name': m_name,
                    'Date': str(upd_date),
                    'Raw_Value': calc_results['Overall'],
                    'Value': None,
                    'Unit': '', 'Limit': '', 'Status': calc_results['Overall'],
                    'Unit_Name': new_unit,
                    'Voltage_Level': new_volt
                })
                
                # Update Bearing status specifically if calculated
                if 'Bearing' in calc_results:
                     new_rows.append({
                        'Equipment': selected_eq_manage,
                        'Parameter': 'Bearing',
                        'Month': m_name,
                        'Year': y_val,
                        'Month_Name': m_name,
                        'Date': str(upd_date),
                        'Raw_Value': calc_results['Bearing'],
                        'Value': None,
                        'Unit': '', 'Limit': '', 'Status': calc_results['Bearing'],
                        'Unit_Name': new_unit,
                        'Voltage_Level': new_volt
                    })
                
                # Also update metadata for ALL past rows of this equipment?
                # Ideally yes, to keep consistency.
                # Update existing rows in 'df' for this equipment
                df.loc[df['Equipment'] == selected_eq_manage, 'Unit_Name'] = new_unit
                df.loc[df['Equipment'] == selected_eq_manage, 'Voltage_Level'] = new_volt
                
                new_df_rows = pd.DataFrame(new_rows)
                
                # Combine with existing DF (append)
                updated_df = pd.concat([df, new_df_rows], ignore_index=True)
                
                # Save
                file_path = get_data_path('Report MCSA.xls')
                save_mcsa_data(updated_df, file_path)
                
                st.session_state.data_changed = True
                st.success("Data berhasil disimpan.")
                st.rerun()
                
    elif action == "Hapus Data Equipment":
        st.warning(f"Apakah Anda yakin ingin menghapus SEMUA data untuk {selected_eq_manage}?")
        if st.button("Ya, Hapus Permanen", disabled=not edit_mode):
            # Filter out this equipment
            cleaned_df = df[df['Equipment'] != selected_eq_manage]
            
            file_path = get_data_path('Report MCSA.xls')
            save_mcsa_data(cleaned_df, file_path)
            
            st.session_state.data_changed = True
            st.success(f"Data {selected_eq_manage} telah dihapus.")
            st.rerun()

    elif action == "Reload dari Excel (Reset)":
        st.warning("⚠️ **PERINGATAN**: Tindakan ini akan membaca ulang file Excel (`Report MCSA.xls`) dan Laporan Word, lalu menimpa database CSV saat ini. Semua perubahan manual yang Anda lakukan di aplikasi akan hilang jika belum disimpan ke file sumber.")
        
        if st.button("Ya, Reload Ulang Semua Data", type="primary", disabled=not edit_mode):
            with st.spinner("Membaca ulang data dari Excel & Word..."):
                file_path = get_data_path('Report MCSA.xls')
                # Load with force_excel=True to ignore existing CSV
                df_new = load_mcsa_data(file_path, force_excel=True)
                
                # Save to CSV to persist the reset
                save_mcsa_data(df_new, file_path)
                
                st.session_state.data_changed = True
            
            st.success("Database berhasil di-reset ulang dari sumber Excel & Word!")
            st.rerun()

elif page == "Sync Laporan Word":
    st.header("📥 Sync & Upload Laporan Word")

    edit_mode = bool(st.session_state.get('edit_mode', False))
    if not edit_mode:
        st.warning('Mode Edit nonaktif. Upload dan sinkronisasi dinonaktifkan.')

    laporan_root = get_data_path('Laporan')
    if not os.path.exists(laporan_root):
        os.makedirs(laporan_root)

    file_path = get_data_path('Report MCSA.xls')
    st.caption(f"Sumber laporan: {laporan_root}")

    # --- UPLOAD SECTION ---
    st.subheader("Upload Laporan Baru")
    uploaded_files = st.file_uploader("Pilih file laporan (.docx)", type=['docx', 'docm'], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button(f"Upload & Proses {len(uploaded_files)} File", disabled=not edit_mode):
            progress_bar = st.progress(0)
            for i, uploaded_file in enumerate(uploaded_files):
                save_path = os.path.join(laporan_root, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            st.success(f"{len(uploaded_files)} file berhasil disimpan ke folder Laporan.")
            
            # Auto-trigger sync
            with st.spinner("Mengekstrak data dan gambar dari laporan..."):
                meta = get_folder_metadata(laporan_root)
                df_word, rep = parse_all_reports_with_report(laporan_root, meta)
                
                # Merge logic
                if not df_word.empty:
                    df_word['Date'] = pd.to_datetime(df_word.get('Date', pd.NaT), errors='coerce')
                    merged = pd.concat([df, df_word], ignore_index=True)
                    merged['Date'] = pd.to_datetime(merged.get('Date', pd.NaT), errors='coerce')
                    merged = merged.drop_duplicates(subset=['Equipment', 'Parameter', 'Date'], keep='last')
                    save_mcsa_data(merged, file_path)
                    
                    st.success("Data berhasil diekstrak dan database diperbarui!")
                    st.rerun()
                else:
                    st.warning("File terupload, namun tidak ada data valid yang bisa diekstrak.")

    st.markdown("---")

    required_params = [
        'Load',
        'Current 1', 'Current 2', 'Current 3', 'Dev Current',
        'Voltage 1', 'Voltage 2', 'Voltage 3', 'Dev Voltage',
        'power factor',
        'THD Current %', 'THD Voltage %',
        'Real Power',
        'Rotorbar Health',
        'Upper Sideband', 'Lower Sideband',
        'Bearing',
        'Kondisi'
    ]

    st.subheader("Sinkronisasi Manual (Scan Folder)")
    if st.button("Scan Semua File di Folder Laporan", type="primary", disabled=not edit_mode):
        with st.spinner("Memproses laporan Word..."):
            meta = get_folder_metadata(laporan_root)
            df_word, rep = parse_all_reports_with_report(laporan_root, meta)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total File", int(rep.get('total_files', 0)))
        c2.metric("Berhasil", int(rep.get('parsed_files', 0)))
        c3.metric("Gagal", int(rep.get('failed_files', 0)))

        if rep.get('failures'):
            st.subheader("File Gagal Diproses")
            st.dataframe(pd.DataFrame(rep['failures']), use_container_width=True)

        if df_word.empty:
            st.warning("Tidak ada data Word yang berhasil diparsing.")
        else:
            df_word['Date'] = pd.to_datetime(df_word.get('Date', pd.NaT), errors='coerce')
            st.subheader("Ringkasan Missing Parameter")
            latest_word = df_word.sort_values('Date').drop_duplicates(subset=['Equipment', 'Parameter'], keep='last')
            present = latest_word.groupby('Equipment')['Parameter'].apply(lambda s: set(s.astype(str))).to_dict()
            summary_rows = []
            for eq, params in present.items():
                missing = [p for p in required_params if p not in params]
                if missing:
                    summary_rows.append({'Equipment': eq, 'Missing_Count': len(missing), 'Missing': ', '.join(missing)})
            if summary_rows:
                st.dataframe(pd.DataFrame(summary_rows).sort_values(['Missing_Count', 'Equipment'], ascending=[False, True]), use_container_width=True)
            else:
                st.success("Semua equipment memiliki parameter minimum lengkap.")

            st.subheader("Quality Check Data Numerik")
            qc = latest_word.copy()
            qc['Value_num'] = pd.to_numeric(qc.get('Value', pd.NA), errors='coerce').fillna(pd.to_numeric(qc.get('Raw_Value', pd.NA), errors='coerce'))
            qc = qc[qc['Parameter'].isin(['Load', 'THD Current %', 'THD Voltage %', 'Dev Voltage', 'Dev Current']) & qc['Value_num'].notna()]
            flags = []
            for _, r in qc.iterrows():
                p = r['Parameter']
                v = float(r['Value_num'])
                bad = False
                if p == 'Load' and (v < 0 or v > 120):
                    bad = True
                if p in {'THD Current %', 'THD Voltage %'} and (v < 0 or v > 50):
                    bad = True
                if p in {'Dev Voltage', 'Dev Current'} and (v < 0 or v > 50):
                    bad = True
                if bad:
                    flags.append({'Equipment': r['Equipment'], 'Parameter': p, 'Date': r.get('Date', ''), 'Value': v})

            if flags:
                st.dataframe(pd.DataFrame(flags), use_container_width=True)
            else:
                st.success("Tidak ditemukan anomali numerik pada parameter utama.")

            merged = pd.concat([df, df_word], ignore_index=True)
            merged['Date'] = pd.to_datetime(merged.get('Date', pd.NaT), errors='coerce')
            merged = merged.drop_duplicates(subset=['Equipment', 'Parameter', 'Date'], keep='last')
            save_mcsa_data(merged, file_path)
            st.success("Sync selesai dan data tersimpan ke mcsa_updated.csv.")
            st.rerun()

elif page == "Materi Training":
    st.header("📚 Materi Training")

    base_dir = os.path.dirname(__file__)
    materi_dir = os.path.join(base_dir, "Materi")
    if not os.path.exists(materi_dir):
        st.warning(f"Folder tidak ditemukan: {materi_dir}")
        st.stop()

    materi_files = [f for f in os.listdir(materi_dir) if f.lower().endswith('.json')]
    materi_files = sorted(materi_files, key=lambda s: s.lower())
    if not materi_files:
        st.info("Tidak ada file materi (.json) di folder Materi.")
        st.stop()

    left, right = st.columns([2, 3])
    with left:
        prefer_contains = (st.session_state.get('materi_prefer_name_contains') or '').strip().lower()
        if prefer_contains:
            st.session_state['materi_prefer_name_contains'] = ''
            preferred = [i for i, f in enumerate(materi_files) if prefer_contains in f.lower()]
            if preferred:
                st.session_state['materi_selected_file'] = materi_files[preferred[0]]

        selected_file = st.selectbox(
            "Pilih Materi",
            materi_files,
            key='materi_selected_file'
        )
        query = st.text_input("Cari (kata kunci)", value=st.session_state.get('materi_query', ''))
        st.session_state['materi_query'] = query
        show_mode = st.radio("Tampilan", ["Per Bagian", "Hasil Pencarian"], horizontal=True)

    path = os.path.join(materi_dir, selected_file)
    mtime = None
    try:
        mtime = os.path.getmtime(path)
    except Exception:
        mtime = None

    @st.cache_data(show_spinner=False)
    def _load_json(file_path: str, mtime_key: Optional[float]):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    try:
        data = _load_json(path, mtime)
    except Exception as e:
        st.error(f"Gagal membaca materi: {e}")
        st.stop()

    def _to_str(x) -> str:
        if x is None:
            return ''
        return str(x)

    def _normalize_v2(doc: dict) -> dict:
        sections = []
        raw_sections = doc.get('sections') if isinstance(doc, dict) else None
        if isinstance(raw_sections, list):
            for s in raw_sections:
                if not isinstance(s, dict):
                    continue
                sections.append({
                    'id': s.get('id'),
                    'heading': _to_str(s.get('heading') or s.get('title') or '').strip(),
                    'content': _to_str(s.get('content') or '').strip()
                })
        return {
            'id': doc.get('id'),
            'title': _to_str(doc.get('title') or doc.get('name') or '').strip(),
            'tags': [ _to_str(t).strip() for t in (doc.get('tags') or []) if _to_str(t).strip() ],
            'level': _to_str(doc.get('level') or '').strip(),
            'source': _to_str(doc.get('source') or '').strip(),
            'language': _to_str(doc.get('language') or '').strip(),
            'sections': [s for s in sections if s.get('heading') or s.get('content')]
        }

    def _normalize_v1(pages_list: list, fallback_title: str) -> dict:
        sections = []
        for item in pages_list:
            if not isinstance(item, dict):
                continue
            pg = item.get('page')
            heading = f"Halaman {pg}" if pg is not None else "Halaman"
            sections.append({
                'id': pg,
                'heading': heading,
                'content': _to_str(item.get('content') or '').strip()
            })
        return {
            'id': None,
            'title': fallback_title,
            'tags': [],
            'level': '',
            'source': '',
            'language': '',
            'sections': [s for s in sections if s.get('content')]
        }

    docs = []
    if isinstance(data, dict) and isinstance(data.get('sections'), list):
        docs = [_normalize_v2(data)]
    elif isinstance(data, list):
        if any(isinstance(x, dict) and isinstance(x.get('sections'), list) for x in data):
            docs = [_normalize_v2(x) for x in data if isinstance(x, dict) and isinstance(x.get('sections'), list)]
        else:
            docs = [_normalize_v1(data, os.path.splitext(selected_file)[0])]
    elif isinstance(data, dict) and isinstance(data.get('pages'), list):
        docs = [_normalize_v1(data.get('pages') or [], os.path.splitext(selected_file)[0])]

    docs = [d for d in docs if d.get('sections')]
    if not docs:
        st.warning("Format materi tidak dikenali atau kosong.")
        st.stop()

    available_languages = sorted({(d.get('language') or '').strip().lower() for d in docs if (d.get('language') or '').strip()})
    lang_options = ['All'] + [l.upper() for l in available_languages]
    with left:
        sel_lang = st.selectbox('Language', lang_options, index=0)

    filtered_docs = docs
    if sel_lang != 'All':
        want = sel_lang.strip().lower()
        filtered_docs = [d for d in filtered_docs if (d.get('language') or '').strip().lower() == want]
        if not filtered_docs:
            filtered_docs = docs

    if len(filtered_docs) > 1:
        titles = []
        for d in filtered_docs:
            t = d.get('title') or d.get('id') or 'Materi'
            titles.append(t)
        with left:
            selected_doc_title = st.selectbox('Pilih Topik', titles)
        selected_doc = next((d for d, t in zip(filtered_docs, titles) if t == selected_doc_title), filtered_docs[0])
    else:
        selected_doc = filtered_docs[0]

    with right:
        st.caption(f"Sumber: {path}")

        title = selected_doc.get('title') or os.path.splitext(selected_file)[0]
        st.subheader(title)

        meta = []
        tags = selected_doc.get('tags') or []
        if tags:
            meta.append('Tags: ' + ', '.join(tags))
        if selected_doc.get('level'):
            meta.append('Level: ' + selected_doc.get('level'))
        if selected_doc.get('source'):
            meta.append('Source: ' + selected_doc.get('source'))
        if selected_doc.get('language'):
            meta.append('Language: ' + selected_doc.get('language'))
        if meta:
            st.caption(' | '.join(meta))

        sections = selected_doc.get('sections') or []
        query_text = query.strip().lower()

        def _section_text(sec: dict) -> str:
            return (sec.get('heading', '') + '\n' + sec.get('content', '') + '\n' + ' '.join(tags)).lower()

        if show_mode == "Hasil Pencarian" and query_text:
            matches = [s for s in sections if query_text in _section_text(s)]
            st.write(f"Ditemukan {len(matches)} bagian yang cocok.")
            for s in matches[:50]:
                heading = s.get('heading') or 'Bagian'
                with st.expander(heading, expanded=False):
                    st.text(s.get('content', ''))
        else:
            st.write(f"Total bagian: {len(sections)}")
            for s in sections[:200]:
                heading = s.get('heading') or 'Bagian'
                with st.expander(heading, expanded=False):
                    st.text(s.get('content', ''))

# --- CHATBOT PAGE ---
elif page == "Chatbot":
    st.header("🤖 MCSA Virtual Assistant")
    
    bot = MCSAChatbot(df_latest_augmented, df_all=df)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Tanya kondisi motor..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = bot.process_query(prompt)
        
        with st.chat_message("assistant"):
            st.markdown(response)
            if getattr(bot, 'last_export', None):
                st.download_button(
                    'Download CSV (Jawaban)',
                    data=str(bot.last_export).encode('utf-8'),
                    file_name='chatbot_export.csv',
                    mime='text/csv'
                )
        st.session_state.messages.append({"role": "assistant", "content": response})

# --- REPORT PAGE ---
elif page == "Laporan PPT":
    st.header("📄 Generate Laporan PPT")
    st.write("Klik tombol di bawah untuk mengunduh laporan status equipment dalam format PowerPoint.")
    
    if st.button("Generate PPT"):
        with st.spinner("Sedang membuat PPT..."):
            ppt_io = create_ppt(df_latest)
            
            st.download_button(
                label="Download Laporan MCSA (.pptx)",
                data=ppt_io,
                file_name="Laporan_MCSA.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
            )
            st.success("PPT Siap diunduh!")
