import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.standards import generate_initial_analysis

class MCSAChatbot:
    def __init__(self, df_latest, df_all=None):
        self.df_latest = df_latest
        self.df_all = df_all if df_all is not None else df_latest
        self.df = df_latest
        self.equipments = self.df_latest['Equipment'].unique() if not self.df_latest.empty else []
        self.equipments_lower = [str(e).lower() for e in self.equipments]
        self.last_export = None
        
    def process_query(self, query, context=None):
        query = query.lower().strip()
        context = context or {}
        
        response = ""
        suggestions = []
        new_context = context.copy()

        # Intent: List Alarm / Warning
        if "alarm" in query or "warning" in query or "masalah" in query:
            response = self.get_alarm_list()
            suggestions = ["Status Unit 1", "Status Unit 2", "Status Unit 3"]
            return response, suggestions, new_context

        # Intent: Status per Unit / Tegangan / Bulan
        unit = None
        volt = None
        month_key = None
        m = re.search(r"unit\s*(1|2|3|common)", query)
        if m:
            u = m.group(1).upper()
            unit = 'UNIT ' + u if u in {'1','2','3'} else 'UNIT COMMON'
        if '6.3' in query or 'kv' in query:
            volt = '6.3 KV'
        elif '400' in query or '380' in query or 'v' in query:
            volt = '380/400 V'
        m2 = re.search(r"(20\d{2}-\d{2})", query)
        if m2:
            month_key = m2.group(1)
        
        if unit or volt or month_key:
            response = self.get_group_status(unit=unit, volt=volt, month_key=month_key)
            suggestions = ["List Alarm", "Help"]
            return response, suggestions, new_context
            
        # Intent: Status/Condition of specific asset
        # Check if any equipment name is in the query
        found_eq = None
        for eq, eq_lower in zip(self.equipments, self.equipments_lower):
            if eq_lower in query:
                found_eq = eq
                break
        
        # Context Handling: If no equipment found in query, check context
        if not found_eq and context.get('last_equipment'):
            # Check for follow-up keywords
            if any(k in query for k in ['analisa', 'rekomendasi', 'saran', 'trend', 'detail', 'status']):
                found_eq = context.get('last_equipment')

        if found_eq:
            # Update context
            new_context['last_equipment'] = found_eq
            
            # Specific sub-intents
            if "trend" in query:
                 # Generate trend chart for the equipment
                 chart_fig = self.generate_trend_chart(found_eq)
                 if chart_fig:
                     response = f"📈 **Trend {found_eq}**\nBerikut adalah grafik trend untuk equipment ini:"
                     suggestions = [f"Status {found_eq}", f"Analisa {found_eq}", "List Alarm"]
                     # Store chart in context for display in app
                     new_context['trend_chart'] = chart_fig
                 else:
                     response = f"📈 **Trend {found_eq}**:\nData trend tidak tersedia untuk equipment ini.\nSilakan lihat grafik detail di menu **Dashboard** > pilih {found_eq}."
                     suggestions = [f"Status {found_eq}", f"Analisa {found_eq}", "List Alarm"]
            elif "analisa" in query or "rekomendasi" in query:
                 # Reuse get_asset_status but focus on analysis
                 full_status = self.get_asset_status(found_eq)
                 # Extract analysis part if possible, or just show full status
                 response = full_status
                 suggestions = [f"Trend {found_eq}", "List Alarm"]
            else:
                 response = self.get_asset_status(found_eq)
                 suggestions = [f"Analisa {found_eq}", f"Trend {found_eq}", "List Alarm"]
            
            return response, suggestions, new_context
            
        # Intent: Help / Hello
        if "halo" in query or "hi" in query or "help" in query or "menu" in query:
            response = "Halo! Saya MCSA Bot. Anda bisa tanya:\n1. 'Status [Nama Equipment]' (misal: Status BC102)\n2. 'List Alarm' untuk lihat equipment bermasalah.\n3. 'Status Unit 1' atau 'Status 6.3 KV'."
            suggestions = ["List Alarm", "Status Unit 1", "Status 6.3 KV"]
            return response, suggestions, new_context
            
        response = "Maaf, saya tidak mengerti. Coba sebutkan nama equipment atau ketik 'List Alarm'."
        suggestions = ["List Alarm", "Help"]
        return response, suggestions, new_context

    def get_alarm_list(self):
        # Filter where 'Kondisi' is not normal/standby
        cond_rows = self.df_latest[self.df_latest['Parameter'] == 'Kondisi']
        alarms = cond_rows[~cond_rows['Raw_Value'].str.lower().isin(['normal', 'standby'])]
        
        if alarms.empty:
            return "✅ Semua equipment dalam kondisi Normal/Standby."
        
        response = "⚠️ **Daftar Equipment Warning/Alarm:**\n"
        for _, row in alarms.iterrows():
            response += f"- **{row['Equipment']}**: {row['Raw_Value'].upper()}\n"
        return response

    def get_group_status(self, unit=None, volt=None, month_key=None):
        df = self.df_all.copy()
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if unit and 'Unit_Name' in df.columns:
            df = df[df['Unit_Name'].astype(str).str.upper() == unit]
        if volt and 'Voltage_Level' in df.columns:
            df = df[df['Voltage_Level'].astype(str) == volt]
        if month_key:
            try:
                year, month = month_key.split('-')
                start = pd.Timestamp(int(year), int(month), 1)
                end = start + pd.offsets.MonthEnd(0)
                df = df[(df['Date'] >= start) & (df['Date'] <= end)]
            except Exception:
                pass
        cond = df[df['Parameter'] == 'Kondisi']
        if cond.empty:
            return 'Tidak ada data kondisi untuk filter ini.'
        latest = cond.sort_values('Date').drop_duplicates(subset=['Equipment'], keep='last')
        status_map = latest[['Equipment','Raw_Value']]
        lines = [f"- {r['Equipment']}: {str(r['Raw_Value']).upper()}" for _, r in status_map.iterrows()]
        text = f"Ringkasan Status{(' ' + unit) if unit else ''}{(' ' + volt) if volt else ''}{(' ' + month_key) if month_key else ''} (Total {len(latest)}):\n" + "\n".join(lines)
        self.last_export = status_map.to_csv(index=False)
        return text

    def get_asset_status(self, eq_name):
        eq_latest = self.df_latest[self.df_latest['Equipment'] == eq_name]
        if eq_latest.empty:
            return f"Data untuk {eq_name} tidak ditemukan."

        eq_hist = self.df_all[self.df_all['Equipment'] == eq_name] if not self.df_all.empty else pd.DataFrame()
            
        # Get Condition
        cond_row = eq_latest[eq_latest['Parameter'] == 'Kondisi']
        condition = cond_row['Raw_Value'].iloc[0] if not cond_row.empty else "Unknown"

        last_dt = None
        if not eq_hist.empty and 'Date' in eq_hist.columns:
            dt = pd.to_datetime(eq_hist['Date'], errors='coerce')
            k_mask = eq_hist.get('Parameter', pd.Series(dtype=str)).astype(str).eq('Kondisi')
            dt_k = dt[k_mask]
            if dt_k.notna().any():
                last_dt = dt_k.max()
            elif dt.notna().any():
                last_dt = dt.max()

        last_dt_str = '-'
        if isinstance(last_dt, pd.Timestamp) and not pd.isna(last_dt):
            last_dt_str = last_dt.date().isoformat()
        
        # Get Key Parameters (e.g. Sidebands, Load)
        # We can list all or top few
        response = f"📊 **Status {eq_name}**: {condition.upper()}\n\n"

        response += f"**Tanggal Sample Terakhir:** {last_dt_str}\n\n"
        
        response += "**Parameter Terakhir:**\n"
        for _, row in eq_latest.iterrows():
            param = row['Parameter']
            val = row['Raw_Value']
            unit = row['Unit'] if pd.notna(row['Unit']) else ""
            if param != 'Kondisi':
                response += f"- {param}: {val} {unit}\n"

        if not eq_hist.empty:
            analysis = generate_initial_analysis(eq_hist)
            recs = analysis.get('recommendations') or []
            if recs:
                response += "\n**Analisa & Rekomendasi Awal:**\n"
                for r in recs[:8]:
                    response += f"- {r}\n"
            refs = analysis.get('references') or []
            if refs:
                response += "\n**Referensi:** " + ' | '.join(refs) + "\n"
                
        return response

    def generate_trend_chart(self, eq_name):
        """Generate trend chart for specific equipment"""
        if self.df_all is None or self.df_all.empty:
            return None
            
        # Filter data for the specific equipment
        eq_data = self.df_all[self.df_all['Equipment'] == eq_name].copy()
        if eq_data.empty:
            return None
            
        # Convert date column if exists
        if 'Date' in eq_data.columns:
            eq_data['Date'] = pd.to_datetime(eq_data['Date'], errors='coerce')
            eq_data = eq_data.dropna(subset=['Date'])
        
        if eq_data.empty:
            return None
            
        # Get numeric parameters for trending
        numeric_params = []
        for param in eq_data['Parameter'].unique():
            param_data = eq_data[eq_data['Parameter'] == param]
            # Try to convert to numeric
            try:
                pd.to_numeric(param_data['Raw_Value'], errors='raise')
                numeric_params.append(param)
            except:
                continue
        
        if not numeric_params:
            return None
            
        # Create subplot for each numeric parameter
        fig = make_subplots(
            rows=len(numeric_params), cols=1,
            subplot_titles=[f"Trend {param}" for param in numeric_params],
            vertical_spacing=0.1
        )
        
        for i, param in enumerate(numeric_params, 1):
            param_data = eq_data[eq_data['Parameter'] == param].copy()
            param_data = param_data.sort_values('Date')
            
            # Convert to numeric
            param_data['Value_numeric'] = pd.to_numeric(param_data['Raw_Value'], errors='coerce')
            param_data = param_data.dropna(subset=['Value_numeric'])
            
            if not param_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=param_data['Date'],
                        y=param_data['Value_numeric'],
                        mode='lines+markers',
                        name=param,
                        hovertemplate='<b>%{x}</b><br>%{y:.2f}<extra></extra>'
                    ),
                    row=i, col=1
                )
        
        if len(fig.data) == 0:
            return None
            
        fig.update_layout(
            height=200 * len(numeric_params),
            showlegend=True,
            title_text=f"Trend Analysis - {eq_name}",
            hovermode='x unified'
        )
        
        return fig
