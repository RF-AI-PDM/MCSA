import pandas as pd
from src.data_loader import load_mcsa_data, get_latest_data, get_data_path
import os, json
from src.ppt_generator import create_ppt
from src.chatbot import MCSAChatbot

def verify_all():
    print("--- 1. Loading Data ---")
    df = load_mcsa_data(get_data_path('Report MCSA.xls'))
    print(f"Loaded {len(df)} rows.")
    
    print("\n--- 2. Getting Latest Data ---")
    df_latest = get_latest_data(df)
    print(f"Latest data has {len(df_latest)} rows.")
    print("Sample Equipment:", df_latest['Equipment'].iloc[0])
    
    print("\n--- 3. Testing Chatbot ---")
    bot = MCSAChatbot(df_latest, df_all=df)
    print("Query: 'List Alarm'")
    print("Response:", bot.process_query("List Alarm"))
    
    eq_name = df_latest['Equipment'].iloc[0]
    print(f"Query: 'Status {eq_name}'")
    print("Response:", bot.process_query(f"Status {eq_name}"))
    
    print("\n--- 4. Testing PPT Generation ---")
    ppt_io = create_ppt(df_latest)
    size = ppt_io.getbuffer().nbytes
    print(f"PPT Generated. Size: {size} bytes")
    
    print("\n--- 5. Testing Materi Loading ---")
    base_dir = os.path.dirname(__file__)
    materi_dir = os.path.join(base_dir, 'Materi')
    if not os.path.exists(materi_dir):
        print(f"Materi folder not found: {materi_dir}")
    else:
        files = [f for f in os.listdir(materi_dir) if f.lower().endswith('.json')]
        print(f"Found {len(files)} materi files")
        ok = 0
        fail = []
        for fn in files:
            path = os.path.join(materi_dir, fn)
            try:
                with open(path, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                # quick schema check
                is_v2_single = isinstance(data, dict) and isinstance(data.get('sections'), list)
                is_v2_array = isinstance(data, list) and any(isinstance(x, dict) and isinstance(x.get('sections'), list) for x in data)
                is_v1_pages = (isinstance(data, list) and all(isinstance(x, dict) and ('content' in x) for x in data)) or (isinstance(data, dict) and isinstance(data.get('pages'), list))
                if not (is_v2_single or is_v2_array or is_v1_pages):
                    raise ValueError('Unknown schema')
                ok += 1
            except Exception as e:
                fail.append((fn, str(e)))
        print(f"Materi OK: {ok}, Failed: {len(fail)}")
        if fail:
            for fn, msg in fail[:10]:
                print(f" - {fn}: {msg}")

    print("\n✅ Verification Complete!")

if __name__ == "__main__":
    verify_all()
