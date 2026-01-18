# 🔧 MCSA Assistant

<p align="center">
  <b>Dashboard & Intelligent Assistant for Motor Current Signature Analysis (MCSA)</b><br/>
  <i>From raw current signals to actionable maintenance insight</i>
</p>

---

## 🌟 Overview

**MCSA Assistant** adalah aplikasi **Dashboard Analitik & Chatbot Teknis** yang dirancang khusus untuk mendukung analisis **Motor Current Signature Analysis (MCSA)** secara terstruktur, konsisten, dan siap industri.

Aplikasi ini membantu engineer dan tim maintenance dalam:

* Monitoring kondisi motor listrik
* Deteksi dini fault (Rotor Bar, Bearing, Electrical Fault)
* Standarisasi severity & status
* Otomatisasi pelaporan teknis

---

## 🧩 Core Modules & Features

### 📊 1. Condition Monitoring Dashboard

* Status equipment:

  * 🟢 Normal
  * 🟡 High
  * 🔴 Alarm
  * ⚪ Standby
* Filter dinamis:

  * Unit
  * Voltage Level
* Tampilan ringkas untuk fleet monitoring

---

### 📈 2. Trend & Historical Analysis

* Grafik tren untuk **parameter numerik**:

  * Line Current
  * Sideband Amplitude (dB)
  * THD
* Status historis untuk **parameter kategorikal**:

  * Overall Condition
  * Bearing Condition

---

### 🤖 3. Intelligent MCSA Chatbot

* Natural language query berbasis data MCSA
* Contoh pertanyaan:

  * "Unit mana yang Alarm saat ini?"
  * "Tampilkan semua motor dengan status High"
  * "Ringkasan kondisi unit A"

Chatbot difokuskan untuk **engineering insight**, bukan sekadar FAQ.

---

### 📄 4. Word Report Sync Engine

* Scan & parsing laporan MCSA (*.docx*)
* Sinkronisasi otomatis ke:

  ```
  mcsa_updated.csv
  ```
* Output kualitas data:

  * File gagal parsing
  * Missing parameter
  * Validasi & quality check

Mendukung standarisasi data lintas vendor & engineer.

---

### ⚙️ 5. Rotor Bar Analysis (Hybrid Logic)

* Re-calculation **Rotor Bar Severity**:

  * Level 1 – 4 (IEC-style)
* Status visual 3 warna:

  * Normal / High / Alarm
* Logika utama:

  * Sideband amplitude (dB)
* Parameter penguat (opsional):

  * Rotor Bar Health Index
  * Slip / Se

Dirancang agar **robust terhadap data tidak lengkap**.

---

### 📝 6. Performance Summary (Bahasa Indonesia)

* Ekstraksi pilihan bertanda **`X`** dari section *Performance Summary*
* Konversi otomatis ke:

  * Parameter teks terstruktur
  * Bahasa Indonesia

Cocok untuk laporan internal & manajemen non-teknis.

---

### 📊 7. Automated PowerPoint Generator

* Generate laporan **PowerPoint (.pptx)** otomatis
* Berisi:

  * Status unit
  * Grafik tren
  * Severity fault
  * Ringkasan kondisi

Menghemat waktu engineer untuk reporting.

---

### 📚 8. Training & Technical Knowledge Base

* Referensi teknis berbasis **JSON**
* Fitur:

  * Search
  * View langsung dari dashboard
* Digunakan sebagai:

  * Materi training
  * Standar internal analisis MCSA

---

## 🏗️ Design Philosophy

* ✅ Engineering-first (bukan sekadar visual)
* ✅ Reproducible & traceable analysis
* ✅ Vendor-agnostic
* ✅ Siap dikembangkan ke AI-based diagnostics

---

## 🎯 Target Users

* Reliability Engineer
* Electrical / Rotating Equipment Engineer
* Predictive Maintenance Team
* Asset Management

---

## 🚀 Future Roadmap

* Integrasi **Vibration & DGA Analysis**
* AI-based fault recommendation
* Confidence scoring berbasis load & data quality
* Offline deployment (air-gapped environment)

---

## 📌 Notes

Project ini dikembangkan sebagai **engineering framework**, bukan sekadar aplikasi dashboard.

> 💡 *Built by engineers, for engineers — turning MCSA data into decisions.*


## Cara Instalasi
1.  Pastikan Python sudah terinstall.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Cara Menjalankan
Klik ganda file `run.bat` atau jalankan perintah berikut di terminal:
```bash
streamlit run app.py
```

## Data & Folder
- `data/Report MCSA.xls`: Sumber data Excel (fallback jika `mcsa_updated.csv` belum ada).
- `data/mcsa_updated.csv`: Data gabungan (hasil sync Word + edit manual + merge).
- `data/Laporan/`: Struktur laporan Word yang direkomendasikan:
  - `data/Laporan/UNIT 1/380-400/...docx`
  - `data/Laporan/UNIT 2/6.3/...docx`

## Materi Training
- Simpan file materi di folder `Materi/` (se-level dengan `app.py`).
- Aplikasi mendukung 2 format JSON:

### Format v1 (Halaman)
Berupa list halaman atau object dengan `pages`.

```json
[
  {"page": 1, "content": "..."},
  {"page": 2, "content": "..."}
]
```

atau

```json
{
  "pages": [
    {"page": 1, "content": "..."}
  ]
}
```

### Format v2 (Artikel + Sections)
Berupa satu object artikel atau array artikel.

```json
{
  "id": "mcsa-bearing-fault",
  "title": "Analisis MCSA untuk Bearing Fault",
  "tags": ["MCSA", "Motor", "Bearing", "Fault"],
  "level": "intermediate",
  "source": "Internal Training",
  "language": "id",
  "sections": [
    {"id": "intro", "heading": "Pendahuluan", "content": "..."},
    {"id": "spectrum", "heading": "Karakteristik Spektrum", "content": "..."}
  ]
}
```

### Catatan Performa
- Untuk folder materi yang besar (mis. ratusan MB), pencarian dilakukan di dalam file yang dipilih agar tetap ringan.
- Jika file materi sangat besar, disarankan memecah per topik agar waktu buka lebih cepat.

## Sync Laporan Word
1. Jalankan aplikasi.
2. Buka halaman **Sync Laporan Word**.
3. Klik **Scan & Sync**.
4. Setelah selesai:
   - Data hasil parsing akan tersimpan/ter-update di `data/mcsa_updated.csv`.
   - Akan tampil daftar file gagal (jika ada), ringkasan missing parameter, dan quality check.

## Rotor Bar & Status
- Parameter rotor bar yang digunakan/ditambahkan:
  - `Upper Sideband`, `Lower Sideband`, `Rotorbar Health` (RB Hlt Index)
  - `Se Fund`, `Se Harm`, `Rotorbar Level %` (jika tersedia di report)
  - `Rotorbar Severity Level` (1–4)
  - `Rotorbar` (status 3 warna: Normal/Alarm/High)
- Saat load data, aplikasi melakukan **recalculation hanya untuk data latest per equipment** agar tetap ringan.

## Performance Summary (Bahasa Indonesia)
- Parameter yang disimpan berupa teks dan diawali prefix:
  - `Ringkasan Kinerja - Kesimpulan`
  - `Ringkasan Kinerja - Faktor Daya`
  - `Ringkasan Kinerja - Arus`
  - `Ringkasan Kinerja - Tegangan`
  - `Ringkasan Kinerja - Beban`
  - `Ringkasan Kinerja - Koneksi Fasa`
  - `Ringkasan Kinerja - Rotor`
  - `Ringkasan Kinerja - Stator`
  - `Ringkasan Kinerja - Air-gap Rotor/Stator`
  - `Ringkasan Kinerja - Distorsi Harmonik`
  - `Ringkasan Kinerja - Misalignment/Unbalance`
  - `Ringkasan Kinerja - Bearing`

## Verifikasi
Jalankan verifikasi cepat:
```bash
python verify_app.py
```

## Rekomendasi & Analisa Kondisi (MCSA/ESA, ATPOLL II)
- **Tujuan**: Menetapkan langkah tindak lanjut yang konsisten saat kondisi berubah (Normal/Alarm/High) berdasarkan praktik internasional MCSA/ESA.
- **Pengambilan Data (ATPOLL II)**:
  - Gunakan clamp arus dan tegangan sesuai spesifikasi alat, pastikan koneksi aman dan polaritas benar.
  - Ambil data pada beban yang cukup (≥40–60% nameplate) untuk analisa rotor bar yang lebih reliabel.
  - Rekam durasi yang memadai (≥30–60 detik) untuk stabilitas spektrum, hindari transien start/stop.
  - Catat metadata: Unit, Voltage, beban (perkiraan atau FLA/Load%), tanggal/jam, kondisi operasi (standby/normal).
- **Interpretasi Utama**:
  - Rotor Bar: evaluasi sideband sekitar fundamental (Upper/Lower SB). Ambang dasar: bagus jika < -54 dB; waspada -54 s/d -45 dB; kritis jika ≥ -45 dB. Perkuat keputusan dengan RB Hlt Index dan rasio Se harm vs Se fund bila tersedia.
  - Unbalance Tegangan/Arus: gunakan deviasi % fase; tegangan >1% waspada, >2% tinggi; arus >5% waspada, >10% tinggi.
  - THD Tegangan: rujukan IEEE 519; >5% waspada, >8% tinggi.
  - Bearing/Stator/Air-gap: gunakan Performance Summary untuk insight kualitatif dan tindak lanjut awal.
- **Langkah Tindak Lanjut**:
  - Normal: lanjutkan operasi; lakukan trending berkala (mingguan/bulanan) dan simpan rekaman.
  - Alarm: lakukan inspeksi terarah (cek koneksi fase, terminal, ground reference; survei vibrasi untuk misalignment/unbalance; review beban terhadap nameplate), tingkatkan frekuensi trending.
  - High: rencanakan tindakan segera (survei vibrasi menyeluruh, inspeksi koneksi dan terminal, evaluasi beban dan supply), siapkan rencana maintenance/penurunan beban untuk mitigasi risiko.
- **Best Practice**:
  - Bandingkan hasil saat beban rendah vs tinggi; hasil rotor bar pada beban rendah bisa tidak konklusif.
  - Konsistenkan titik ukur dan durasi rekaman untuk komparabilitas antar sesi.
  - Gunakan panel Ringkasan Performance untuk komunikasi cepat lintas tim (operasi/maintenance).

## Struktur File
- `data/`: Folder data (Report MCSA.xls, Laporan Word).
- `src/`: Source code (loader, generator, chatbot).
- `app.py`: Aplikasi utama (Streamlit).
