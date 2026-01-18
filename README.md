# MCSA Assistant

Aplikasi ini adalah Dashboard dan Chatbot untuk analisis Motor Current Signature Analysis (MCSA).

## Fitur
1. **Dashboard**: Monitoring kondisi equipment (Normal/Alarm/High/Standby) dengan filter Unit/Voltage.
2. **Trend**: Grafik tren parameter numerik dan status untuk parameter non-numerik (Kondisi/Bearing).
3. **Chatbot**: Tanya jawab seputar status equipment (mis. daftar Alarm/High).
4. **Sync Laporan Word**: Scan & sync hasil parsing laporan Word ke `mcsa_updated.csv` + ringkasan file gagal, missing parameter, dan quality check.
5. **Rotor Bar (Hybrid)**: Hitung ulang severity rotor bar (level 1–4) dan status 3 warna (Normal/Alarm/High) berbasis sideband dB sebagai utama, dengan penguat RB Hlt Index/Se bila ada.
6. **Performance Summary (Indonesia)**: Ambil pilihan bertanda `X` dari “Performance Summary” dan simpan sebagai parameter teks versi Indonesia.
7. **PPT Generator**: Membuat laporan PowerPoint secara otomatis.
8. **Materi Training**: Referensi teknis (JSON) yang dapat dicari dan dibuka dari dashboard.

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
