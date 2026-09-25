import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#718096"))
        page_text = f"Halaman {self._pageNumber} dari {page_count}"
        self.drawRightString(8.5 * inch - 0.75 * inch, 0.45 * inch, page_text)
        self.drawString(0.75 * inch, 0.45 * inch, "TK 1 Analisis Numerik - Kontrak Kerja & Interface Nomor 2 (SETAR)")
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(0.75 * inch, 0.6 * inch, 8.5 * inch - 0.75 * inch, 0.6 * inch)
        self.restoreState()

def generate_pdf(output_path):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=3
    )

    code_style = ParagraphStyle(
        'Code_Custom',
        fontName='Courier',
        fontSize=7.8,
        leading=10.5,
        textColor=colors.HexColor("#1A202C")
    )

    story = []

    # Title & Header
    story.append(Paragraph("KONTRAK KERJA DAN SPESIFIKASI ANTARMUKA", title_style))
    story.append(Paragraph("<b>Tugas Kelompok 1 Analisis Numerik Gasal 2026/2027 — Soal Nomor 2 (SETAR)</b><br/>"
                           "<i>Metode Khusus Kelompok Genap: Givens Rotations & Persamaan Normal</i>", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=10))

    # Section 1: Overview & Parallel Principle
    story.append(Paragraph("1. Prinsip Kolaborasi Paralel", h1_style))
    story.append(Paragraph(
        "Untuk memaksimalkan efisiensi dan memastikan pengerjaan dapat berjalan <b>100% paralel tanpa saling menunggu</b>, "
        "tugas Nomor 2 dibagi menjadi dua peran teknis yang independen namun komplementer. "
        "Person A (Solver & Teori) langsung menguji kodenya menggunakan <b>matriks toy di Halaman 10 PDF</b>, "
        "sementara Person B (Data & Finansial) memproses dataset <code>stock_train.csv</code> dan <code>stock_test.csv</code>. "
        "Keduanya terikat oleh antarmuka fungsi (interface) di bawah ini.",
        body_style
    ))

    # Table of Task Breakdown
    table_data = [
        [
            Paragraph("<b>Aspek</b>", body_style),
            Paragraph("<b>Person A (Kamu)<br/>Numerical & Algorithm Engineer</b>", body_style),
            Paragraph("<b>Person B (Temanmu)<br/>Data & Financial Modeling</b>", body_style)
        ],
        [
            Paragraph("<b>Sub-Poin Tugas</b>", body_style),
            Paragraph("Poin <b>iii, iv, v</b> + Pseudocode", body_style),
            Paragraph("Poin <b>i, ii, vi, vii</b> + README.md", body_style)
        ],
        [
            Paragraph("<b>Fokus Coding</b>", body_style),
            Paragraph("• Solver Persamaan Normal dari nol<br/>"
                      "• Solver QR Givens Rotations dari nol<br/>"
                      "• Back-substitution dari nol<br/>"
                      "• Verifikasi eliminasi awal (matriks <i>G</i><sub>k</sub>)", body_style),
            Paragraph("• Preprocessing CSV & hitung return<br/>"
                      "• Konstruksi matriks desain <i>A</i> dan <i>b</i><br/>"
                      "• Evaluasi out-of-sample ke test set<br/>"
                      "• Perhitungan metrik RMSE", body_style)
        ],
        [
            Paragraph("<b>Fokus Analisis & Laporan</b>", body_style),
            Paragraph("• Analisis <i>condition number</i> κ<sub>2</sub>(<i>A</i><sup>T</sup><i>A</i>)<br/>"
                      "• Perbandingan FLOPs & konsumsi memori<br/>"
                      "• Teori dampak outlier ekstrem (<i>L</i><sub>2</sub> norm)<br/>"
                      "• Pseudocode kedua algoritma", body_style),
            Paragraph("• Formulasi matematis & dimensi sistem<br/>"
                      "• Investigasi isu skala/kondisi matriks <i>A</i><br/>"
                      "• Visualisasi grafik overlay time series<br/>"
                      "• Interpretasi finansial koefisien (momentum vs mean-reverting)", body_style)
        ],
        [
            Paragraph("<b>Berkas Utama</b>", body_style),
            Paragraph("<code>src/solvers.py</code><br/><code>test_toy.py</code>", body_style),
            Paragraph("<code>src/data_pipeline.py</code><br/><code>README.md</code>", body_style)
        ]
    ]

    t = Table(table_data, colWidths=[1.3 * inch, 2.85 * inch, 2.85 * inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EBF8FF")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))

    # Section 2: Contract 1
    story.append(Paragraph("2. Kontrak 1: Person B → Person A (Data Pipeline)", h1_style))
    story.append(Paragraph(
        "Person B mengimplementasikan modul <code>src/data_pipeline.py</code>. "
        "Matriks <i>A</i> harus memiliki <b>6 kolom dengan urutan baku</b> sesuai Persamaan (5) pada dokumen tugas, "
        "agar koefisien solusi <i>x</i> selaras dengan parameter model:",
        body_style
    ))

    code_p1 = """# File: src/data_pipeline.py
import numpy as np

def build_dataset(csv_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    \"\"\"
    Membaca CSV saham, menghitung return, dan menyusun matriks overdetermined A & target b.
    
    Output:
    - A: np.ndarray (shape=(M, 6), dtype=float64)
         Kolom 0: I_t               (1 jika R_{t-1} >= 0, else 0)
         Kolom 1: I_t * R_{t-1}     (Lag-1 Bullish)
         Kolom 2: I_t * R_{t-2}     (Lag-2 Bullish)
         Kolom 3: 1 - I_t           (1 jika R_{t-1} < 0, else 0)
         Kolom 4: (1 - I_t)*R_{t-1} (Lag-1 Bearish)
         Kolom 5: (1 - I_t)*R_{t-2} (Lag-2 Bearish)
    - b: np.ndarray (shape=(M,), dtype=float64) -> Target return hari ke-t (R_t)
    - actual_returns: np.ndarray (shape=(Total,), dtype=float64) -> deret return utuh untuk plot
    
    Ketentuan:
    - Return R_t = (P_t - P_{t-1}) / P_{t-1}
    - Karena lag p=2, baris dimulai dari observasi t=3 (indeks 2 pada 0-indexed).
    - Jika ada N data harga, jumlah baris M = N - 3.
    \"\"\"
"""
    story.append(Preformatted(code_p1, code_style, maxLineLength=85))
    story.append(Spacer(1, 6))

    # Section 3: Contract 2
    story.append(Paragraph("3. Kontrak 2: Person A → Person B (Numerical Solvers)", h1_style))
    story.append(Paragraph(
        "Person A mengimplementasikan modul <code>src/solvers.py</code> tanpa menggunakan pustaka penyelesaian SPL/LSP bawaan. "
        "Output berupa dictionary terstruktur yang langsung dapat dikonsumsi oleh script evaluasi Person B:",
        body_style
    ))

    code_p2 = """# File: src/solvers.py
import numpy as np

def solve_normal_equations(A: np.ndarray, b: np.ndarray) -> dict:
    \"\"\"
    Menyelesaikan Least Squares Problem via Persamaan Normal: A^T A x = A^T b.
    Returns:
        {
            'x': np.ndarray (shape=(6,)),            # Solusi koefisien
            'residual_norm': float,                   # ||A*x - b||_2
            'cond_A': float,                          # kappa_2(A)
            'cond_ATA': float                         # kappa_2(A^T A)
        }
    \"\"\"

def solve_givens_qr(A: np.ndarray, b: np.ndarray) -> dict:
    \"\"\"
    Menyelesaikan Least Squares Problem via Givens Rotations QR + Back Substitution.
    Returns:
        {
            'x': np.ndarray (shape=(6,)),            # Solusi kuadrat terkecil x_LS
            'residual_norm': float,                   # ||A*x_LS - b||_2
            'R': np.ndarray (shape=(6, 6)),          # Matriks segitiga atas R
            'G1': np.ndarray                         # Matriks rotasi pertama pengeliminasi subdiagonal
        }
    \"\"\"

def predict(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    \"\"\"Menghitung R_hat = A @ x untuk data train/test.\"\"\"
    return A @ x

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    \"\"\"Menghitung RMSE = sqrt( mean( (y_true - y_pred)**2 ) ).\"\"\"
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
"""
    story.append(Preformatted(code_p2, code_style, maxLineLength=85))
    story.append(Spacer(1, 6))

    # Section 4: Integration
    story.append(Paragraph("4. Integrasi Akhir (Digunakan Bersama di <code>main.py</code>)", h1_style))
    code_main = """# File: main.py
from src.data_pipeline import build_dataset
from src.solvers import solve_normal_equations, solve_givens_qr, predict, calculate_rmse

# 1. Pipeline Data (Hasil Person B)
A_train, b_train, ret_train = build_dataset("data/stock_train.csv")
A_test, b_test, ret_test    = build_dataset("data/stock_test.csv")

# 2. Eksekusi Solver (Hasil Person A)
res_normal = solve_normal_equations(A_train, b_train)
res_givens = solve_givens_qr(A_train, b_train)

# 3. Evaluasi Out-of-Sample (Person B)
x_opt = res_givens['x']
rmse_train = calculate_rmse(b_train, predict(A_train, x_opt))
rmse_test  = calculate_rmse(b_test, predict(A_test, x_opt))
print(f"Evaluasi Model: RMSE Train = {rmse_train:.6f} | RMSE Test = {rmse_test:.6f}")
"""
    story.append(Preformatted(code_main, code_style, maxLineLength=85))
    story.append(Spacer(1, 6))

    # Section 5: Roadmap & Deliverables
    story.append(Paragraph("5. Roadmap Eksekusi & Check-List Laporan", h1_style))
    story.append(Paragraph("• <b>Fase 1 (Mandiri)</b>: Person A buat & tes solver di <code>test_toy.py</code> pakai contoh Hal. 10. Person B buat <code>data_pipeline.py</code> dan cek distribusi return.", bullet_style))
    story.append(Paragraph("• <b>Fase 2 (Integrasi)</b>: Gabungkan di <code>main.py</code>, hitung RMSE, catat waktu & memori, generate plot overlay kontinu.", bullet_style))
    story.append(Paragraph("• <b>Fase 3 (Laporan)</b>: Pastikan memuat <i>Pseudocode</i> (Person A), <i>Tabel Performa & Kondisi Matriks</i> (Person A), <i>Grafik Overlay Train-Test</i> (Person B), <i>Interpretasi Finansial</i> (Person B), dan <i>README.md</i>.", bullet_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF berhasil dibuat di: {output_path}")

if __name__ == '__main__':
    target = os.path.join(r"U:\Universitas Indonesia - S1\Semester 5\anum\tk\1", "Kontrak_Kerja_Nomor_2_SETAR.pdf")
    generate_pdf(target)
