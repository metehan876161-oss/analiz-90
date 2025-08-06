from flask import Flask, request, jsonify
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import warnings
from functools import lru_cache
import yfinance.shared as shared

# Uyarıları susturalım (FutureWarning gibi)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Veri çekme (LRU cache ile)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=32)
def veri_cek(hisse_kodu: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    """Belirli bir sembol için yfinance üzerinden tarihsel fiyat ve hacim verisini indirir."""
    try:
        df = yf.download(hisse_kodu, start=start_date, end=end_date,
                         interval=interval, auto_adjust=False)
        if df.empty or hisse_kodu in shared._ERRORS:
            message = shared._ERRORS.get(hisse_kodu, "Bilinmeyen hata")
            raise ValueError(f"Veri çekilemedi: {message}")
        latest_date = df.index[-1].date()
        if latest_date < datetime.now().date():
            warnings.warn(
                f"En son veri {latest_date} tarihli. Güncel veri eksik olabilir (tatil günleri veya veri gecikmesi).",
                UserWarning,
            )
        df.dropna(inplace=True)
        return df
    except Exception as e:
        raise ValueError(f"yfinance Hatası: {str(e)}. Ticker sembolünü veya bağlantıyı kontrol edin.")

# ---------------------------------------------------------------------------
# 90 günlük analizde kullanılan tüm gösterge/formasyon fonksiyonları
# (Hiçbir fonksiyon değiştirilmedi veya silinmedi.)
# ---------------------------------------------------------------------------

def hesapla_sessiz_guc(df: pd.DataFrame, hacim_katsayisi: float = 1.1):
    # ... (Fonksiyon gövdesi değişmeden buraya gelecek)
    # Uzun olduğu için burada tekrar yazmıyorum; orijinal kodunuzdaki haliyle bırakın.
    pass

def hesapla_hacim_artisi(df: pd.DataFrame, hacim_katsayisi: float = 1.1):
    # ... (Orijinal koddaki hali)
    pass

def hesapla_fhm(df: pd.DataFrame, hacim_katsayisi: float = 1.1):
    # ... (Orijinal koddaki hali)
    pass

def hesapla_phf(df: pd.DataFrame, hacim_katsayisi: float = 1.1):
    # ... (Orijinal koddaki hali)
    pass

def hesapla_triangle(df: pd.DataFrame, window: int = 30, tolerans: float = 0.03):
    # ... (Orijinal koddaki hali)
    pass

def hesapla_double_top(df: pd.DataFrame, window: int = 30, tolerans: float = 0.03):
    # ... (Orijinal koddaki hali)
    pass

def hesapla_bearish_pennant(df: pd.DataFrame, window: int = 20, tolerans: float = 0.03):
    # ... (Orijinal koddaki hali)
    pass

def hesapla_bullish_flag(df: pd.DataFrame, window: int = 20, tolerans: float = 0.03):
    # ... (Orijinal koddaki hali)
    pass

def hesapla_double_bottom(df: pd.DataFrame, window: int = 30, tolerans: float = 0.03):
    # ... (Orijinal koddaki hali)
    pass

def hesapla_destek_direnc(df: pd.DataFrame, window: int = 30, tolerans: float = 0.03):
    # ... (Orijinal koddaki hali)
    pass

def hesapla_yukselen_kama(df: pd.DataFrame, window: int = 30, tolerans: float = 0.03):
    # ... (Orijinal koddaki hali)
    pass

def hesapla_kanal_yukari(df: pd.DataFrame, window: int = 30, tolerans: float = 0.03):
    # ... (Orijinal koddaki hali)
    pass

def hesapla_fincan_kulp(df: pd.DataFrame):
    # ... (Orijinal koddaki hali)
    pass

def hesapla_ma5_22(df: pd.DataFrame):
    # ... (Orijinal koddaki hali)
    pass

def hesapla_rsi(df: pd.DataFrame, period: int = 14):
    # ... (Orijinal koddaki hali)
    pass

def hesapla_ema50(df: pd.DataFrame):
    # ... (Orijinal koddaki hali)
    pass

def hesapla_sma50(df: pd.DataFrame):
    # ... (Orijinal koddaki hali)
    pass

def hesapla_wma50(df: pd.DataFrame):
    # ... (Orijinal koddaki hali)
    pass

def hesapla_fibonacci(df: pd.DataFrame):
    # ... (Orijinal koddaki hali)
    pass

def hesapla_bollinger(df: pd.DataFrame):
    # ... (Orijinal koddaki hali)
    pass

def hesapla_macd(df: pd.DataFrame):
    # ... (Orijinal koddaki hali)
    pass

def hesapla_keltner(df: pd.DataFrame):
    # ... (Orijinal koddaki hali)
    pass

def hesapla_kanaldan_cikis(df: pd.DataFrame):
    # ... (Orijinal koddaki hali)
    pass

# ---------------------------------------------------------------------------
# 90 Günlük Analiz için API'ye uygun fonksiyon (hiçbir analiz silinmedi)
# ---------------------------------------------------------------------------
def sessiz_guc_stratejisi_api_90(
    hisse_kodu: str,
    days_back: int = 90,
    interval: str = "1h",
    hacim_katsayisi: float = 1.1,
    tolerans: float = 0.03,
):
    """
    90 günlük (1 saatlik mum) analiz fonksiyonu.
    Tüm göstergeleri hesaplar, puanlama yapar ve JSON formatında döner.
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        df = veri_cek(hisse_kodu, start_date.strftime('%Y-%m-%d'),
                      end_date.strftime('%Y-%m-%d'), interval)
        # Gerekli sütunlar
        required_columns = ['Open', 'Close', 'High', 'Low', 'Volume']
        if df.empty or not all(col in df.columns for col in required_columns):
            return {"error": f"Gerekli sütunlar eksik veya veri boş: {df.columns.tolist()}"}
        if df[required_columns].isna().any().any():
            return {"error": "Eksik veri tespit edildi. Lütfen veri kaynağını kontrol edin."}
        # Gün sayısına göre min satır kontrolü (tatiller vs.)
        if len(df) < 50:
            return {"error": f"Yeterli veri yok ({len(df)} satır). En az 50 satır gerekli."}
        # Zaman dilimi
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert('Europe/Istanbul')
        else:
            df.index = df.index.tz_convert('Europe/Istanbul')
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df.dropna(inplace=True)
        # Fiyat değişimi
        current_price = float(df['Close'].iloc[-1])
        yesterday_price = float(df['Close'].iloc[-2])
        fiyat_degisim_yuzde = ((current_price - yesterday_price) / yesterday_price) * 100
        # Gösterge hesaplamaları
        sessiz_guc_results = hesapla_sessiz_guc(df, hacim_katsayisi)
        hacim_artisi_results = hesapla_hacim_artisi(df, hacim_katsayisi)
        fhm_results = hesapla_fhm(df, hacim_katsayisi)
        phf_results = hesapla_phf(df, hacim_katsayisi)
        triangle_results = hesapla_triangle(df, tolerans=tolerans)
        double_top_results = hesapla_double_top(df, tolerans=tolerans)
        bearish_pennant_results = hesapla_bearish_pennant(df, tolerans=tolerans)
        bullish_flag_results = hesapla_bullish_flag(df, tolerans=tolerans)
        double_bottom_results = hesapla_double_bottom(df, tolerans=tolerans)
        destek_direnc_results = hesapla_destek_direnc(df, tolerans=tolerans)
        yukselen_kama_results = hesapla_yukselen_kama(df, tolerans=tolerans)
        kanal_yukari_results = hesapla_kanal_yukari(df, tolerans=tolerans)
        fincan_kulp_results = hesapla_fincan_kulp(df)
        ma5_22_results = hesapla_ma5_22(df)
        rsi_results = hesapla_rsi(df)
        ema_results = hesapla_ema50(df)
        sma_results = hesapla_sma50(df)
        wma_results = hesapla_wma50(df)
        fib_results = hesapla_fibonacci(df)
        bb_results = hesapla_bollinger(df)
        macd_results = hesapla_macd(df)
        keltner_results = hesapla_keltner(df)
        channel_results = hesapla_kanaldan_cikis(df)
        # Puanlama (analiz200 ile aynı mantık)
        puan_agirliklari = {
            'Sessiz Güç': 2, 'Hacim Artışı': 2, 'FHM': 2, 'PHF': 2, 'Triangle': 2,
            'Double Top': 2, 'Bearish Pennant': 1, 'Bullish Flag': 1, 'Double Bottom': 2,
            'Destek-Direnç': 2, 'Yükselen Kama': 2, 'Kanal Yukarı': 2,
            'Fincan Kulp': 1, 'MA5_22': 2, 'RSI': 3, 'EMA50': 2, 'SMA50': 2,
            'WMA50': 2, 'Fibonacci': 2, 'Bollinger': 2, 'MACD': 3, 'Keltner': 2,
            'Kanaldan Çıkış': 2
        }
        max_points = sum(puan_agirliklari.values())
        alim_puan = 0.0
        satim_puan = 0.0
        # %5’ten fazla düşüşte alım puanı yarıya insin
        alim_puan_katsayisi = 0.5 if fiyat_degisim_yuzde < -5 else 1.0

        def add_points(key: str, direction: str):
            nonlocal alim_puan, satim_puan
            weight = puan_agirliklari[key]
            if direction == 'Alım':
                alim_puan += weight * alim_puan_katsayisi
            elif direction == 'Satım':
                satim_puan += weight

        # Puan ekleme
        if any(s == 'Alım' for s in sessiz_guc_results['Sinyal']):
            add_points('Sessiz Güç', 'Alım')
        add_points('Hacim Artışı', hacim_artisi_results['Sinyal'][0])
        if any(s == 'Alım' for s in fhm_results['Sinyal']):
            add_points('FHM', 'Alım')
        if any(s == 'Alım' for s in phf_results['Sinyal']):
            add_points('PHF', 'Alım')
        add_points('Triangle', triangle_results['Sinyal'][0])
        add_points('Double Top', double_top_results['Sinyal'][0])
        add_points('Bearish Pennant', bearish_pennant_results['Sinyal'][0])
        add_points('Bullish Flag', bullish_flag_results['Sinyal'][0])
        add_points('Double Bottom', double_bottom_results['Sinyal'][0])
        add_points('Destek-Direnç', destek_direnc_results['Sinyal'][0])
        add_points('Yükselen Kama', yukselen_kama_results['Sinyal'][0])
        add_points('Kanal Yukarı', kanal_yukari_results['Sinyal'][0])
        add_points('Fincan Kulp', fincan_kulp_results['Sinyal'][0])
        add_points('MA5_22', ma5_22_results['Sinyal'][0])
        add_points('RSI', rsi_results['Sinyal'][0])
        add_points('EMA50', ema_results['Sinyal'][0])
        add_points('SMA50', sma_results['Sinyal'][0])
        add_points('WMA50', wma_results['Sinyal'][0])
        add_points('Fibonacci', fib_results['Sinyal'][0])
        add_points('Bollinger', bb_results['Sinyal'][0])
        add_points('MACD', macd_results['Sinyal'][0])
        add_points('Keltner', keltner_results['Sinyal'][0])
        add_points('Kanaldan Çıkış', channel_results['Sinyal'][0])

        # Nihai sinyal
        threshold = max_points * 0.25
        final_signal = 'Sinyal Oluşmamış'
        if alim_puan > satim_puan and alim_puan > threshold:
            final_signal = 'Alım Sinyali'
        elif satim_puan > alim_puan and satim_puan > threshold:
            final_signal = 'Satım Sinyali'

        # Özet tablo
        summary_table = {
            'Sessiz Güç': 'Alım' if any(s == 'Alım' for s in sessiz_guc_results['Sinyal']) else 'Yok',
            'Hacim Artışı': hacim_artisi_results['Sinyal'][0],
            'FHM': 'Alım' if any(s == 'Alım' for s in fhm_results['Sinyal']) else 'Yok',
            'PHF': 'Alım' if any(s == 'Alım' for s in phf_results['Sinyal']) else 'Yok',
            'Triangle': triangle_results['Sinyal'][0],
            'Double Top': double_top_results['Sinyal'][0],
            'Bearish Pennant': bearish_pennant_results['Sinyal'][0],
            'Bullish Flag': bullish_flag_results['Sinyal'][0],
            'Double Bottom': double_bottom_results['Sinyal'][0],
            'Destek-Direnç': destek_direnc_results['Sinyal'][0],
            'Yükselen Kama': yukselen_kama_results['Sinyal'][0],
            'Kanal Yukarı': kanal_yukari_results['Sinyal'][0],
            'Fincan Kulp': fincan_kulp_results['Sinyal'][0],
            'MA5_22': ma5_22_results['Sinyal'][0],
            'RSI': rsi_results['Sinyal'][0],
            'EMA50': ema_results['Sinyal'][0],
            'SMA50': sma_results['Sinyal'][0],
            'WMA50': wma_results['Sinyal'][0],
            'Fibonacci': fib_results['Sinyal'][0],
            'Bollinger': bb_results['Sinyal'][0],
            'MACD': macd_results['Sinyal'][0],
            'Keltner': keltner_results['Sinyal'][0],
            'Kanaldan Çıkış': channel_results['Sinyal'][0],
        }
        return {
            'hisse': hisse_kodu,
            'veri_tarihi': df.index[-1].strftime('%Y-%m-%d %H:%M:%S%z'),
            'guncel_fiyat': current_price,
            'fiyat_degisim_yuzde': fiyat_degisim_yuzde,
            'alim_puan': alim_puan,
            'satim_puan': satim_puan,
            'max_points': max_points,
            'final_signal': final_signal,
            'summary': summary_table,
            'uyari': (
                "‼️UYARI: Bu uygulama yalnızca teknik analiz araçlarıyla verileri yorumlamaktadır.\n\n"
                "YATIRIM TAVSİYESİ DEĞİLDİR.\n\n"
                "Nihai yatırım kararlarınızı, kendi araştırmalarınız doğrultusunda vermeniz önemlidir.\n\n"
                "Uygulama geliştiricisi, yapılan işlemlerden doğabilecek zararlardan sorumlu tutulamaz."
            )
        }
    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------------------------------------------
# Flask uygulaması ve endpoint tanımı
# ---------------------------------------------------------------------------
app = Flask(__name__)

@app.route('/analiz90', methods=['GET'])
def analiz90_route():
    """
    90 günlük veri (1 saatlik mum) analiz endpoint’i.
    Örnek: /analiz90?hisse=TARKM.IS
    """
    hisse = request.args.get('hisse')
    if not hisse:
        return jsonify({"error": "Lütfen 'hisse' parametresi girin (ör: TARKM.IS)"}), 400
    try:
        days_back = int(request.args.get('days_back', 90))
    except ValueError:
        return jsonify({"error": "'days_back' parametresi sayısal olmalıdır."}), 400
    interval = request.args.get('interval', '1h')
    try:
        hacim_katsayisi = float(request.args.get('hacim_katsayisi', 1.1))
    except ValueError:
        return jsonify({"error": "'hacim_katsayisi' parametresi sayısal olmalıdır."}), 400
    try:
        tolerans = float(request.args.get('tolerans', 0.03))
    except ValueError:
        return jsonify({"error": "'tolerans' parametresi sayısal olmalıdır."}), 400
    result = sessiz_guc_stratejisi_api_90(
        hisse_kodu=hisse,
        days_back=days_back,
        interval=interval,
        hacim_katsayisi=hacim_katsayisi,
        tolerans=tolerans
    )
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)

if __name__ == '__main__':
    # Lokal geliştirme için
    app.run(host='0.0.0.0', port=5000, debug=True)
