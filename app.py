import mplfinance as mpf
import pandas as pd
from PIL import Image
from datetime import datetime, timedelta
from ultralytics import YOLO
import os
import glob
import json
from pathlib import Path
import configparser
import logging
import random
import cv2
import numpy as np

# Replace the relative path to your weight file
model_path = 'weights/custom_yolov8.pt'

def load_config(config_file='config.txt'):
    """Lädt Konfiguration aus TXT-Datei mit configparser"""
    logger = logging.getLogger(__name__)
    
    config_parser = configparser.ConfigParser(allow_no_value=True, inline_comment_prefixes=('#', ';'))
    config = {}
    
    try:
        # ConfigParser erwartet Sections, also fügen wir eine DEFAULT Section hinzu
        with open(config_file, 'r', encoding='utf-8') as f:
            config_content = '[DEFAULT]\n' + f.read()
        
        config_parser.read_string(config_content)
        
        # Alle Werte aus der DEFAULT Section extrahieren
        for key, value in config_parser['DEFAULT'].items():
            config[key.upper()] = value
        
        logger.info(f"Konfiguration erfolgreich geladen aus {config_file}")
        return config
        
    except FileNotFoundError as e:
        logger.error(f"Konfigurationsdatei {config_file} nicht gefunden")
        raise FileNotFoundError(f"Konfigurationsdatei {config_file} nicht gefunden") from e
    except configparser.Error as e:
        logger.error(f"Fehler beim Parsen der Konfigurationsdatei: {e}")
        raise configparser.Error(f"Fehler beim Parsen der Konfigurationsdatei: {e}") from e
    except Exception as e:
        logger.error(f"Unerwarteter Fehler beim Laden der Konfiguration: {e}")
        raise

# Configuration laden
config = load_config()
PARQUET_DATA_PATH = Path(config.get("PARQUET_DATA_PATH", "data/parquet")).expanduser()
METADATA_DB_PATH = Path(config.get("METADATA_DB_PATH", "data/metadata.db")).expanduser()
TEMP_CHARTS_PATH = Path(config.get("TEMP_CHARTS_PATH", "temp/charts")).expanduser()
RESULTS_PATH = Path(config.get("RESULTS_PATH", "results")).expanduser()
JSON_RESULTS_PATH = Path(config.get("JSON_RESULTS_PATH", RESULTS_PATH / "json")).expanduser()
INPUT_FILE = Path(config.get("INPUT_FILE", "ticker_list.txt")).expanduser()

# AI Detection Configuration
CONFIDENCE_THRESHOLD = 0.24  # Minimum confidence für AI-Erkennungen
SAVE_ONLY_WITH_DETECTIONS = True  # True = nur speichern wenn Erkennungen vorhanden, False = immer speichern

def read_ticker_list(input_file):
    """Liest Ticker-Liste aus TXT-Datei"""
    try:
        with open(input_file, 'r') as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
        print(f"Ticker-Liste geladen: {len(tickers)} Ticker gefunden")
        return tickers
    except FileNotFoundError:
        print(f"Fehler: Input-Datei {input_file} nicht gefunden")
        return []
    except Exception as e:
        print(f"Fehler beim Lesen der Ticker-Liste: {e}")
        return []

def extract_relative_patterns(data):
    """Extrahiert relative Bewegungsmuster aus historischen OHLCV-Daten"""
    patterns = []
    
    for i in range(1, len(data)):
        prev_close = data.iloc[i-1]['Close']
        current = data.iloc[i]
        
        pattern = {
            'open_change': (current['Open'] - prev_close) / prev_close,
            'high_rel': (current['High'] - current['Open']) / current['Open'],
            'low_rel': (current['Low'] - current['Open']) / current['Open'],
            'close_rel': (current['Close'] - current['Open']) / current['Open'],
            'volume_change': current['Volume'] / data.iloc[i-1]['Volume'] if data.iloc[i-1]['Volume'] > 0 else 1.0
        }
        patterns.append(pattern)
    
    return patterns

def generate_monte_carlo_extension(data, patterns, days=30):
    """Generiert Monte Carlo Verlängerung basierend auf historischen Mustern"""
    extended_data = data.copy()
    
    for day in range(1, days + 1):
        # Zufälliges Muster auswählen
        pattern = random.choice(patterns)
        
        # Neue Werte basierend auf dem Muster berechnen
        prev_close = extended_data.iloc[-1]['Close']
        
        new_open = prev_close * (1 + pattern['open_change'])
        new_high = new_open * (1 + pattern['high_rel'])
        new_low = new_open * (1 + pattern['low_rel'])
        new_close = new_open * (1 + pattern['close_rel'])
        new_volume = extended_data.iloc[-1]['Volume'] * pattern['volume_change']
        
        # Sicherstellen dass High >= max(Open, Close) und Low <= min(Open, Close)
        new_high = max(new_high, new_open, new_close)
        new_low = min(new_low, new_open, new_close)
        
        # Neues Datum (nächster Handelstag)
        last_date = extended_data.index[-1]
        new_date = last_date + timedelta(days=1)
        
        # Neue Zeile hinzufügen
        new_row = pd.Series({
            'Open': new_open,
            'High': new_high,
            'Low': new_low,
            'Close': new_close,
            'Adj Close': new_close,  # Vereinfachung
            'Volume': int(new_volume)
        }, name=new_date)
        
        extended_data = pd.concat([extended_data, new_row.to_frame().T])
    
    return extended_data

def load_parquet_data(ticker):
    """Lädt Parquet-Daten für einen Ticker basierend auf der dokumentierten Struktur"""
    try:
        # Alle Parquet-Dateien in allen Jahr-Partitionen laden
        parquet_files = list(PARQUET_DATA_PATH.glob("year=*/*.parquet"))
        
        if not parquet_files:
            print(f"Keine Parquet-Dateien gefunden in {PARQUET_DATA_PATH}")
            return None
        
        # Alle Parquet-Dateien laden und nach Ticker filtern
        dataframes = []
        for parquet_file in parquet_files:
            try:
                df = pd.read_parquet(parquet_file)
                # Nach Ticker filtern
                ticker_data = df[df['ticker'] == ticker.upper()]
                if not ticker_data.empty:
                    dataframes.append(ticker_data)
            except Exception as e:
                print(f"Fehler beim Laden von {parquet_file}: {e}")
                continue
        
        if not dataframes:
            print(f"Keine Daten für Ticker {ticker} gefunden")
            return None
        
        # Alle Daten kombinieren und sortieren
        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df['date'] = pd.to_datetime(combined_df['date'])
        combined_df = combined_df.sort_values('date')
        
        # Nur die letzten 180 Datenpunkte behalten
        combined_df = combined_df.tail(180)
        
        # Index auf Datum setzen und Spalten für mplfinance anpassen
        combined_df.set_index('date', inplace=True)
        
        # Spalten umbenennen für mplfinance Kompatibilität
        column_mapping = {
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'adjusted_close': 'Adj Close',
            'volume': 'Volume'
        }
        combined_df.rename(columns=column_mapping, inplace=True)
        
        print(f"Daten für {ticker} geladen: {len(combined_df)} Datensätze von {combined_df.index.min()} bis {combined_df.index.max()}")
        return combined_df
        
    except Exception as e:
        print(f"Fehler beim Laden der Parquet-Daten für {ticker}: {e}")
        return None

def generate_chart(ticker, data, chunk_size=180, figsize=(18, 6.5), dpi=100, use_monte_carlo=False, mc_days=30):
    """Generiert Chart aus Parquet-Daten und speichert als temporäres Bild"""
    try:
        if data is None or data.empty:
            print(f"Keine Daten für {ticker} verfügbar")
            return None
        
        # Monte Carlo Verlängerung wenn aktiviert
        if use_monte_carlo:
            # Muster aus den Daten extrahieren
            patterns = extract_relative_patterns(data)
            
            if len(patterns) < 10:
                print(f"Zu wenige Daten für Monte Carlo Muster für {ticker}: {len(patterns)} Muster")
                return None
            
            # Monte Carlo Verlängerung generieren
            extended_data = generate_monte_carlo_extension(data, patterns, mc_days)
            
            # Nur die letzten chunk_size Kerzen für den Chart verwenden
            data_subset = extended_data.tail(chunk_size)
            
            # Markierung wo echte Daten enden und MC beginnt
            mc_start_index = chunk_size - mc_days
            chart_title = f"{ticker} Latest {len(data_subset)} Candles (MC: {mc_days} days)"
            
        else:
            # Letzte 180 Tage auswählen
            data_subset = data.iloc[-chunk_size:]
            chart_title = f"{ticker} Latest {len(data_subset)} Candles"
        
        if len(data_subset) < 10:  # Mindestens 10 Datenpunkte für sinnvollen Chart
            print(f"Zu wenige Daten für {ticker}: {len(data_subset)} Datenpunkte")
            return None
        
        # Chart erstellen
        fig, ax = mpf.plot(data_subset, 
                          type="candle", 
                          style="yahoo",
                          title=chart_title,
                          axisoff=True,
                          ylabel="",
                          ylabel_lower="",
                          volume=False,
                          figsize=figsize,
                          returnfig=True)
        
        # Monte Carlo Trennlinie hinzufügen wenn aktiviert
        if use_monte_carlo:
            ax[0].axvline(x=mc_start_index, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        # Temporäres Bild speichern
        suffix = "_mc_chart.png" if use_monte_carlo else "_chart.png"
        temp_image_path = TEMP_CHARTS_PATH / f"{ticker}{suffix}"
        fig.savefig(temp_image_path, format='png', dpi=dpi, bbox_inches='tight')
        
        print(f"Chart für {ticker} erstellt: {temp_image_path}")
        return temp_image_path
        
    except Exception as e:
        print(f"Fehler beim Erstellen des Charts für {ticker}: {e}")
        return None

def perform_detection(image_path, model, confidence=CONFIDENCE_THRESHOLD):
    """Führt Objekterkennung auf einem Bild durch"""
    try:
        # Bild laden
        image = Image.open(image_path)
        
        # Objekterkennung durchführen
        results = model.predict(image, conf=confidence)
        
        # Ergebnisse verarbeiten
        detection_results = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    conf_value = float(box.conf.tolist()[0]) if len(box.conf) > 0 else 0.0
                    # Nur Erkennungen mit Confidence > Threshold behalten
                    if conf_value > CONFIDENCE_THRESHOLD:
                        detection_results.append({
                            'bbox': box.xywh.tolist()[0] if len(box.xywh) > 0 else [],
                            'confidence': conf_value,
                            'class': int(box.cls.tolist()[0]) if len(box.cls) > 0 else -1
                        })
        
        return detection_results, results
        
    except Exception as e:
        print(f"Fehler bei der Objekterkennung für {image_path}: {e}")
        return [], None

def save_detection_results(ticker, detection_results, results, output_folder, is_monte_carlo=False, mc_start_index=None):
    """Speichert AI-Erkennungsergebnisse"""
    try:
        # Erkennungsergebnisse als JSON speichern
        results_file = JSON_RESULTS_PATH / f"{ticker}_detection_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'ticker': ticker,
                'timestamp': datetime.now().isoformat(),
                'detections': detection_results,
                'total_detections': len(detection_results),
                'monte_carlo': is_monte_carlo,
                'mc_start_index': mc_start_index if is_monte_carlo else None
            }, f, indent=2)
        
        # Annotiertes Bild speichern (falls verfügbar)
        if results and len(results) > 0:
            annotated_image = results[0].plot()
            
            # Monte Carlo Trennlinie hinzufügen wenn aktiviert
            if is_monte_carlo and mc_start_index is not None:
                # Sicherstellen dass das Bild im richtigen Format für OpenCV ist
                if not isinstance(annotated_image, np.ndarray):
                    annotated_image = np.array(annotated_image)
                
                # Sicherstellen dass es ein kontinuierliches Array ist
                annotated_image = np.ascontiguousarray(annotated_image)
                
                # Bildabmessungen
                height, width = annotated_image.shape[:2]
                
                # X-Position der Trennlinie berechnen (basierend auf Chart-Breite)
                # Annahme: Chart nimmt etwa 90% der Bildbreite ein, startet bei 5%
                chart_start_x = int(width * 0.05)
                chart_width = int(width * 0.90)
                
                # Position der Trennlinie im Chart-Bereich
                line_x = chart_start_x + int((mc_start_index / 180) * chart_width)
                
                # Dicke rote Linie zeichnen
                cv2.line(annotated_image, (line_x, 0), (line_x, height), (0, 0, 255), thickness=4)
                
                # Text hinzufügen
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = "MONTE CARLO START"
                text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
                text_x = line_x - text_size[0] // 2
                text_y = 30
                
                # Weißer Hintergrund für Text
                cv2.rectangle(annotated_image, (text_x - 5, text_y - 20), 
                             (text_x + text_size[0] + 5, text_y + 5), (255, 255, 255), -1)
                
                # Roten Text schreiben
                cv2.putText(annotated_image, text, (text_x, text_y), font, 0.7, (0, 0, 255), 2)
            
            # BGR zu RGB konvertieren für PIL
            if annotated_image.shape[2] == 3:  # Falls BGR Format
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            else:
                annotated_image_rgb = annotated_image
                
            annotated_image_pil = Image.fromarray(annotated_image_rgb)
            annotated_image_path = Path(output_folder) / f"{ticker}_annotated.png"
            annotated_image_pil.save(annotated_image_path)
            
        print(f"Erkennungsergebnisse für {ticker} gespeichert in {output_folder}")
        return True
        
    except Exception as e:
        print(f"Fehler beim Speichern der Ergebnisse für {ticker}: {e}")
        return False

def setup_directories():
    """Erstellt notwendige Ordnerstruktur"""
    directories = [TEMP_CHARTS_PATH, RESULTS_PATH, JSON_RESULTS_PATH]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Ordner erstellt/überprüft: {directory}")

def process_ticker_batch_monte_carlo(ticker_list, confidence=CONFIDENCE_THRESHOLD, mc_days=30):
    """Verarbeitet alle Ticker sequenziell mit Monte Carlo Verlängerung"""
    # YOLO-Modell laden
    try:
        model = YOLO(model_path)
        print(f"YOLO-Modell geladen: {model_path}")
    except Exception as e:
        print(f"Fehler beim Laden des YOLO-Modells: {e}")
        return
    
    successful_processed = 0
    total_tickers = len(ticker_list)
    chunk_size = 180  # Definiere chunk_size hier für die Berechnung
    
    print(f"\nStarte Monte Carlo Batch-Verarbeitung von {total_tickers} Tickern...")
    print(f"Monte Carlo Tage: {mc_days}")
    
    for i, ticker in enumerate(ticker_list, 1):
        print(f"\n[{i}/{total_tickers}] Verarbeite {ticker} mit Monte Carlo...")
        
        # 1. Parquet-Daten laden
        data = load_parquet_data(ticker)
        if data is None:
            print(f"Überspringe {ticker} - keine Daten verfügbar")
            continue
        
        # 2. Chart mit Monte Carlo generieren
        chart_path = generate_chart(ticker, data, use_monte_carlo=True, mc_days=mc_days)
        if chart_path is None:
            print(f"Überspringe {ticker} - Chart-Erstellung fehlgeschlagen")
            continue
        
        # 3. Objekterkennung durchführen
        detection_results, yolo_results = perform_detection(chart_path, model, confidence)
        
        # 4. Ergebnisse speichern basierend auf Konfiguration
        should_save = not SAVE_ONLY_WITH_DETECTIONS or len(detection_results) > 0
        
        if should_save:
            # Monte Carlo Ergebnisse in separatem Ordner speichern
            mc_results_path = RESULTS_PATH / "monte_carlo"
            mc_results_path.mkdir(exist_ok=True)
            
            # mc_start_index für die Trennlinie berechnen
            mc_start_index = chunk_size - mc_days
            
            if save_detection_results(ticker, detection_results, yolo_results, mc_results_path, 
                                    is_monte_carlo=True, mc_start_index=mc_start_index):
                successful_processed += 1
                print(f"✓ {ticker} erfolgreich verarbeitet ({len(detection_results)} Erkennungen)")
            else:
                print(f"✗ Fehler beim Speichern der Ergebnisse für {ticker}")
        else:
            print(f"Überspringe {ticker} - keine Erkennungen gefunden")
        
        # 5. Temporäres Chart-Bild löschen
        try:
            Path(chart_path).unlink(missing_ok=True)
        except:
            pass
    
    print(f"\n=== Monte Carlo Batch-Verarbeitung abgeschlossen ===")
    print(f"Erfolgreich verarbeitet: {successful_processed}/{total_tickers} Ticker")

def process_ticker_batch(ticker_list, confidence=CONFIDENCE_THRESHOLD):
    """Verarbeitet alle Ticker sequenziell"""
    # YOLO-Modell laden
    try:
        model = YOLO(model_path)
        print(f"YOLO-Modell geladen: {model_path}")
    except Exception as e:
        print(f"Fehler beim Laden des YOLO-Modells: {e}")
        return
    
    successful_processed = 0
    total_tickers = len(ticker_list)
    
    print(f"\nStarte Batch-Verarbeitung von {total_tickers} Tickern...")
    
    for i, ticker in enumerate(ticker_list, 1):
        print(f"\n[{i}/{total_tickers}] Verarbeite {ticker}...")
        
        # 1. Parquet-Daten laden
        data = load_parquet_data(ticker)
        if data is None:
            print(f"Überspringe {ticker} - keine Daten verfügbar")
            continue
        
        # 2. Chart generieren
        chart_path = generate_chart(ticker, data, use_monte_carlo=False)
        if chart_path is None:
            print(f"Überspringe {ticker} - Chart-Erstellung fehlgeschlagen")
            continue
        
        # 3. Objekterkennung durchführen
        detection_results, yolo_results = perform_detection(chart_path, model, confidence)
        
        # 4. Ergebnisse speichern basierend auf Konfiguration
        should_save = not SAVE_ONLY_WITH_DETECTIONS or len(detection_results) > 0
        
        if should_save:
            if save_detection_results(ticker, detection_results, yolo_results, RESULTS_PATH, 
                                    is_monte_carlo=False):
                successful_processed += 1
                print(f"✓ {ticker} erfolgreich verarbeitet ({len(detection_results)} Erkennungen)")
            else:
                print(f"✗ Fehler beim Speichern der Ergebnisse für {ticker}")
        else:
            print(f"Überspringe {ticker} - keine Erkennungen gefunden")
        
        # 5. Temporäres Chart-Bild löschen
        try:
            Path(chart_path).unlink(missing_ok=True)
        except:
            pass
    
    print(f"\n=== Batch-Verarbeitung abgeschlossen ===")
    print(f"Erfolgreich verarbeitet: {successful_processed}/{total_tickers} Ticker")

def main():
    """Hauptfunktion für Batch-Processing"""
    # Logging konfigurieren
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("=== ChartScanAI Batch-Processing ===")
    
    # 1. Ordnerstruktur einrichten
    setup_directories()
    
    # 2. Ticker-Liste aus Input-Datei lesen
    ticker_list = read_ticker_list(INPUT_FILE)
    if not ticker_list:
        print(f"Keine Ticker gefunden. Erstellen Sie eine {INPUT_FILE} mit einem Ticker pro Zeile.")
        return
    
    # 3. Monte Carlo Modus abfragen
    print("\nOptionen:")
    print("1. Normale Verarbeitung")
    print("2. Monte Carlo Verarbeitung")
    
    choice = input("Wählen Sie eine Option (1 oder 2): ").strip()
    
    if choice == "2":
        mc_days = input("Anzahl Monte Carlo Tage (Standard: 30): ").strip()
        mc_days = int(mc_days) if mc_days.isdigit() else 30
        process_ticker_batch_monte_carlo(ticker_list, CONFIDENCE_THRESHOLD, mc_days)
    else:
        process_ticker_batch(ticker_list)
    
    print("\nBatch-Processing beendet.")

if __name__ == "__main__":
    main()
