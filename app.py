import streamlit as st
import yfinance as yf
import mplfinance as mpf
import pandas as pd
from PIL import Image
from datetime import datetime, timedelta
from io import BytesIO
from ultralytics import YOLO
import os
import glob

# Replace the relative path to your weight file
model_path = 'weights/custom_yolov8.pt'

# Logo URL
logo_url = "images/chartscan.png"

# Setting page layout
st.set_page_config(
    page_title="ChartScanAI",  # Setting page title
    page_icon="📊",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded",    # Expanding sidebar by default
)

# Function to load data from local CSV files
def load_local_data(ticker, data_folder="data"):
    """Load End of Day data from local CSV files"""
    ticker_upper = ticker.upper()
    ticker_folder = os.path.join(data_folder, ticker_upper)
    
    # If exact match doesn't exist, try to find folder with suffix (e.g., AAPL.US)
    if not os.path.exists(ticker_folder):
        # Look for folders that start with the ticker name
        possible_folders = []
        if os.path.exists(data_folder):
            for folder_name in os.listdir(data_folder):
                folder_path = os.path.join(data_folder, folder_name)
                if os.path.isdir(folder_path) and folder_name.upper().startswith(ticker_upper):
                    possible_folders.append(folder_path)
        
        if possible_folders:
            # Use the first matching folder
            ticker_folder = possible_folders[0]
        else:
            return None
    
    if not os.path.exists(ticker_folder):
        return None
    
    # Find CSV files in the ticker folder
    csv_files = glob.glob(os.path.join(ticker_folder, "*.csv"))
    
    if not csv_files:
        return None
    
    # Load and combine all CSV files
    dataframes = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Try to identify date column (common names)
            date_columns = ['Date', 'date', 'DATE', 'Datum', 'datum']
            date_col = None
            for col in date_columns:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)
            
            # Standardize column names to match yfinance format
            column_mapping = {
                'open': 'Open', 'Open': 'Open', 'OPEN': 'Open',
                'high': 'High', 'High': 'High', 'HIGH': 'High',
                'low': 'Low', 'Low': 'Low', 'LOW': 'Low',
                'close': 'Close', 'Close': 'Close', 'CLOSE': 'Close',
                'volume': 'Volume', 'Volume': 'Volume', 'VOLUME': 'Volume',
                'adjusted_close': 'Adj Close', 'Adjusted_Close': 'Adj Close', 
                'ADJUSTED_CLOSE': 'Adj Close', 'adj_close': 'Adj Close'
            }
            
            df.rename(columns=column_mapping, inplace=True)
            dataframes.append(df)
            
        except Exception as e:
            st.warning(f"Fehler beim Laden der Datei {csv_file}: {e}")
            continue
    
    if dataframes:
        # Combine all dataframes and sort by date
        combined_df = pd.concat(dataframes)
        combined_df = combined_df.sort_index()
        # Remove duplicates
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        return combined_df
    
    return None

# Function to download and plot chart
def generate_chart(ticker, interval="1d", chunk_size=180, figsize=(18, 6.5), dpi=100, use_local_data=False, data_folder="data"):
    if use_local_data:
        # Load data from local CSV files
        data = load_local_data(ticker, data_folder)
        if data is None:
            st.error(f"Keine lokalen Daten für {ticker} gefunden. Überprüfen Sie den Ordner {data_folder}/{ticker.upper()}")
            return None
    else:
        if interval == "1h":
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)
            period = None
        else:
            start_date = None
            end_date = None
            period = "max"
        
        # Download data for the ticker
        data = yf.download(ticker, interval=interval, start=start_date, end=end_date, period=period)
    
    # Ensure the index is a DatetimeIndex and check if data is not empty
    if not data.empty:
        data.index = pd.to_datetime(data.index)
        # Select only the latest 180 candles
        data = data.iloc[-chunk_size:]

        # Plot the chart
        fig, ax = mpf.plot(data, type="candle", style="yahoo",
                           title=f"{ticker} Latest {chunk_size} Candles",
                           axisoff=True,
                           ylabel="",
                           ylabel_lower="",
                           volume=False,
                           figsize=figsize,
                           returnfig=True)

        # Save the plot to a BytesIO object
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=dpi)  # Ensure DPI is set here
        buffer.seek(0)
        return buffer
    else:
        st.error("No data found for the specified ticker and interval.")
        return None

# Creating sidebar
with st.sidebar:
    # Add a logo to the top of the sidebar
    st.image(logo_url, use_column_width="auto")
    st.write("")
    st.header("Configurations")     # Adding header to sidebar
    # Section to generate and download chart
    st.subheader("Generate Chart")
    ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL):")
    
    # Data source selection
    data_source = st.radio(
        "Datenquelle wählen:",
        ["Yahoo Finance", "Lokale CSV-Dateien"],
        help="Wählen Sie zwischen Online-Daten von Yahoo Finance oder lokalen CSV-Dateien"
    )
    
    use_local_data = data_source == "Lokale CSV-Dateien"
    
    if use_local_data:
        data_folder = st.text_input("Pfad zum Datenordner:", value="data", 
                                   help="Pfad zum Ordner mit den Ticker-Unterordnern")
        interval = "1d"  # Local data is typically daily
        st.info("Lokale Daten werden als tägliche End-of-Day Daten behandelt")
    else:
        interval = st.selectbox("Select Interval", ["1d", "1h", "1wk"])
        data_folder = "data"  # Default value
    
    chunk_size = 180  # Default chunk size
    if st.button("Generate Chart"):
        if ticker:
            chart_buffer = generate_chart(ticker, interval=interval, chunk_size=chunk_size, 
                                        use_local_data=use_local_data, data_folder=data_folder)
            if chart_buffer:
                st.success(f"Chart generated successfully.")
                st.download_button(
                    label=f"Download {ticker} Chart",
                    data=chart_buffer,
                    file_name=f"{ticker}_latest_{chunk_size}_candles.png",
                    mime="image/png"
                )
                st.image(chart_buffer, caption=f"{ticker} Chart", use_column_width=True)
        else:
            st.error("Please enter a valid ticker symbol.")
    st.write("")
    st.subheader("Upload Image for Detection")
    # Adding file uploader to sidebar for selecting images
    source_img = st.file_uploader(
        "Upload an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    # Model Options
    confidence = float(st.slider(
        "Select Model Confidence", 25, 100, 30)) / 100

# Creating main page heading
st.title("ChartScanAI")
st.caption('📈 To use the app, choose one of the following options:')

st.markdown('''
**Option 1: Upload Your Own Image**
1. **Upload Image:** Use the sidebar to upload a candlestick chart image from your local PC.
2. **Detect Objects:** Click the :blue[Detect Objects] button to analyze the uploaded chart.

**Option 2: Generate and Analyze Chart**
1. **Generate Chart:** Provide the ticker symbol and choose data source (Yahoo Finance or local CSV files) in the sidebar to create and download a chart (latest 180 candles).
2. **Upload Generated Chart:** Use the sidebar to upload the generated chart image.
3. **Detect Objects:** Click the :blue[Detect Objects] button to analyze the generated chart.

**Lokale CSV-Dateien verwenden:**
- Erstellen Sie einen Ordner (z.B. "data") mit Unterordnern für jeden Ticker (z.B. "AAPL", "MSFT")
- Legen Sie CSV-Dateien mit End-of-Day Daten in die entsprechenden Ticker-Ordner
- CSV-Dateien sollten Spalten wie Date, Open, High, Low, Close, Volume enthalten
''')

# Creating two columns on the main page
col1, col2 = st.columns(2)

# Adding image to the first column if image is uploaded
if source_img:
    with col1:
        # Opening the uploaded image
        uploaded_image = Image.open(source_img)
        # Adding the uploaded image to the page with a caption
        st.image(uploaded_image,
                 caption="Uploaded Image",
                 use_column_width=True
                 )

# Load the model
try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

# Perform object detection if the button is clicked
if st.sidebar.button('Detect Objects'):
    if source_img:
        # Re-open the image to reset the file pointer
        source_img.seek(0)
        uploaded_image = Image.open(source_img)
        
        # Perform object detection
        res = model.predict(uploaded_image, conf=confidence)
        boxes = res[0].boxes
        res_plotted = res[0].plot()[:, :, ::-1]
        with col2:
            st.image(res_plotted,
                     caption='Detected Image',
                     use_column_width=True
                     )
            try:
                with st.expander("Detection Results"):
                    for box in boxes:
                        st.write(box.xywh)
            except Exception as ex:
                st.write("Error displaying detection results.")
    else:
        st.error("Please upload an image first.")
