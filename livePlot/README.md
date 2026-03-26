# DOA (Direction of Arrival) Visualization Dashboard

A Flask + Plotly web application for interactive DOA visualization with SPL threshold filtering.

## Features

- **Graph 1 (Interactive)**
  - Real-time SPL threshold filtering with slider
  - Compressed time axis showing only filtered data
  - Hover tooltips showing time, angle, and intensity values
  - PNG download capability

- **Graph 2 (Static)**
  - Full dataset visualization (no filtering)
  - Displays complete time series
  - Hover tooltips for data inspection
  - PNG download capability

## Project Structure

```
project/
├── app.py                  # Flask application & data processing
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html         # HTML dashboard template
├── static/
│   └── style.css          # CSS styling
└── dataFromReal/
    └── 23_03_data_2.csv   # CSV data file (place here)
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare CSV Data

Place your CSV file at:
```
dataFromReal/23_03_data_2.csv
```

Expected CSV format:
- Column 1: `dB_SPL` (Sound Pressure Level)
- Columns 2+: Angle columns (as floats, e.g., -90, -85.5, -80, ..., 90)
- Each row: one time step with intensity values at different angles

### 3. Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

### 4. Access the Dashboard

Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

### Interacting with Graph 1
1. **Adjust the SPL Threshold Slider** (top graph)
   - Minimum: Lowest SPL value in dataset
   - Maximum: Highest SPL value in dataset
   - The filtered data will update in real-time

2. **Hover Over Data**
   - Time Step: X-axis position
   - Angle (degrees): Y-axis position
   - Intensity/SPL Value: Data value at that point

3. **Download Plots**
   - Click "Download Graph X as PNG" button
   - File will be saved with timestamp

### Graph 2
- Static display of full dataset (no interaction needed)
- Use hover to inspect specific data points
- Download the complete dataset view

## Data Processing

### Graph 1 (Filtered)
- Only rows where `SPL >= threshold` are included
- Time axis is **compressed** to show only valid data
- Peak DOA is recalculated per filtered dataset

### Graph 2 (Full)
- All rows are displayed
- Time axis spans complete dataset
- Peak DOA calculated from full intensity matrix

## Troubleshooting

### Port 5000 Already in Use
```bash
# Change port in app.py
app.run(debug=True, host='localhost', port=5001)
```

### CSV Not Found
- Ensure `dataFromReal/23_03_data_2.csv` exists in project root
- Check file path is correct

### Slider Not Updating
- Check browser console (F12) for JavaScript errors
- Ensure Flask server is running without errors

### Plotly Not Loading
- Internet connection required (CDN-hosted Plotly library)
- Or install `kaleido` for offline rendering:
  ```bash
  pip install kaleido
  ```

## Configuration

### Change CSV File
In `app.py`, line 13:
```python
FILE_PATH = "dataFromReal/your_file.csv"
```

### Change Port
In `app.py`, last line:
```python
app.run(debug=True, host='localhost', port=5000)  # Change 5000
```

### Change Plot Heights
In `app.py`, in `create_filtered_figure()` and `create_static_figure()`:
```python
fig.update_layout(
    height=600,  # Change this value (in pixels)
    ...
)
```

## Performance Notes

- **Initial load**: Creates both plots (can take 1-2 seconds)
- **Slider update**: Filters data and re-renders (0.5-1 second)
- **Large datasets**: May slow down for >10,000 time steps

## Browser Compatibility

- Chrome/Edge: Full support
- Firefox: Full support
- Safari: Full support
- IE11: Not supported (use modern browser)

## License

Free to use and modify.
