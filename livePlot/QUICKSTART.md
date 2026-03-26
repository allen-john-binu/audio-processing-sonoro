# 🚀 QUICK START GUIDE

## Installation (5 minutes)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Place CSV File
```
Create folder: dataFromReal/
Place file: 23_03_data_2.csv in this folder
```

### Step 3: Start Server
```bash
python app.py
```

You should see:
```
[INFO] CSV loaded: dataFromReal/23_03_data_2.csv
[INFO] Starting Flask server on http://localhost:5000
```

### Step 4: Open in Browser
```
http://localhost:5000
```

---

## What Each File Does

| File | Purpose |
|------|---------|
| `app.py` | Main Flask server, data processing, Plotly figure generation |
| `templates/index.html` | Web page layout and slider controls |
| `static/style.css` | Simple styling (minimal, clean) |
| `requirements.txt` | Python package versions |
| `README.md` | Full documentation |

---

## Key Features

✅ **Graph 1 (Interactive)**
- Slider filters data by SPL threshold
- Real-time updates
- Shows compressed time axis (only filtered data)
- Hover to see exact values
- Download as PNG

✅ **Graph 2 (Static)**
- Full dataset without filtering
- Hover to inspect values
- Download as PNG

✅ **Both Graphs**
- Show intensity heatmap + SPL strip
- Show peak DOA as cyan dots
- Responsive design (fits browser width)

---

## How It Works

1. **Server starts** → Loads CSV file once
2. **User opens browser** → Renders both graphs
3. **User moves slider** → JavaScript sends request to Flask
4. **Flask filters data** → Creates new Plotly figure
5. **Server returns HTML** → Browser updates Graph 1
6. **User clicks download** → PNG saved to computer

---

## Customization

### Change Colors
In `app.py`:
```python
colorscale="Inferno"  # Try: "Viridis", "Plasma", "Turbo"
marker=dict(color="cyan")  # Try: "red", "lime", "yellow"
```

### Change Plot Height
In `app.py`:
```python
fig.update_layout(height=600)  # Increase for taller plots
```

### Change Port
In `app.py`, last line:
```python
app.run(debug=True, host='localhost', port=5000)  # Change to 5001, 8000, etc.
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'flask'` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: dataFromReal/23_03_data_2.csv` | Create folder and place CSV file |
| `Address already in use` | Change port in app.py (line ~87) |
| Plots don't update on slider | Check browser console (F12) for errors |
| Can't download PNG | Try different browser or check ad blockers |

---

## File Locations Reference

```
project_root/
├── app.py
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
├── static/
│   └── style.css
└── dataFromReal/
    └── 23_03_data_2.csv  ← YOUR DATA FILE HERE
```

---

## Test It

After running `python app.py`:

1. Open `http://localhost:5000`
2. Move slider left/right → Graph 1 updates
3. Hover over graphs → See values pop up
4. Click "Download" → PNG saved

Done! 🎉
