import streamlit as st
import lasio
import segyio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit.components.v1 as components
import os
import base64

# --- 1. THEME & STYLING ---
st.set_page_config(page_title="PUDM Viewer", layout="wide")

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_file = "Web_Background_Black.jpg"

try:
    bin_str = get_base64_of_bin_file(img_file)
    css = f"""
    .stApp {{
        background-image: url("data:image/jpg;base64,{bin_str}");
        background-attachment: fixed;
        background-size: cover;
    }}
    .stApp::before {{
        content: ""; position: absolute; top: 0; left: 0; width: 100%; height: 100%;
        background-color: rgba(0, 0, 0, 0.4); z-index: -1;
    }}
    [class*="st-key-las_"], [class*="st-key-seismic_"] {{
        background-color: rgba(255, 255, 255, 0.12) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 20px !important;
        -webkit-backdrop-filter: blur(20px) saturate(160%) !important;
        backdrop-filter: blur(20px) saturate(160%) !important;
        padding: 2rem !important; margin-bottom: 2rem !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.6) !important;
    }}
    .main-title {{
        text-align: center; color: white; font-size: 42px; font-weight: bold;
        margin-top: 50px; margin-bottom: 80px;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
    }}
    h3 {{ color: #FFFFFF !important; font-weight: bold !important; }}
    .upload-label {{ color: white; text-align: center; font-size: 18px; }}
    """
    st.html(f"<style>{css}</style>")
except FileNotFoundError:
    st.warning(f"Background image '{img_file}' not found.")

# --- 2. HELPER FUNCTIONS (NOW WITH CACHING) ---

@st.cache_data
def detect_endianness(path):
    for endian in ['big', 'little']:
        try:
            with segyio.open(path, "r", ignore_geometry=True, endian=endian) as f:
                fmt = f.bin[segyio.BinField.Format]
                ns = f.bin[segyio.BinField.Samples]
                if 1 <= fmt <= 8 and 0 < ns < 100000 and f.tracecount > 0:
                    return endian
        except Exception:
            continue
    return 'big'

@st.cache_data
def detect_3d_geometry(path, endian):
    il_field = segyio.TraceField.INLINE_3D
    xl_field = segyio.TraceField.CROSSLINE_3D
    try:
        with segyio.open(path, "r", ignore_geometry=False, endian=endian) as f:
            if len(f.ilines) > 1 and len(f.xlines) > 1:
                expected = len(f.ilines) * len(f.xlines)
                actual = f.tracecount
                if expected == actual:
                    return ('standard_3d', f.ilines.tolist(), f.xlines.tolist(), f.samples.tolist(), il_field, xl_field, "✅ Standard 3D File")
                else:
                    return ('nonstandard_3d', f.ilines.tolist(), f.xlines.tolist(), f.samples.tolist(), il_field, xl_field, f"⚠️ Irregular 3D (Missing Traces: {expected - actual})")
    except Exception:
        pass
    try:
        with segyio.open(path, "r", ignore_geometry=True, endian=endian) as f:
            ilines = f.attributes(il_field)[:]
            xlines = f.attributes(xl_field)[:]
            unique_il = np.unique(ilines)
            unique_xl = np.unique(xlines)
            if len(unique_il) > 1 and len(unique_xl) > 1:
                return ('nonstandard_3d', sorted(unique_il.tolist()), sorted(unique_xl.tolist()), f.samples.tolist(), il_field, xl_field, "❌ Broken 3D File (Polygon / Grid Corrupted)")
            else:
                return ('2d', None, None, None, None, None, "📄 2D Seismic File")
    except Exception:
        return ('corrupted', None, None, None, None, None, "🚫 Heavily Corrupted (Cannot Parse)")

@st.cache_data
def get_polygon_headers(path, endian, il_field, xl_field):
    with segyio.open(path, "r", ignore_geometry=True, endian=endian) as f:
        return f.attributes(il_field)[:], f.attributes(xl_field)[:]

@st.cache_data
def get_textual_header(path, endian):
    try:
        with segyio.open(path, "r", ignore_geometry=True, endian=endian) as f:
            return segyio.tools.wrap(f.text[0])
    except Exception as e:
        return f"Could not read EBCDIC header: {e}"

@st.cache_data
def get_binary_header_summary(path, endian):
    try:
        with segyio.open(path, "r", ignore_geometry=True, endian=endian) as f:
            fmt_code = f.bin[segyio.BinField.Format]
            fmt_dict = {1: "IBM Float (32-bit)", 2: "INT32", 3: "INT16", 5: "IEEE Float (32-bit)", 8: "INT8"}
            data_format = fmt_dict.get(fmt_code, f"Unknown ({fmt_code})")
            
            interval_us = f.bin[segyio.BinField.Interval]
            interval_ms = interval_us / 1000.0 if interval_us else 0
            
            samples = f.bin[segyio.BinField.Samples]
            time_max = (samples - 1) * interval_ms if samples and interval_ms else 0
            
            return {
                "Byte Order": "Big Endian" if endian == "big" else "Little Endian",
                "Data Format": data_format,
                "Max Samples": samples,
                "Interval": f"{interval_ms} ms",
                "Time Range": f"0 - {time_max} ms"
            }
    except Exception as e:
        return None

@st.cache_data(show_spinner=False)
def scan_full_geometry(path, endian):
    """PHASE B: Scans all trace headers to find absolute MIN and MAX coordinate values."""
    try:
        with segyio.open(path, "r", ignore_geometry=True, endian=endian) as f:
            def get_ext(field):
                try:
                    arr = f.attributes(field)[:]
                    min_v, max_v = np.min(arr), np.max(arr)
                    
                    # Smart formatting: Int if integer, 2 decimals if float
                    fmt_min = f"{int(min_v)}" if min_v == int(min_v) else f"{float(min_v):.2f}"
                    fmt_max = f"{int(max_v)}" if max_v == int(max_v) else f"{float(max_v):.2f}"
                    
                    return fmt_min, fmt_max
                except:
                    return "N/A", "N/A"
            
            return {
                "Field Record (fldr)": get_ext(segyio.TraceField.FieldRecord),
                "Energy Src Point (ESP)": get_ext(segyio.TraceField.EnergySourcePoint),
                "CDP": get_ext(segyio.TraceField.CDP),
                "Inline": get_ext(segyio.TraceField.INLINE_3D),
                "Crossline": get_ext(segyio.TraceField.CROSSLINE_3D),
                "Offset": get_ext(segyio.TraceField.offset),
                "Source X": get_ext(segyio.TraceField.SourceX),
                "Source Y": get_ext(segyio.TraceField.SourceY),
                "Receiver X": get_ext(segyio.TraceField.GroupX),
                "Receiver Y": get_ext(segyio.TraceField.GroupY),
                "CDP X": get_ext(segyio.TraceField.CDP_X),
                "CDP Y": get_ext(segyio.TraceField.CDP_Y),
            }
    except Exception as e:
        return None

@st.cache_data
def get_las_section_df(section_dict):
    """Safely extracts LAS section items into a clean 4-column Pandas DataFrame."""
    data = []
    for item in section_dict:
        data.append({
            "Mnemonic": item.mnemonic,
            "Unit": item.unit if item.unit else "",
            "Value": str(item.value) if item.value else "",
            "Description": item.descr if item.descr else ""
        })
    return pd.DataFrame(data)

# --- STATE CALLBACKS FOR UI BUTTONS ---
def set_val(key, val):
    st.session_state[key] = val

def add_val(key, delta, min_val, max_val):
    st.session_state[key] = max(min_val, min(max_val, st.session_state[key] + delta))

# --- 3. STATE MANAGEMENT ---
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'file_path' not in st.session_state:
    st.session_state.file_path = None

# --- 4. PAGE LOGIC ---

# PAGE A: HOME
if st.session_state.page == 'home':
    st.markdown('<div class="main-title">PUDM WELL LOG AND SEISMIC VIEWER</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        file_path = st.text_input("", placeholder="Enter local file path (e.g. C:/Data/Seismic.sgy)", label_visibility="collapsed")
        st.markdown('<div class="upload-label">Enter the full local path to your file<br>LAS, SGY</div>', unsafe_allow_html=True)
        if file_path:
            clean_path = file_path.strip().strip('"').strip("'")
            if not os.path.isfile(clean_path):
                st.error("File not found. Please check the path and ensure the app has access.")
            else:
                ext = clean_path.split('.')[-1].lower()
                if ext not in ['las', 'sgy']:
                    st.error("Unsupported file type. Please use .las or .sgy files.")
                else:
                    st.session_state.file_path = clean_path
                    st.session_state.page = 'well_log' if ext == 'las' else 'seismic'
                    st.rerun()

# PAGE B: WELL LOG VIEW
elif st.session_state.page == 'well_log':
    col_t1, col_t2 = st.columns([12, 1])
    if col_t2.button("☁️"):
        st.session_state.page = 'home'
        st.rerun()

    las_path = st.session_state.file_path
    try:
        las = lasio.read(las_path)
        df = las.df().reset_index()
        with st.container(key="las_info"):
            st.subheader("WELL LOG INFORMATIONS")
            run_number = "Unknown"
            for block in [las.params, las.well]:
                for item in block:
                    if 'RUN' in item.mnemonic.upper():
                        run_number = item.value
                        break
                if run_number != "Unknown": break
            strt = las.well['STRT'].value if 'STRT' in las.well else "N/A"
            stop = las.well['STOP'].value if 'STOP' in las.well else "N/A"
            unit = las.well['STRT'].unit if 'STRT' in las.well else "m"
            st.info(f"📍 **Logged Interval:** {strt} to {stop} {unit} &nbsp; | &nbsp; **Detected Run:** {run_number}")
            
            if las.version:
                with st.expander("Version Information (~V)"):
                    st.dataframe(get_las_section_df(las.version), hide_index=True, use_container_width=True)
            if las.well:
                with st.expander("Well Information (~W)", expanded=True):
                    st.dataframe(get_las_section_df(las.well), hide_index=True, use_container_width=True)
            if las.params:
                with st.expander("Parameter Information (~P)"):
                    st.dataframe(get_las_section_df(las.params), hide_index=True, use_container_width=True)
            if las.other:
                with st.expander("Other Information (~O)"): 
                    st.text(las.other) # Leaving this as text because it's just raw paragraphs!

        with st.container(key="las_curves"):
            st.subheader("Well Log Curves")
            depth_col = next((c for c in las.keys() if c.upper() in ['DEPT', 'DEPTH']), None)
            
            if depth_col:
                available_curves = [c for c in las.keys() if c != depth_col]
                num_curves = len(available_curves)
                cols_per_row = 5
                num_rows = (num_curves + cols_per_row - 1) // cols_per_row
                depth_unit = las.curves[depth_col].unit if depth_col in las.curves else "m"

                # --- NEW: MASTER COLOR PICKER FOR BASE CURVES ---
                default_colors = [
                    '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', 
                    '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f'
                ]
                
                single_curve_colors = {}
                with st.expander("🎨 Customize Individual Curve Colors", expanded=False):
                    color_cols = st.columns(5)
                    for idx, curve in enumerate(available_curves):
                        with color_cols[idx % 5]:
                            single_curve_colors[curve] = st.color_picker(
                                f"{curve}", 
                                value=default_colors[idx % len(default_colors)], 
                                key=f"base_color_{curve}"
                            )
                
                st.write("") # Quick spacer

                # --- NEW: PLOTLY SUBPLOT ENGINE ---
                for r in range(num_rows):
                    start_idx = r * cols_per_row
                    end_idx = min(start_idx + cols_per_row, num_curves)
                    row_curves = available_curves[start_idx:end_idx]
                    
                    # We keep your brilliant column-ratio math to prevent stretched plots!
                    display_cols = st.columns([len(row_curves), cols_per_row - len(row_curves) + 0.1])
                    
                    with display_cols[0]:
                        # 1. Create synced subplots for this specific row
                        fig_row = make_subplots(
                            rows=1, cols=len(row_curves), 
                            shared_yaxes=True, 
                            horizontal_spacing=0.05, # Keep the tracks tight
                            subplot_titles=row_curves
                        )
                        
                        # 2. Draw each curve into its respective column
                        for i, curve in enumerate(row_curves):  
                            c_color = single_curve_colors[curve]
                            
                            fig_row.add_trace(go.Scatter(
                                x=df[curve],
                                y=df[depth_col],
                                name=curve,
                                mode='lines',
                                line=dict(color=c_color, width=1.5)
                            ), row=1, col=i+1)
                            
                            # Logarithmic scale check for Resistivity
                            is_log = 'RES' in curve.upper() and df[curve].min() > 0
                            
                            fig_row.update_xaxes(
                                type='log' if is_log else 'linear',
                                showgrid=True, gridcolor='#E0E0E0', color='white',
                                row=1, col=i+1
                            )

                        fig_row.update_yaxes(
                            title_text=f"Depth ({depth_unit})" if i == 0 else "", 
                            autorange='reversed', 
                            showgrid=True, 
                            gridcolor='#E0E0E0', 
                            color='white',

                            showspikes=True,       # Force the line to appear
                            spikecolor='#000000',  # Pure Black
                            spikethickness=1.0,    # Slightly thicker than a default line
                            spikedash='dash',     # 'solid', 'dot', or 'dash'
                            spikemode='across'
                        )

                        fig_row.update_yaxes(
                            title_text=f"Depth ({depth_unit})", 
                            row=1, col=1
                        )

                        # 4. Global UI Styling
                        fig_row.update_layout(
                            plot_bgcolor='#FFFFFF',
                            paper_bgcolor='rgba(0,0,0,0)',
                            height=800, # Stretch them tall so they look like real well logs
                            showlegend=False, # Titles are at the top, no legend needed
                            margin=dict(t=50, b=50, l=60, r=20),
                            hovermode='y unified' # Cross-track tooltip sync!
                        )
                        
                        # 5. Force the subplot titles to be white AND push them up
                        for annotation in fig_row['layout']['annotations']: 
                            annotation['font'] = dict(color='white', size=14, weight='bold')
                            
                            # UPGRADE 2: Push the titles exactly 15 pixels UP away from the plot line
                            annotation['yshift'] = 15 

                        # Render the chart
                        st.plotly_chart(fig_row, use_container_width=True, config={'displayModeBar': True}, key=f"plotly_base_row_{r}")

                        # --- NEW: MULTI-TRACK CUSTOM STACKED VIEW ---
                st.markdown("---")
                st.markdown("### 🎯 Advanced Multi-Track Analysis")
                st.markdown("Build up to 3 custom stacked tracks side-by-side for cross-correlation.")
                
                # Create 3 columns for our tracks
                track_cols = st.columns(3)
                
                # Loop through the columns to build them dynamically
                for i, col in enumerate(track_cols):
                    with col:
                        st.markdown(f"#### Track {i+1}")
                        
                        stacked_curves = st.multiselect(
                            f"Select curves:", 
                            available_curves, 
                            key=f"custom_track_{i}"
                        )
                        
                        if stacked_curves:
                            import plotly.graph_objects as go
                            
                            fig_stack = go.Figure()
                            
                            # Our Enterprise default palette
                            default_colors = [
                                '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', 
                                '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f'
                            ]
                            
                            # --- NEW: DYNAMIC COLOR PICKER UI ---
                            # --- NEW: COMPACT DYNAMIC COLOR PICKER UI ---
                            curve_colors = {}
                            with st.expander("🎨 Customize Curve Colors", expanded=False):
                                # UPGRADE: Increased to 5 columns for a much tighter, horizontal grid
                                num_cols = 5
                                color_cols = st.columns(num_cols)
                                
                                for j, curve in enumerate(stacked_curves):
                                    default_c = default_colors[j % len(default_colors)]
                                    
                                    with color_cols[j % num_cols]:
                                        # UPGRADE: Dropped the word "Color" to keep the label ultra-clean
                                        curve_colors[curve] = st.color_picker(
                                            f"{curve}", 
                                            value=default_c, 
                                            key=f"color_picker_track_{i}_{curve}"
                                        )

                            # --- THE PLOTLY ENGINE ---
                            layout_updates = {
                                'yaxis': dict(
                                    title=f"Depth ({depth_unit})", 
                                    autorange='reversed', 
                                    gridcolor='#E0E0E0',
                                    color='white',

                                    showspikes=True,       # Force the line to appear
                                    spikecolor='#000000',  # Pure Black
                                    spikethickness=1.0,    # Slightly thicker than a default line
                                    spikedash='dash',     # 'solid', 'dot', or 'dash'
                                    spikemode='across'
                                ),
                                'xaxis': dict(
                                    side='top',
                                    showgrid=True, 
                                    gridcolor='#E0E0E0',
                                    color='white',
                                    tickfont=dict(color='white')
                                ),
                                'plot_bgcolor': '#FFFFFF', 
                                'paper_bgcolor': 'rgba(0,0,0,0)', 
                                'margin': dict(t=60, b=150, l=50, r=20), 
                                'showlegend': True, 
                                'legend': dict(
                                    orientation="h",
                                    yanchor="top",
                                    y=-0.05, 
                                    xanchor="center",
                                    x=0.5,
                                    font=dict(color='white')
                                ),
                                'hovermode': 'y unified' 
                            }

                            for j, curve in enumerate(stacked_curves):
                                # CRITICAL: We now use the user's chosen color from our dictionary!
                                c_color = curve_colors[curve]
                                
                                fig_stack.add_trace(go.Scatter(
                                    x=df[curve],
                                    y=df[depth_col],
                                    name=curve,
                                    mode='lines',
                                    line=dict(color=c_color, width=1.5)
                                ))

                            fig_stack.update_layout(**layout_updates)
                            
                            st.plotly_chart(fig_stack, use_container_width=True, height=800, config={'displayModeBar': True}, key=f"plotly_track_{i}")
                        else:
                            st.info(f"Select curves to populate Track {i+1}")

        # --- NEW: 3D WELLBORE TUBE VIEWER ---
        with st.container(key="las_3d"):
            st.markdown("---") # Visual Separator
            st.subheader("☁️ Interactive 3D Wellbore Trajectory")
            
            if depth_col and len(available_curves) > 0:
                # 1. UI: Let the user choose what "paint" to put on the pipe
                c1, c2 = st.columns([1, 3])
                target_curve = c1.selectbox("Select Curve to Paint on 3D Tube:", available_curves)
                
                if c1.button("Generate 3D Digital Core", use_container_width=True):
                    with st.spinner(f"Building 3D Tube mapped with {target_curve}..."):
                        import pyvista as pv
                        import streamlit.components.v1 as components
                        
                        # Drop any empty rows so the line doesn't break
                        clean_df = df[[depth_col, target_curve]].dropna()
                        
                        # 2. THE SCAFFOLDING (X, Y, Z coordinates)
                        # We use -Z so the well points downward in 3D space!
                        z_coords = -clean_df[depth_col].values
                        x_coords = np.zeros_like(z_coords)
                        y_coords = np.zeros_like(z_coords)
                        points = np.column_stack((x_coords, y_coords, z_coords))

                        # 3. Build a perfect point-to-point PyVista Line
                        poly = pv.PolyData(points)
                        # Connect the dots (Line from pt 0 to 1, 1 to 2, etc.)
                        lines = np.full((len(points)-1, 3), 2, dtype=np.int_)
                        lines[:, 1] = np.arange(0, len(points)-1)
                        lines[:, 2] = np.arange(1, len(points))
                        poly.lines = lines

                        # 4. THE PAINT
                        # Attach the selected curve data to the center line
                        poly.point_data[target_curve] = clean_df[target_curve].values

                        # 5. THE GEOMETRY
                        # Blow the line up into a cylinder. 
                        # We dynamically calculate the radius based on depth so it isn't infinitely thin
                        depth_range = clean_df[depth_col].max() - clean_df[depth_col].min()
                        dynamic_radius = max(5.0, depth_range * 0.015) 
                        tube = poly.tube(radius=dynamic_radius)

                        # 6. THE THEATER (Render and Export)
                        plotter = pv.Plotter(window_size=[800, 600], off_screen=True)
                        plotter.background_color = "#1E1E1E"
                        
                        # We use 'jet' cmap, a classic oil & gas standard for well logs
                        plotter.add_mesh(tube, scalars=target_curve, cmap="jet",
                                         show_scalar_bar=True, scalar_bar_args={'title': f"{target_curve} Value"})
                        
                        plotter.add_axes(line_width=5, labels_off=False)
                        plotter.view_isometric()

                        html_file = "temp_well_3d.html"
                        plotter.export_html(html_file)
                        plotter.close()

                    # 7. THE DISPLAY
                    with open(html_file, 'r', encoding='utf-8') as f:
                        source_html = f.read()
                        
                    with c2:
                        components.html(source_html, width=800, height=600)

    except Exception as e:
        st.error(f"Error reading LAS: {e}")

# PAGE C: SEISMIC VIEW
elif st.session_state.page == 'seismic':
    col_t1, col_t2 = st.columns([12, 1])
    if col_t2.button("☁️"):
        st.session_state.page = 'home'
        st.rerun()

    sgy_path = st.session_state.file_path

    try:
        endian = detect_endianness(sgy_path)
        mode, ilines, xlines, samples_list, det_il_field, det_xl_field, diag_msg = detect_3d_geometry(sgy_path, endian)

        with st.container(key="seismic_info"):
            st.subheader("SEISMIC INFORMATIONS")
            
            # --- CONSOLIDATED METRICS DASHBOARD ---
            bin_summary = get_binary_header_summary(sgy_path, endian) or {}
            
            st.write(f"**Analysis:** {diag_msg}")
            
            c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
            if mode in ['standard_3d', 'nonstandard_3d']:
                c1.write(f"**Inlines**<br>{len(ilines)}", unsafe_allow_html=True)
                c2.write(f"**Crosslines**<br>{len(xlines)}", unsafe_allow_html=True)
            elif mode == '2d':
                with segyio.open(sgy_path, "r", ignore_geometry=True, endian=endian) as f_meta:
                    c1.write(f"**Traces**<br>{f_meta.tracecount}", unsafe_allow_html=True)
                    c2.write(f"**Samples**<br>{len(f_meta.samples)}", unsafe_allow_html=True)
            else:
                c1.write("**Traces**<br>N/A", unsafe_allow_html=True)
                c2.write("**Samples**<br>N/A", unsafe_allow_html=True)
                
            c3.write(f"**Byte Order**<br>{bin_summary.get('Byte Order', 'N/A')}", unsafe_allow_html=True)
            c4.write(f"**Data Format**<br>{bin_summary.get('Data Format', 'N/A')}", unsafe_allow_html=True)
            c5.write(f"**Interval**<br>{bin_summary.get('Interval', 'N/A')}", unsafe_allow_html=True)
            c6.write(f"**Max Samples**<br>{bin_summary.get('Max Samples', 'N/A')}", unsafe_allow_html=True)
            c7.write(f"**Time Range**<br>{bin_summary.get('Time Range', 'N/A')}", unsafe_allow_html=True)

            # Raw EBCDIC Textual Header
            ebcdic_text = get_textual_header(sgy_path, endian)
            with st.expander("📄 View Raw SEG-Y Textual Header (EBCDIC)"):
                st.code(ebcdic_text, language="text")
                
            # Deep Scan Geometry
            with st.expander("🔍 Deep Scan: Full Coordinate & Geometry Limits"):
                st.write("Scan every trace header to extract exact physical boundaries (like the desktop app).")
                if st.button("Run Deep Trace Scan"):
                    with st.spinner("Scanning hundreds of thousands of headers..."):
                        geo_stats = scan_full_geometry(sgy_path, endian)
                        if geo_stats:
                            scan_data = [{"Header Attribute": k, "MIN": v[0], "MAX": v[1]} for k, v in geo_stats.items()]
                            st.dataframe(scan_data)

        with st.container(key="seismic_plots"):
            st.subheader("SEISMIC PLOT")

            seismic_colors = {
                "Red-Blue": "RdBu",
                "Grey Scale": "gray",
                "Red-Black": ["black", "white", "red"],
                "Blue-Yellow": ["blue", "white", "yellow"]
            }
            
            # --- SLOT 1: Standard 3D ---
            if mode == 'standard_3d':
                with segyio.open(sgy_path, "r", ignore_geometry=False, endian=endian) as f3d:
                    if 'idx_il_1' not in st.session_state: st.session_state.idx_il_1 = len(ilines) // 2
                    if 'idx_xl_1' not in st.session_state: st.session_state.idx_xl_1 = len(xlines) // 2
                    
                    # --- THE UI TOGGLE ---
                    st.markdown("### 🎯 Display Engine")
                    view_engine = st.radio("Select Rendering Mode", 
                                           ["2D Vector Engine (Plotly)", "3D Cloud Streaming (PyVista)"], 
                                           horizontal=True, label_visibility="collapsed")
                    st.markdown("---")
                    
                    if view_engine == "2D Vector Engine (Plotly)":
                        # --- INLINE SECTION (Plotly) ---
                        st.markdown("### 🎯 Inline Target")
                        c1, c2, c3, _ = st.columns([1, 3, 1, 7])
                        c1.button("⏮️", key="start_il1", on_click=set_val, args=('idx_il_1', 0), use_container_width=True)
                        c2.number_input("IL", min_value=0, max_value=len(ilines)-1, key='idx_il_1', label_visibility="collapsed")
                        c3.button("⏭️", key="end_il1", on_click=set_val, args=('idx_il_1', len(ilines)-1), use_container_width=True)

                        # --- NEW: COLOR & GAIN UI ---
                        ui1, ui2, _ = st.columns([3, 4, 5])
                        cmap_il1 = ui1.selectbox("Color Palette", list(seismic_colors.keys()), key="cmap_il1")
                        gain_il1 = ui2.slider("Amplitude Thickness (Clip %)", min_value=50, max_value=100, value=98, step=1, key="gain_il1")

                        mid_il = ilines[st.session_state.idx_il_1]
                        data_il = f3d.iline[mid_il].T
                        
                        # --- UPGRADED: DYNAMIC PERCENTILE GAIN ---
                        vm_il = np.percentile(np.absolute(data_il), gain_il1)
                        
                        # --- UPGRADED: DYNAMIC COLOR SCALE ---
                        fig_il = px.imshow(data_il, color_continuous_scale=seismic_colors[cmap_il1], range_color=[-vm_il, vm_il],
                                           x=xlines, y=samples_list, aspect='auto', title=f"3D Inline: {mid_il}",
                                           labels={"x": "Crossline", "y": "Time (ms)", "color": "Amplitude"})
                        fig_il.update_layout(plot_bgcolor='#E0E0E0', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                        fig_il.update_traces(zsmooth='best')
                        st.plotly_chart(fig_il, use_container_width=True, height=700)

                        st.markdown("---")

                        # --- CROSSLINE SECTION (Plotly) ---
                        st.markdown("### 🎯 Crossline Target")
                        c1, c2, c3, _ = st.columns([1, 3, 1, 7])
                        c1.button("⏮️", key="start_xl1", on_click=set_val, args=('idx_xl_1', 0), use_container_width=True)
                        c2.number_input("XL", min_value=0, max_value=len(xlines)-1, key='idx_xl_1', label_visibility="collapsed")
                        c3.button("⏭️", key="end_xl1", on_click=set_val, args=('idx_xl_1', len(xlines)-1), use_container_width=True)

                        # --- NEW: COLOR & GAIN UI ---
                        ui1, ui2, _ = st.columns([3, 4, 5])
                        cmap_xl1 = ui1.selectbox("Color Palette", list(seismic_colors.keys()), key="cmap_xl1")
                        gain_xl1 = ui2.slider("Amplitude Thickness (Clip %)", min_value=50, max_value=100, value=98, step=1, key="gain_xl1")

                        mid_xl = xlines[st.session_state.idx_xl_1]
                        data_xl = f3d.xline[mid_xl].T
                        
                        vm_xl = np.percentile(np.absolute(data_xl), gain_xl1)
                        
                        fig_xl = px.imshow(data_xl, color_continuous_scale=seismic_colors[cmap_xl1], range_color=[-vm_xl, vm_xl],
                                           x=ilines, y=samples_list, aspect='auto', title=f"3D Crossline: {mid_xl}",
                                           labels={"x": "Inline", "y": "Time (ms)", "color": "Amplitude"})
                        fig_xl.update_layout(plot_bgcolor='#E0E0E0', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                        fig_xl.update_traces(zsmooth='best')
                        st.plotly_chart(fig_xl, use_container_width=True, height=700)
                        
                    else:
                        # --- 3D PYVISTA ENGINE (NATIVE EXPORT) ---
                        import pyvista as pv
                        
                        st.markdown("### ☁️ Interactive 3D Intersection")
                        
                        tc1, tc2 = st.columns(2)
                        tc1.number_input("Target Inline (X-Axis)", min_value=0, max_value=len(ilines)-1, key='idx_il_1')
                        tc2.number_input("Target Crossline (Y-Axis)", min_value=0, max_value=len(xlines)-1, key='idx_xl_1')
                        
                        mid_il = ilines[st.session_state.idx_il_1]
                        mid_xl = xlines[st.session_state.idx_xl_1]

                        # --- THE VAULT (Extract & Close) ---
                        with st.spinner("Extracting Slices from Disk..."):
                            with segyio.open(sgy_path, "r", ignore_geometry=False, endian=endian) as f3d:
                                data_il = np.copy(f3d.iline[mid_il])
                                data_xl = np.copy(f3d.xline[mid_xl])

                        # --- THE RENDERER (Build & Save) ---
                        with st.spinner("Building 3D Mesh and Exporting Engine..."):
                            vm = np.percentile(np.absolute(data_il), 98)

                            y_grid_il, z_grid_il = np.meshgrid(xlines, samples_list, indexing='ij')
                            x_grid_il = np.full_like(y_grid_il, mid_il)
                            
                            x_grid_xl, z_grid_xl = np.meshgrid(ilines, samples_list, indexing='ij')
                            y_grid_xl = np.full_like(x_grid_xl, mid_xl)

                            mesh_il = pv.StructuredGrid(x_grid_il, y_grid_il, z_grid_il)
                            mesh_xl = pv.StructuredGrid(x_grid_xl, y_grid_xl, z_grid_xl)

                            mesh_il.point_data["Amplitude"] = data_il.flatten()
                            mesh_xl.point_data["Amplitude"] = data_xl.flatten()

                            # Create Plotter in off-screen mode
                            plotter = pv.Plotter(window_size=[800, 600], off_screen=True)
                            plotter.background_color = "#1E1E1E"
                            plotter.set_scale(zscale=-1)
                            
                            plotter.add_mesh(mesh_il, scalars="Amplitude", cmap="RdBu", clim=[-vm, vm], show_scalar_bar=False)
                            plotter.add_mesh(mesh_xl, scalars="Amplitude", cmap="RdBu", clim=[-vm, vm], show_scalar_bar=True)
                            
                            plotter.add_axes(line_width=5, labels_off=False)
                            plotter.view_isometric()

                            # --- THE BULLETPROOF FIX ---
                            # We export the 3D scene to a lightweight vtk.js HTML file
                            html_file = "temp_seismic_3d.html"
                            plotter.export_html(html_file)
                            plotter.close() # Free up the Server RAM immediately!

                        # --- THE DISPLAY ---
                        # Streamlit just reads the text file and displays it safely
                        with open(html_file, 'r', encoding='utf-8') as f:
                            source_html = f.read()
                            
                        components.html(source_html, width=800, height=600)

            # --- SLOT 2: Nonstandard 3D (Enterprise Grid Padder) ---
            elif mode == 'nonstandard_3d':
                with st.spinner(f"🛠️ Reconstructing Polygon Grid with Native Padding..."):
                    try:
                        if 'idx_il_2' not in st.session_state: st.session_state.idx_il_2 = len(ilines) // 2
                        if 'idx_xl_2' not in st.session_state: st.session_state.idx_xl_2 = len(xlines) // 2
                        
                        # --- THE UI TOGGLE ---
                        st.markdown("### 🎯 Display Engine")
                        view_engine_2 = st.radio("Select Rendering Mode", 
                                               ["2D Vector Engine (Plotly)", "3D Cloud Streaming (PyVista)"], 
                                               horizontal=True, label_visibility="collapsed", key="toggle_slot2")
                        st.markdown("---")

                        # Cache pull for headers
                        all_il, all_xl = get_polygon_headers(sgy_path, endian, det_il_field, det_xl_field)
                        xl_to_idx = {val: i for i, val in enumerate(xlines)}
                        il_to_idx = {val: i for i, val in enumerate(ilines)}

                        if view_engine_2 == "2D Vector Engine (Plotly)":
                            with segyio.open(sgy_path, "r", ignore_geometry=True, endian=endian) as f:
                                # --- INLINE SECTION ---
                                st.markdown("### 🎯 Recovered Inline Target")
                                c1, c2, c3, _ = st.columns([1, 3, 1, 7])
                                c1.button("⏮️", key="start_il2", on_click=set_val, args=('idx_il_2', 0), use_container_width=True)
                                c2.number_input("IL", min_value=0, max_value=len(ilines)-1, key='idx_il_2', label_visibility="collapsed")
                                c3.button("⏭️", key="end_il2", on_click=set_val, args=('idx_il_2', len(ilines)-1), use_container_width=True)

                                # --- NEW: COLOR & GAIN UI ---
                                ui1, ui2, _ = st.columns([3, 4, 5])
                                cmap_il2 = ui1.selectbox("Color Palette", list(seismic_colors.keys()), key="cmap_il2")
                                gain_il2 = ui2.slider("Amplitude Thickness (Clip %)", min_value=50, max_value=100, value=98, step=1, key="gain_il2")

                                mid_il = ilines[st.session_state.idx_il_2]
                                tr_idx_il = np.where(all_il == mid_il)[0]
                                data_il = np.full((len(samples_list), len(xlines)), np.nan)
                                for idx in tr_idx_il:
                                    xl_val = all_xl[idx]
                                    if xl_val in xl_to_idx:
                                        data_il[:, xl_to_idx[xl_val]] = f.trace[idx]

                                if len(tr_idx_il) > 0:
                                    vm_il = np.nanpercentile(np.absolute(data_il), gain_il2) 
                                    fig_il = px.imshow(data_il, color_continuous_scale=seismic_colors[cmap_il2], range_color=[-vm_il, vm_il],
                                                       x=xlines, y=samples_list, aspect='auto', title=f"Reconstructed 3D Inline: {mid_il}",
                                                       labels={"x": "Crossline", "y": "Time (ms)", "color": "Amplitude"})
                                    fig_il.update_layout(plot_bgcolor='#E0E0E0', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                                    fig_il.update_traces(zsmooth='best')
                                    st.plotly_chart(fig_il, use_container_width=True, height=700)

                                st.markdown("---") # Visual Separator

                                # --- CROSSLINE SECTION ---
                                st.markdown("### 🎯 Recovered Crossline Target")
                                c1, c2, c3, _ = st.columns([1, 3, 1, 7])
                                c1.button("⏮️", key="start_xl2", on_click=set_val, args=('idx_xl_2', 0), use_container_width=True)
                                c2.number_input("XL", min_value=0, max_value=len(xlines)-1, key='idx_xl_2', label_visibility="collapsed")
                                c3.button("⏭️", key="end_xl2", on_click=set_val, args=('idx_xl_2', len(xlines)-1), use_container_width=True)

                                # --- NEW: COLOR & GAIN UI ---
                                ui1, ui2, _ = st.columns([3, 4, 5])
                                cmap_xl2 = ui1.selectbox("Color Palette", list(seismic_colors.keys()), key="cmap_xl2")
                                gain_xl2 = ui2.slider("Amplitude Thickness (Clip %)", min_value=50, max_value=100, value=98, step=1, key="gain_xl2")

                                mid_xl = xlines[st.session_state.idx_xl_2]
                                tr_idx_xl = np.where(all_xl == mid_xl)[0]
                                data_xl = np.full((len(samples_list), len(ilines)), np.nan)
                                for idx in tr_idx_xl:
                                    il_val = all_il[idx]
                                    if il_val in il_to_idx:
                                        data_xl[:, il_to_idx[il_val]] = f.trace[idx]
                                        
                                if len(tr_idx_xl) > 0:
                                    vm_xl = np.nanpercentile(np.absolute(data_xl), gain_xl2)
                                    fig_xl = px.imshow(data_xl, color_continuous_scale=seismic_colors[cmap_xl2], range_color=[-vm_xl, vm_xl],
                                                       x=ilines, y=samples_list, aspect='auto', title=f"Reconstructed 3D Crossline: {mid_xl}",
                                                       labels={"x": "Inline", "y": "Time (ms)", "color": "Amplitude"})
                                    fig_xl.update_layout(plot_bgcolor='#E0E0E0', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                                    fig_xl.update_traces(zsmooth='best')
                                    st.plotly_chart(fig_xl, use_container_width=True, height=700)
                                    
                        else:
                            # --- 3D PYVISTA ENGINE (GRID PADDER) ---
                            import pyvista as pv
                            import streamlit.components.v1 as components
                            
                            st.markdown("### ☁️ Interactive 3D Intersection (Irregular Grid)")
                            
                            tc1, tc2 = st.columns(2)
                            tc1.number_input("Target Inline (X-Axis)", min_value=0, max_value=len(ilines)-1, key='idx_il_2')
                            tc2.number_input("Target Crossline (Y-Axis)", min_value=0, max_value=len(xlines)-1, key='idx_xl_2')
                            
                            mid_il = ilines[st.session_state.idx_il_2]
                            mid_xl = xlines[st.session_state.idx_xl_2]

                            # --- THE VAULT (Extract & Pad Traces) ---
                            with st.spinner("Extracting and Padding Irregular Geometry..."):
                                with segyio.open(sgy_path, "r", ignore_geometry=True, endian=endian) as f:
                                    # Create shape (traces, samples) to perfectly map to PyVista Meshgrid
                                    data_il_3d = np.full((len(xlines), len(samples_list)), np.nan)
                                    tr_idx_il = np.where(all_il == mid_il)[0]
                                    for idx in tr_idx_il:
                                        xl_val = all_xl[idx]
                                        if xl_val in xl_to_idx:
                                            data_il_3d[xl_to_idx[xl_val], :] = f.trace[idx]

                                    data_xl_3d = np.full((len(ilines), len(samples_list)), np.nan)
                                    tr_idx_xl = np.where(all_xl == mid_xl)[0]
                                    for idx in tr_idx_xl:
                                        il_val = all_il[idx]
                                        if il_val in il_to_idx:
                                            data_xl_3d[il_to_idx[il_val], :] = f.trace[idx]

                            # --- THE RENDERER (Build & Export) ---
                            with st.spinner("Building 3D Mesh and Exporting HTML Engine..."):
                                vm = np.nanpercentile(np.absolute(data_il_3d), 98)

                                y_grid_il, z_grid_il = np.meshgrid(xlines, samples_list, indexing='ij')
                                x_grid_il = np.full_like(y_grid_il, mid_il)
                                
                                x_grid_xl, z_grid_xl = np.meshgrid(ilines, samples_list, indexing='ij')
                                y_grid_xl = np.full_like(x_grid_xl, mid_xl)

                                mesh_il = pv.StructuredGrid(x_grid_il, y_grid_il, z_grid_il)
                                mesh_xl = pv.StructuredGrid(x_grid_xl, y_grid_xl, z_grid_xl)

                                mesh_il.point_data["Amplitude"] = data_il_3d.flatten()
                                mesh_xl.point_data["Amplitude"] = data_xl_3d.flatten()

                                plotter = pv.Plotter(window_size=[800, 600], off_screen=True)
                                plotter.background_color = "#1E1E1E"
                                plotter.set_scale(zscale=-1)
                                
                                # CRITICAL: nan_opacity=0.0 makes the padded "empty" space totally invisible!
                                plotter.add_mesh(mesh_il, scalars="Amplitude", cmap="RdBu", clim=[-vm, vm], show_scalar_bar=False, nan_opacity=0.0)
                                plotter.add_mesh(mesh_xl, scalars="Amplitude", cmap="RdBu", clim=[-vm, vm], show_scalar_bar=True, nan_opacity=0.0)
                                
                                plotter.add_axes(line_width=5, labels_off=False)
                                plotter.view_isometric()

                                html_file = "temp_seismic_3d_slot2.html"
                                plotter.export_html(html_file)
                                plotter.close() 

                            # --- THE DISPLAY ---
                            with open(html_file, 'r', encoding='utf-8') as f:
                                source_html = f.read()
                                
                            components.html(source_html, width=800, height=600)

                    except Exception as pad_err:
                        st.error(f"Grid Padder Recovery Error: {pad_err}")
            
            # --- SLOT 3: Unreadable File ---
            elif mode == 'corrupted':
                st.error(diag_msg)
                
            # --- SLOT 4: Standard 2D (The 100% High-Res Window Chunking) ---
            elif mode == '2d':
                with segyio.open(sgy_path, "r", ignore_geometry=True, endian=endian) as f2d:
                    total_traces = f2d.tracecount
                    window_size = 2000 # The RAM Safe Zone
                    
                    if 'idx_trace_4' not in st.session_state: st.session_state.idx_trace_4 = total_traces // 2
                    
                    st.markdown("### 🎯 Trace Window Navigation (2,000 Traces per page)")
                    st.markdown("**Center Trace Sniper**")
                    step = window_size // 2 
                    c1, c2, c3, _ = st.columns([1, 3, 1, 7])
                    
                    c1.button("⏮", key="start_tr4", on_click=set_val, args=('idx_trace_4', 0), use_container_width=True)
                    c2.number_input("Target", min_value=0, max_value=total_traces-1, step=step, key='idx_trace_4', label_visibility="collapsed")
                    c3.button("⏭", key="end_tr4", on_click=set_val, args=('idx_trace_4', total_traces-1), use_container_width=True)

                    # --- NEW: COLOR & GAIN UI ---
                    ui1, ui2, _ = st.columns([3, 4, 5])
                    cmap_tr4 = ui1.selectbox("Color Palette", list(seismic_colors.keys()), key="cmap_tr4")
                    gain_tr4 = ui2.slider("Amplitude Thickness (Clip %)", min_value=50, max_value=100, value=98, step=1, key="gain_tr4")

                    # Window Extraction Logic (NO Decimation)
                    center_t = st.session_state.idx_trace_4
                    start_t = max(0, center_t - (window_size // 2))
                    end_t = min(total_traces, center_t + (window_size // 2))
                    
                    data_2d = segyio.tools.collect(f2d.trace[start_t:end_t]).T
                    
                    # Apply Dynamic Gain here!
                    vm_2d = np.percentile(np.absolute(data_2d), gain_tr4)
                    x_axis = np.arange(start_t, end_t)
                    
                    fig_2d = px.imshow(
                        data_2d, 
                        color_continuous_scale=seismic_colors[cmap_tr4], 
                        range_color=[-vm_2d, vm_2d],
                        x=x_axis, 
                        y=f2d.samples, 
                        aspect='auto', 
                        title=f"2D Seismic Section: Traces {start_t} to {end_t} (100% Resolution)",
                        labels={"x": "Trace Number", "y": "Time (ms)", "color": "Amplitude"} 
                    )
                    fig_2d.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                    fig_2d.update_traces(zsmooth='best')
                    st.plotly_chart(fig_2d, use_container_width=True, height=700)

    except Exception as e:
        st.error(f"Seismic Processing Error: {e}")