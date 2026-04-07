import streamlit as st
import lasio
import segyio
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
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
                    for item in las.version: st.markdown(f"**{item.mnemonic}:** {item.value} <span style='color:gray; font-size:14px'><i>({item.descr})</i></span>", unsafe_allow_html=True)
            if las.well:
                with st.expander("Well Information (~W)", expanded=True):
                    for item in las.well: st.markdown(f"**{item.mnemonic}:** {item.value} <span style='color:gray; font-size:14px'><i>({item.descr})</i></span>", unsafe_allow_html=True)
            if las.params:
                with st.expander("Parameter Information (~P)"):
                    for item in las.params: st.markdown(f"**{item.mnemonic}:** {item.value} <span style='color:gray; font-size:14px'><i>({item.descr})</i></span>", unsafe_allow_html=True)
            if las.other:
                with st.expander("Other Information (~O)"): st.text(las.other)

        with st.container(key="las_curves"):
            st.subheader("Well Log Curves")
            depth_col = next((c for c in las.keys() if c.upper() in ['DEPT', 'DEPTH']), None)
            if depth_col:
                available_curves = [c for c in las.keys() if c != depth_col]
                num_curves = len(available_curves)
                cols_per_row = 5
                num_rows = (num_curves + cols_per_row - 1) // cols_per_row
                depth_unit = las.curves[depth_col].unit if depth_col in las.curves else "m"

                for r in range(num_rows):
                    start_idx = r * cols_per_row
                    end_idx = min(start_idx + cols_per_row, num_curves)
                    row_curves = available_curves[start_idx:end_idx]
                    display_cols = st.columns([len(row_curves), cols_per_row - len(row_curves) + 0.1])
                    with display_cols[0]:
                        row_width = len(row_curves) * 3.5 
                        fig, axes = plt.subplots(1, len(row_curves), figsize=(row_width, 10), sharey=True)
                        if len(row_curves) == 1: axes = [axes]
                        for i, curve in enumerate(row_curves):
                            axes[i].plot(df[curve], df[depth_col], color="#0028B6", lw=1.5)
                            axes[i].set_title(curve, fontweight='bold', color='white', pad=15)
                            axes[i].grid(True, linestyle='-', alpha=0.5, color='#000000')
                            if 'RES' in curve.upper() and df[curve].min() > 0: axes[i].set_xscale('log')
                        axes[0].set_ylabel(f"Depth ({depth_unit})", color='white', fontweight='bold')
                        for ax in axes:
                            ax.tick_params(colors='white')
                            ax.set_facecolor('white') 
                        plt.gca().invert_yaxis()
                        fig.patch.set_alpha(0) 
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
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
                            st.table(scan_data)

        with st.container(key="seismic_plots"):
            st.subheader("SEISMIC PLOT")
            
            # --- SLOT 1: Standard 3D ---
            if mode == 'standard_3d':
                with segyio.open(sgy_path, "r", ignore_geometry=False, endian=endian) as f3d:
                    if 'idx_il_1' not in st.session_state: st.session_state.idx_il_1 = len(ilines) // 2
                    if 'idx_xl_1' not in st.session_state: st.session_state.idx_xl_1 = len(xlines) // 2
                    
                    # --- INLINE SECTION ---
                    st.markdown("### 🎯 Inline Target")
                    # Using a 7-weight empty column to push everything left
                    c1, c2, c3, _ = st.columns([1, 3, 1, 7])
                    c1.button("⏮", key="start_il1", on_click=set_val, args=('idx_il_1', 0), use_container_width=True)
                    c2.number_input("IL", min_value=0, max_value=len(ilines)-1, key='idx_il_1', label_visibility="collapsed")
                    c3.button("⏭", key="end_il1", on_click=set_val, args=('idx_il_1', len(ilines)-1), use_container_width=True)

                    mid_il = ilines[st.session_state.idx_il_1]
                    data_il = f3d.iline[mid_il].T
                    vm_il = np.percentile(np.absolute(data_il), 98)
                    fig_il = px.imshow(data_il, color_continuous_scale='RdBu', range_color=[-vm_il, vm_il],
                                       x=xlines, y=samples_list, aspect='auto', title=f"3D Inline: {mid_il}",
                                       labels={"x": "Crossline", "y": "Time (ms)", "color": "Amplitude"})
                    fig_il.update_layout(plot_bgcolor='#E0E0E0', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                    fig_il.update_traces(zsmooth='best')
                    st.plotly_chart(fig_il, use_container_width=True, height=700)

                    st.markdown("---") # Visual Separator

                    # --- CROSSLINE SECTION ---
                    st.markdown("### 🎯 Crossline Target")
                    c1, c2, c3, _ = st.columns([1, 3, 1, 7])
                    c1.button("⏮", key="start_xl1", on_click=set_val, args=('idx_xl_1', 0), use_container_width=True)
                    c2.number_input("XL", min_value=0, max_value=len(xlines)-1, key='idx_xl_1', label_visibility="collapsed")
                    c3.button("⏭", key="end_xl1", on_click=set_val, args=('idx_xl_1', len(xlines)-1), use_container_width=True)

                    mid_xl = xlines[st.session_state.idx_xl_1]
                    data_xl = f3d.xline[mid_xl].T
                    vm_xl = np.percentile(np.absolute(data_xl), 98)
                    fig_xl = px.imshow(data_xl, color_continuous_scale='RdBu', range_color=[-vm_xl, vm_xl],
                                       x=ilines, y=samples_list, aspect='auto', title=f"3D Crossline: {mid_xl}",
                                       labels={"x": "Inline", "y": "Time (ms)", "color": "Amplitude"})
                    fig_xl.update_layout(plot_bgcolor='#E0E0E0', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                    fig_xl.update_traces(zsmooth='best')
                    st.plotly_chart(fig_xl, use_container_width=True, height=700)

            # --- SLOT 2: Nonstandard 3D (Enterprise Grid Padder) ---
            elif mode == 'nonstandard_3d':
                with st.spinner(f"🛠️ Reconstructing Polygon Grid with Native Padding..."):
                    try:
                        with segyio.open(sgy_path, "r", ignore_geometry=True, endian=endian) as f:
                            if 'idx_il_2' not in st.session_state: st.session_state.idx_il_2 = len(ilines) // 2
                            if 'idx_xl_2' not in st.session_state: st.session_state.idx_xl_2 = len(xlines) // 2
                            
                            all_il, all_xl = get_polygon_headers(sgy_path, endian, det_il_field, det_xl_field)
                            
                            xl_to_idx = {val: i for i, val in enumerate(xlines)}
                            il_to_idx = {val: i for i, val in enumerate(ilines)}

                            # --- INLINE SECTION ---
                            st.markdown("### 🎯 Recovered Inline Target")
                            c1, c2, c3, _ = st.columns([1, 3, 1, 7])
                            c1.button("⏮", key="start_il2", on_click=set_val, args=('idx_il_2', 0), use_container_width=True)
                            c2.number_input("IL", min_value=0, max_value=len(ilines)-1, key='idx_il_2', label_visibility="collapsed")
                            c3.button("⏭", key="end_il2", on_click=set_val, args=('idx_il_2', len(ilines)-1), use_container_width=True)

                            mid_il = ilines[st.session_state.idx_il_2]
                            tr_idx_il = np.where(all_il == mid_il)[0]
                            data_il = np.full((len(samples_list), len(xlines)), np.nan)
                            for idx in tr_idx_il:
                                xl_val = all_xl[idx]
                                if xl_val in xl_to_idx:
                                    data_il[:, xl_to_idx[xl_val]] = f.trace[idx]

                            if len(tr_idx_il) > 0:
                                vm_il = np.nanpercentile(np.absolute(data_il), 98) 
                                fig_il = px.imshow(data_il, color_continuous_scale='RdBu', range_color=[-vm_il, vm_il],
                                                   x=xlines, y=samples_list, aspect='auto', title=f"Reconstructed 3D Inline: {mid_il}",
                                                   labels={"x": "Inline", "y": "Time (ms)", "color": "Amplitude"})
                                fig_il.update_layout(plot_bgcolor='#E0E0E0', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                                fig_il.update_traces(zsmooth='best')
                                st.plotly_chart(fig_il, use_container_width=True, height=700)

                            st.markdown("---") # Visual Separator

                            # --- CROSSLINE SECTION ---
                            st.markdown("### 🎯 Recovered Crossline Target")
                            c1, c2, c3, _ = st.columns([1, 3, 1, 7])
                            c1.button("⏮", key="start_xl2", on_click=set_val, args=('idx_xl_2', 0), use_container_width=True)
                            c2.number_input("XL", min_value=0, max_value=len(xlines)-1, key='idx_xl_2', label_visibility="collapsed")
                            c3.button("⏭", key="end_xl2", on_click=set_val, args=('idx_xl_2', len(xlines)-1), use_container_width=True)

                            mid_xl = xlines[st.session_state.idx_xl_2]
                            tr_idx_xl = np.where(all_xl == mid_xl)[0]
                            data_xl = np.full((len(samples_list), len(ilines)), np.nan)
                            for idx in tr_idx_xl:
                                il_val = all_il[idx]
                                if il_val in il_to_idx:
                                    data_xl[:, il_to_idx[il_val]] = f.trace[idx]
                                    
                            if len(tr_idx_xl) > 0:
                                vm_xl = np.nanpercentile(np.absolute(data_xl), 98)
                                fig_xl = px.imshow(data_xl, color_continuous_scale='RdBu', range_color=[-vm_xl, vm_xl],
                                                   x=ilines, y=samples_list, aspect='auto', title=f"Reconstructed 3D Crossline: {mid_xl}",
                                                   labels={"x": "Inline", "y": "Time (ms)", "color": "Amplitude"})
                                fig_xl.update_layout(plot_bgcolor='#E0E0E0', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                                fig_xl.update_traces(zsmooth='best')
                                st.plotly_chart(fig_xl, use_container_width=True, height=700)

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

                    # Window Extraction Logic (NO Decimation)
                    center_t = st.session_state.idx_trace_4
                    start_t = max(0, center_t - (window_size // 2))
                    end_t = min(total_traces, center_t + (window_size // 2))
                    
                    data_2d = segyio.tools.collect(f2d.trace[start_t:end_t]).T
                    vm_2d = np.percentile(np.absolute(data_2d), 98)
                    x_axis = np.arange(start_t, end_t)
                    
                    fig_2d = px.imshow(
                        data_2d, 
                        color_continuous_scale='RdBu', 
                        range_color=[-vm_2d, vm_2d],
                        x=x_axis, 
                        y=f2d.samples, 
                        aspect='auto', 
                        title=f"2D Seismic Section: Traces {start_t} to {end_t} (100% Resolution)",
                        labels={"x": "Trace Number", "y": "Time (ms)", "color": "Amplitude"} # <-- ADDED LABELS HERE
                    )
                    fig_2d.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                    fig_2d.update_traces(zsmooth='best')
                    st.plotly_chart(fig_2d, use_container_width=True, height=700)

    except Exception as e:
        st.error(f"Seismic Processing Error: {e}")