import streamlit as st
import lasio
import segyio
import numpy as np
import matplotlib.pyplot as plt
import os
import base64
from obspy import read as obspy_read
from collections import defaultdict

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

# --- 2. HELPER FUNCTIONS (THE "BRAIN") ---
def detect_endianness(path):
    """Detects the correct byte order to prevent random header numbers."""
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

def detect_3d_geometry(path, endian):
    """Diagnoses the exact type and health of the seismic file using the detected endianness."""
    il_field = segyio.TraceField.INLINE_3D
    xl_field = segyio.TraceField.CROSSLINE_3D
    
    try:
        # STEP 1: Try Standard 3D
        with segyio.open(path, "r", ignore_geometry=False, endian=endian) as f:
            if len(f.ilines) > 1 and len(f.xlines) > 1:
                expected = len(f.ilines) * len(f.xlines)
                actual = f.tracecount
                if expected == actual:
                    return ('standard_3d', f.ilines.tolist(), f.xlines.tolist(), f.samples.tolist(), 
                            il_field, xl_field, "✅ Standard 3D File")
                else:
                    return ('nonstandard_3d', f.ilines.tolist(), f.xlines.tolist(), f.samples.tolist(), 
                            il_field, xl_field, f"⚠️ Irregular 3D (Missing Traces: {expected - actual})")
    except Exception:
        pass

    try:
        # STEP 2: Manual Attribute Scan for Broken 3D
        with segyio.open(path, "r", ignore_geometry=True, endian=endian) as f:
            ilines = f.attributes(il_field)[:]
            xlines = f.attributes(xl_field)[:]
            
            unique_il = np.unique(ilines)
            unique_xl = np.unique(xlines)

            if len(unique_il) > 1 and len(unique_xl) > 1:
                return ('nonstandard_3d', sorted(unique_il.tolist()), sorted(unique_xl.tolist()), 
                        f.samples.tolist(), il_field, xl_field, "❌ Broken 3D File (Grid Corrupted)")
            else:
                return ('2d', None, None, None, None, None, "📄 2D Seismic File")
    except Exception:
        return ('corrupted', None, None, None, None, None, "🚫 Heavily Corrupted (Cannot Parse)")

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
            st.subheader("Well Log Informations")
            cols = st.columns(3)
            keys = [['STRT', 'STOP', 'STEP'], ['WELL', 'FLD', 'CTRY'], ['DATE', 'LATI', 'LONG']]
            for i, group in enumerate(keys):
                for k in group:
                    val = las.well[k].value if k in las.well else "N/A"
                    cols[i].write(f"**{k}:** {val}")

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
                            if 'RES' in curve.upper() and df[curve].min() > 0:
                                axes[i].set_xscale('log')
                        
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
        # 1. RUN DIAGNOSIS
        endian = detect_endianness(sgy_path)
        mode, ilines, xlines, samples_list, det_il_field, det_xl_field, diag_msg = detect_3d_geometry(sgy_path, endian)

        # --- SEISMIC INFORMATION BANNER ---
        with st.container(key="seismic_info"):
            st.subheader("SEISMIC INFORMATIONS")
            cols = st.columns([2, 1, 1])
            cols[0].write(f"**Analysis:** {diag_msg}")
            
            if mode in ['standard_3d', 'nonstandard_3d']:
                cols[1].write(f"**Inlines:** {len(ilines)}")
                cols[2].write(f"**Crosslines:** {len(xlines)}")
            elif mode == '2d':
                with segyio.open(sgy_path, "r", ignore_geometry=True, endian=endian) as f_meta:
                    cols[1].write(f"**Traces:** {f_meta.tracecount}")
                    cols[2].write(f"**Samples:** {len(f_meta.samples)}")
            else:
                cols[1].write("**Traces:** N/A")
                cols[2].write("**Samples:** N/A")

        # --- SEISMIC PLOT SECTION (TRAFFIC CONTROLLER) ---
        with st.container(key="seismic_plots"):
            st.subheader("SEISMIC PLOT")
            fig = None
            
            # SLOT 1: Standard 3D (Fast Plotting)
            if mode == 'standard_3d':
                with segyio.open(sgy_path, "r", ignore_geometry=False, endian=endian) as f3d:
                    mid_il = f3d.ilines[len(f3d.ilines)//2]
                    mid_xl = f3d.xlines[len(f3d.xlines)//2]
                    
                    fig, axes = plt.subplots(2, 1, figsize=(16, 20))
                    
                    data_il = f3d.iline[mid_il].T
                    vm_il = np.percentile(np.absolute(data_il), 98)
                    axes[0].imshow(data_il, cmap='RdBu', aspect='auto', vmin=-vm_il, vmax=vm_il,
                                   extent=[f3d.xlines[0], f3d.xlines[-1], f3d.samples[-1], f3d.samples[0]])
                    axes[0].set_title(f"3D Inline: {mid_il}", color='white', fontweight='bold')  
                    axes[0].tick_params(axis='both', colors='white') 
                    axes[0].set_xlabel("Crosslines", color='white')

                    data_xl = f3d.xline[mid_xl].T
                    vm_xl = np.percentile(np.absolute(data_xl), 98)
                    axes[1].imshow(data_xl, cmap='RdBu', aspect='auto', vmin=-vm_xl, vmax=vm_xl,
                                   extent=[f3d.ilines[0], f3d.ilines[-1], f3d.samples[-1], f3d.samples[0]])
                    axes[1].set_title(f"3D Crossline: {mid_xl}", color='white', fontweight='bold')
                    axes[1].tick_params(axis='both', colors='white')
                    axes[1].set_xlabel("Inlines", color='white')

            # SLOT 2: Broken/Irregular 3D (The ObsPy Surgeon)
            elif mode == 'nonstandard_3d':
                with st.spinner(f"🛠️ {diag_msg}... Reconstructing with ObsPy"):
                    try:
                        # Pass the detected endianness to ObsPy ('>' is big, '<' is little)
                        obspy_endian = '>' if endian == 'big' else '<'
                        stream = obspy_read(sgy_path, format='SEGY', headertypestr=obspy_endian)
                        
                        il_groups = defaultdict(list)
                        xl_groups = defaultdict(list)

                        # NEW DICTIONARY (Official ObsPy header names)
                        segyio_to_obspy = {
                            int(segyio.TraceField.INLINE_3D): 'for_3d_poststack_data_this_field_is_for_in_line_number',
                            int(segyio.TraceField.CROSSLINE_3D): 'for_3d_poststack_data_this_field_is_for_cross_line_number'
                        }
                        il_key = segyio_to_obspy.get(int(det_il_field), 'inline_3d')
                        xl_key = segyio_to_obspy.get(int(det_xl_field), 'crossline_3d')

                        for tr in stream:
                            th = tr.stats.segy.trace_header
                            il_val = getattr(th, il_key, None)
                            xl_val = getattr(th, xl_key, None)
                            
                            if il_val is not None and xl_val is not None:
                                il_groups[il_val].append(tr.data.astype(np.float32))
                                xl_groups[xl_val].append(tr.data.astype(np.float32))

                        mid_il = ilines[len(ilines)//2]
                        mid_xl = xlines[len(xlines)//2]

                        traces_il = il_groups.get(mid_il, [])
                        traces_xl = xl_groups.get(mid_xl, [])

                        if traces_il or traces_xl:
                            nplots = (1 if traces_il else 0) + (1 if traces_xl else 0)
                            fig, axes = plt.subplots(nplots, 1, figsize=(16, 10 * nplots))
                            if nplots == 1: axes = [axes]
                            
                            idx = 0
                            if traces_il:
                                data_il = np.array(traces_il).T
                                vm_il = np.percentile(np.absolute(data_il), 98)
                                axes[idx].imshow(data_il, cmap='RdBu', aspect='auto', vmin=-vm_il, vmax=vm_il)
                                axes[idx].set_title(f"Recovered 3D Inline: {mid_il}", color='white', fontweight='bold')
                                axes[idx].tick_params(colors='white')
                                idx += 1
                                
                            if traces_xl:
                                data_xl = np.array(traces_xl).T
                                vm_xl = np.percentile(np.absolute(data_xl), 98)
                                axes[idx].imshow(data_xl, cmap='RdBu', aspect='auto', vmin=-vm_xl, vmax=vm_xl)
                                axes[idx].set_title(f"Recovered 3D Crossline: {mid_xl}", color='white', fontweight='bold')
                                axes[idx].tick_params(colors='white')
                        else:
                            st.error("Recovery Failed: Could not group traces into a 3D grid.")

                    except Exception as obspy_err:
                        st.error(f"ObsPy Recovery Error: {obspy_err}")
            
            # SLOT 3: Unreadable File
            elif mode == 'corrupted':
                st.error(diag_msg)
                
            # SLOT 4: Standard 2D
            elif mode == '2d':
                with segyio.open(sgy_path, "r", ignore_geometry=True, endian=endian) as f2d:
                    data_2d = segyio.tools.collect(f2d.trace[:]).T
                    vm_2d = np.percentile(np.absolute(data_2d), 98)
                    
                    fig, ax = plt.subplots(figsize=(16, 10))
                    ax.imshow(data_2d, cmap='RdBu', aspect='auto', vmin=-vm_2d, vmax=vm_2d)
                    ax.set_title("2D Seismic Section", color='white', fontweight='bold')
                    ax.set_xlabel("Trace Number", color='white')
                    ax.set_ylabel("Sample Index", color='white')
                    ax.tick_params(colors='white')

            if fig:
                fig.patch.set_alpha(0)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    except Exception as e:
        st.error(f"Seismic Processing Error: {e}")