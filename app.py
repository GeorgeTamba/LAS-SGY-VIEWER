import streamlit as st
import lasio
import segyio
import numpy as np
import matplotlib.pyplot as plt
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
    
    # We apply the background to the App and the Glass effect to your Keys
    css = f"""
    .stApp {{
        background-image: url("data:image/jpg;base64,{bin_str}");
        background-attachment: fixed;
        background-size: cover;
    }}
    
    /* Global Overlay for readability */
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background-color: rgba(0, 0, 0, 0.4);
        z-index: -1;
    }}

    /* STABLE GLASSMOPRHISM via Keys */
    [class*="st-key-las_"], [class*="st-key-seismic_"] {{
        background-color: rgba(255, 255, 255, 0.12) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 20px !important;
        
        /* The Frosting */
        -webkit-backdrop-filter: blur(20px) saturate(160%) !important;
        backdrop-filter: blur(20px) saturate(160%) !important;
        
        padding: 2rem !important;
        margin-bottom: 2rem !important;
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

# --- 2. STATE MANAGEMENT ---
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None

# --- 3. PAGE LOGIC ---

# PAGE A: HOME
if st.session_state.page == 'home':
    st.markdown('<div class="main-title">PUDM WELL LOG AND SEISMIC VIEWER</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader("", type=['las', 'sgy'], label_visibility="collapsed")
        st.markdown('<div class="upload-label">Drag and drop files here<br>LAS, SGY</div>', unsafe_allow_html=True)
        if uploaded_file:
            ext = uploaded_file.name.split('.')[-1].lower()
            st.session_state.uploaded_data = uploaded_file
            st.session_state.page = 'well_log' if ext == 'las' else 'seismic'
            st.rerun()

# PAGE B: WELL LOG VIEW
elif st.session_state.page == 'well_log':
    col_t1, col_t2 = st.columns([12, 1])
    if col_t2.button("☁️"):
        st.session_state.page = 'home'
        st.rerun()

    las_file = st.session_state.uploaded_data
    try:
        raw_content = las_file.read().decode("utf-8")
        las = lasio.read(raw_content)
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
                
                # Calculate how many rows we need
                num_rows = (num_curves + cols_per_row - 1) // cols_per_row
                depth_unit = las.curves[depth_col].unit if depth_col in las.curves else "m"

                for r in range(num_rows):
                    start_idx = r * cols_per_row
                    end_idx = min(start_idx + cols_per_row, num_curves)
                    row_curves = available_curves[start_idx:end_idx]
                    
                    # 1. THE STRETCH-FIX: Constrain the width using Streamlit columns
                    # This ensures the first column only takes up the space needed for the plots
                    display_cols = st.columns([len(row_curves), cols_per_row - len(row_curves) + 0.1])
                    
                    with display_cols[0]:
                        # 2. Adjust figsize proportionally
                        row_width = len(row_curves) * 3.5 
                        fig, axes = plt.subplots(1, len(row_curves), figsize=(row_width, 10), sharey=True)
                        
                        if len(row_curves) == 1: 
                            axes = [axes]

                        for i, curve in enumerate(row_curves):
                            # Using your preferred Cyan color from the revision
                            axes[i].plot(df[curve], df[depth_col], color="#0028B6", lw=1.5)
                            axes[i].set_title(curve, fontweight='bold', color='white', pad=15)
                            axes[i].grid(True, linestyle='-', alpha=0.5, color='#000000')
                            
                            if 'RES' in curve.upper() and df[curve].min() > 0:
                                axes[i].set_xscale('log')
                        
                        # Style axis labels and ticks for dark theme
                        axes[0].set_ylabel(f"Depth ({depth_unit})", color='white', fontweight='bold')
                        for ax in axes:
                            ax.tick_params(colors='white')
                            ax.set_facecolor('white') # Transparent axes
                        
                        plt.gca().invert_yaxis()
                        fig.patch.set_alpha(0) # Transparent figure background
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

    sgy_file = st.session_state.uploaded_data
    temp_path = f"temp_{sgy_file.name}"
    with open(temp_path, "wb") as f: f.write(sgy_file.getbuffer())

    try:
        with segyio.open(temp_path, "r", ignore_geometry=False) as f:
            # SEISMIC INFORMATION BANNER
            with st.container(key="seismic_info"):
                st.subheader("SEISMIC INFORMATIONS")
                cols = st.columns(3)
                cols[0].write(f"**Inlines:** {len(f.ilines)}")
                cols[1].write(f"**Crosslines:** {len(f.xlines)}")
                cols[2].write(f"**Samples:** {len(f.samples)}")

                # SEISMIC PLOT SECTION (Inline + Crossline per your answer)
            with st.container(key="seismic_plots"):
                st.subheader("SEISMIC PLOT")
                mid_il = f.ilines[len(f.ilines)//2]
                mid_xl = f.xlines[len(f.xlines)//2]
                
                fig, axes = plt.subplots(2, 1, figsize=(16, 20))
                
                # Inline Plot
                data_il = f.iline[mid_il].T
                vm_il = np.percentile(np.absolute(data_il), 98)
                axes[0].imshow(data_il, cmap='RdBu', aspect='auto', vmin=-vm_il, vmax=vm_il,
                            extent=[f.xlines[0], f.xlines[-1], f.samples[-1], f.samples[0]])
                axes[0].set_title(f"Seismic Inline: {mid_il}")

                # Crossline Plot
                data_xl = f.xline[mid_xl].T
                vm_xl = np.percentile(np.absolute(data_xl), 98)
                axes[1].imshow(data_xl, cmap='RdBu', aspect='auto', vmin=-vm_xl, vmax=vm_xl,
                            extent=[f.ilines[0], f.ilines[-1], f.samples[-1], f.samples[0]])
                axes[1].set_title(f"Seismic Crossline: {mid_xl}")
                
                plt.tight_layout()
                st.pyplot(fig)
    except Exception as e:
        st.error(f"Error reading SGY: {e}")
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)