import streamlit as st
import lasio
import segyio
import numpy as np
import matplotlib.pyplot as plt
import os
import base64

# --- 1. THEME & STYLING ---
# Using the deep blue gradient and rounded containers from your design
st.set_page_config(page_title="PUDM Viewer", layout="wide")

# Function to convert local image to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# SET YOUR FILENAME HERE
img_file = "Web_Background.jpg"

try:
    bin_str = get_base64_of_bin_file('Web_Background.jpg')

    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bin_str}");
        background-attachment: fixed;
        background-size: cover;
    }}
    
    /* Dark overlay to keep text readable */
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.5); 
        z-index: -1;
    }}
    
    .info-container {{
        background-color: rgba(22, 27, 34, 0.9);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 25px;
    }}
    
    .main-title {{
        text-align: center;
        color: white;
        font-size: 42px;
        font-weight: bold;
        margin-top: 50px;
        margin-bottom: 100px;
    }}
            
    .upload-label {{
        color: white;
        text-align: center;
        font-size: 18px;
    }}
    </style>
    """, unsafe_allow_html=True)

except FileNotFoundError:
    st.warning(f"Background image '{img_file}' not found. Using default gradient.")

# --- 2. STATE MANAGEMENT ---
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None

# --- 3. PAGE LOGIC ---

# PAGE A: HOME (UPLOAD)
if st.session_state.page == 'home':
    st.markdown('<div class="main-title">PUDM WELL LOG AND SEISMIC VIEWER</div>', unsafe_allow_html=True)
    
    # Center-aligned upload box
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader("", type=['las', 'sgy'], label_visibility="collapsed")
        st.markdown('<div class="upload-label">Drag and drop files here<br>Limit 200MB per file • LAS, SGY</div>', unsafe_allow_html=True)
        
        if uploaded_file:
            ext = uploaded_file.name.split('.')[-1].lower()
            st.session_state.uploaded_data = uploaded_file
            st.session_state.page = 'well_log' if ext == 'las' else 'seismic'
            st.rerun()

# PAGE B: WELL LOG VIEW
elif st.session_state.page == 'well_log':
    # Back/Upload button at top right
    col_t1, col_t2 = st.columns([9, 1])
    if col_t2.button("☁️", help="Upload New File"):
        st.session_state.page = 'home'
        st.rerun()

    las_file = st.session_state.uploaded_data
    try:
        raw_content = las_file.read().decode("utf-8")
        las = lasio.read(raw_content)
        df = las.df().reset_index()

        # WELL LOG INFORMATION BANNER
        st.markdown('<div class="info-container">', unsafe_allow_html=True)
        st.subheader("WELL LOG INFORMATIONS")
        cols = st.columns(3)
        keys = [['STRT', 'STOP', 'STEP'], ['WELL', 'FLD', 'CTRY'], ['DATE', 'LATI', 'LONG']]
        for i, group in enumerate(keys):
            for k in group:
                val = las.well[k].value if k in las.well else "N/A"
                cols[i].write(f"**{k}:** {val}")
        st.markdown('</div>', unsafe_allow_html=True)

        # WELL LOG CURVES SECTION
        st.subheader("WELL LOG CURVES")
        depth_col = next((c for c in las.keys() if c.upper() in ['DEPT', 'DEPTH']), None)
        if depth_col:
            curves = [c for c in las.keys() if c != depth_col]
            fig, axes = plt.subplots(1, len(curves), figsize=(len(curves)*3, 10), sharey=True)
            if len(curves) == 1: axes = [axes]
            
            for i, curve in enumerate(curves):
                axes[i].plot(df[curve], df[depth_col], lw=1)
                axes[i].set_title(curve, fontweight='bold', pad=20)
                axes[i].grid(True, alpha=0.2)
                if 'RES' in curve.upper() and df[curve].min() > 0: axes[i].set_xscale('log')
            
            axes[0].set_ylabel("Depth", fontsize=12)
            plt.gca().invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Error reading LAS: {e}")

# PAGE C: SEISMIC VIEW
elif st.session_state.page == 'seismic':
    col_t1, col_t2 = st.columns([9, 1])
    if col_t2.button("☁️", help="Upload New File"):
        st.session_state.page = 'home'
        st.rerun()

    sgy_file = st.session_state.uploaded_data
    temp_path = f"temp_{sgy_file.name}"
    with open(temp_path, "wb") as f:
        f.write(sgy_file.getbuffer())

    try:
        with segyio.open(temp_path, "r", ignore_geometry=False) as f:
            # SEISMIC INFORMATION BANNER
            st.markdown('<div class="info-container">', unsafe_allow_html=True)
            st.subheader("SEISMIC INFORMATIONS")
            cols = st.columns(3)
            cols[0].write(f"**Inlines:** {len(f.ilines)}")
            cols[1].write(f"**Crosslines:** {len(f.xlines)}")
            cols[2].write(f"**Samples:** {len(f.samples)}")
            st.markdown('</div>', unsafe_allow_html=True)

            # SEISMIC PLOT SECTION (Inline + Crossline per your answer)
            st.subheader("SEISMIC PLOT")
            mid_il = f.ilines[len(f.ilines)//2]
            mid_xl = f.xlines[len(f.xlines)//2]
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 16))
            
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