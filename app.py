import streamlit as st
import lasio
import segyio
import numpy as np
import matplotlib.pyplot as plt
import os

# --- SEISMIC PROCESSING LOGIC ---
class PHESeismicModel:
    def __init__(self, data):
        self.data = data
        self.vm = np.percentile(np.absolute(data), 98)

# --- STREAMLIT WEB INTERFACE ---
st.set_page_config(page_title="PUDM Web Viewer", layout="wide")

# --- HEADER / UPLOAD SECTION ---
st.title("🛢️ PUDM Data Management Dashboard")
uploaded_file = st.file_uploader("UPLOAD FILE", type=['las', 'sgy'])

if uploaded_file:
    file_ext = uploaded_file.name.split('.')[-1].lower()

    # Create the horizontal layout format
    # Sidebar will act as the "Header Information" container
    with st.sidebar:
        st.header("📋 HEADER INFORMATION")
        st.info(f"File: {uploaded_file.name}")

    # --- 1. LAS FILE PROCESSING ---
    if file_ext == 'las':
        try:
            raw_content = uploaded_file.read().decode("utf-8")
            las = lasio.read(raw_content) 
            df = las.df().reset_index()

            # Populate Sidebar with Header Info
            with st.sidebar:
                header_keys = ['STRT', 'STOP', 'STEP', 'WELL', 'FLD', 'CTRY', 'DATE']
                for key in header_keys:
                    val = las.well[key].value if key in las.well else "N/A"
                    st.write(f"**{key}:** {val}")
            
            # Main View for Curves
            st.subheader("📈 CURVE INFORMATION")
            depth_col = next((c for c in las.keys() if c.upper() in ['DEPT', 'DEPTH']), None)
            
            if depth_col:
                available_curves = [c for c in las.keys() if c != depth_col]
                num_curves = len(available_curves)
                fig, axes = plt.subplots(1, num_curves, figsize=(num_curves * 2.5, 8), sharey=True)
                if num_curves == 1: axes = [axes]

                for i, curve in enumerate(available_curves):
                    axes[i].plot(df[curve], df[depth_col], lw=0.8)
                    axes[i].set_title(curve, fontweight='bold')
                    axes[i].grid(True, alpha=0.3)
                    if 'RES' in curve.upper() and df[curve].min() > 0:
                        axes[i].set_xscale('log')

                axes[0].set_ylabel(f"Depth", fontsize=10)
                plt.gca().invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.error("No Depth column found.")
        except Exception as e:
            st.error(f"LAS Error: {e}")

    # --- 2. SEISMIC FILE PROCESSING ---
    elif file_ext == 'sgy':
        temp_filename = f"temp_{uploaded_file.name}"
        with open(temp_filename, "wb") as f_temp:
            f_temp.write(uploaded_file.getbuffer())

        try:
            # STRATEGY 1: GEOMETRY-AWARE 3D
            try:
                with segyio.open(temp_filename, "r", ignore_geometry=False) as f:
                    inlines = f.ilines
                    crosslines = f.xlines
                    t_samples = f.samples
                    
                    # Add Seismic Header Info to Sidebar
                    with st.sidebar:
                        st.write(f"**Inlines:** {len(inlines)}")
                        st.write(f"**Crosslines:** {len(crosslines)}")
                        st.write(f"**Samples:** {len(t_samples)}")
                        st.write(f"**Interval:** {f.bin[segyio.BinField.Interval]} µs")
                    
                    mid_il_idx = len(inlines) // 2
                    mid_xl_idx = len(crosslines) // 2
                    inline_data = f.iline[inlines[mid_il_idx]].T
                    xline_data = f.xline[crosslines[mid_xl_idx]].T
                    
                    st.subheader("🖼️ SEISMIC VISUALIZATION")
                    
                    # Professional 3D Layout
                    fig, axes = plt.subplots(2, 1, figsize=(12, 14))
                    
                    vm_il = np.percentile(np.absolute(inline_data), 98)
                    im1 = axes[0].imshow(inline_data, cmap='RdBu', aspect='auto', vmin=-vm_il, vmax=vm_il,
                                       extent=[crosslines[0], crosslines[-1], t_samples[-1], t_samples[0]])
                    axes[0].set_title(f"Seismic Inline: {inlines[mid_il_idx]}")
                    fig.colorbar(im1, ax=axes[0])

                    vm_xl = np.percentile(np.absolute(xline_data), 98)
                    im2 = axes[1].imshow(xline_data, cmap='RdBu', aspect='auto', vmin=-vm_xl, vmax=vm_xl,
                                       extent=[inlines[0], inlines[-1], t_samples[-1], t_samples[0]])
                    axes[1].set_title(f"Seismic Crossline: {crosslines[mid_xl_idx]}")
                    fig.colorbar(im2, ax=axes[1])

                    plt.tight_layout()
                    st.pyplot(fig)

            except Exception:
                st.sidebar.error("❌ FILE IS EITHER 2D OR DEFECT 3D")
                st.error("Visualization failed for this geometry.")

        except Exception as e:
            st.error(f"Seismic Error: {e}")
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
else:
    # Format before upload
    col_left, col_right = st.columns([1, 3])
    with col_left:
        st.container(border=True).write("HEADER INFORMATION (Will appear after upload)")
    with col_right:
        st.container(border=True).write("CURVE INFORMATION & SEISMIC VISUALIZATION (Will appear after upload)")

        