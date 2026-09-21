import io
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from openpyxl.drawing.image import Image as OpenPyxlImage
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import streamlit as st

# Import vamas parser library with fallback handling
try:
    import vamas
    HAS_VAMAS = True
except ImportError:
    HAS_VAMAS = False

st.set_page_config(page_title="XPS & VAMAS Peak Fitting", layout="wide")
st.title("🔬 XPS Peak Fitting & Elemental Analysis")

# --- Reference Binding Energies (eV) ---
XPS_DATABASE = {
    "C 1s": 284.8,
    "C-O": 286.5,
    "C=O": 288.0,
    "O 1s": 531.0,
    "O-Metal": 529.8,
    "Cu 2p3/2": 932.6,
    "Cu 2p1/2": 952.4,
    "Cu(II) Satellite": 942.0,
    "Zr 3d5/2": 182.2,
    "Zr 3d3/2": 184.6,
    "Zn 2p3/2": 1021.8,
    "Zn 2p1/2": 1044.9,
    "N 1s": 399.8,
    "Si 2p": 99.3,
    "SiO2": 103.3,
    "Al 2p": 74.0,
}

# --- Fitting Functions ---
def gaussian(x, amp, center, sigma):
    return amp * np.exp(-(((x - center) ** 2) / (2 * sigma**2)))

def multi_gaussian(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        y += gaussian(x, params[i], params[i+1], params[i+2])
    return y

# --- Sidebar Controls ---
st.sidebar.header("1. Upload XPS File")
uploaded_file = st.sidebar.file_uploader(
    "Upload XPS Data (.vcm, .vms, .csv, .xlsx, .txt)", 
    type=["vcm", "vms", "csv", "xlsx", "xls", "txt"]
)

st.sidebar.header("2. Peak Identification")
selected_elements = st.sidebar.multiselect(
    "Select Target Core Levels",
    options=list(XPS_DATABASE.keys()),
    default=["C 1s", "O 1s", "Cu 2p3/2", "Zr 3d5/2"]
)
energy_tol = st.sidebar.slider("Binding Energy Tolerance (± eV)", 0.5, 3.0, 1.5, 0.1)

st.sidebar.header("3. Fitting Settings")
num_peaks = st.sidebar.number_input("Number of Gaussian Peaks to Fit", min_value=1, max_value=6, value=2)

# --- Data Parsing Logic ---
if uploaded_file is not None:
    try:
        file_ext = uploaded_file.name.split(".")[-1].lower()
        df = None

        # Parse VAMAS (.vcm / .vms) Files
        if file_ext in ["vcm", "vms"]:
            if not HAS_VAMAS:
                st.error("The `vamas` library is missing. Please add `vamas` to your requirements.txt.")
            else:
                # Read raw bytes into VAMAS parser
                bytes_data = uploaded_file.read()
                vms_obj = vamas.Vamas(bytes_data)
                
                # Extract first block spectrum
                block = vms_obj.blocks[0]
                x_vals = block.abscissa
                y_vals = block.corresponding_variables[0].array
                
                df = pd.DataFrame({"BE": x_vals, "Intensity": y_vals})

        # Parse Excel (.xlsx, .xls) Files
        elif file_ext in ["xlsx", "xls"]:
            raw_df = pd.read_excel(uploaded_file)
            col_be = st.selectbox("Select Binding Energy Column", raw_df.columns, index=0)
            col_int = st.selectbox("Select Intensity Column", raw_df.columns, index=min(1, len(raw_df.columns)-1))
            df = raw_df[[col_be, col_int]].dropna().rename(columns={col_be: "BE", col_int: "Intensity"})

        # Parse Text / CSV Files
        else:
            delimiter = st.sidebar.selectbox("Delimiter", [r"\s+", ",", ";", r"\t"], index=0)
            raw_df = pd.read_csv(uploaded_file, delimiter=delimiter, engine="python")
            col_be = st.selectbox("Select Binding Energy Column", raw_df.columns, index=0)
            col_int = st.selectbox("Select Intensity Column", raw_df.columns, index=min(1, len(raw_df.columns)-1))
            df = raw_df[[col_be, col_int]].dropna().rename(columns={col_be: "BE", col_int: "Intensity"})

        if df is not None and not df.empty:
            # Clean and Sort Data
            df["BE"] = pd.to_numeric(df["BE"])
            df["Intensity"] = pd.to_numeric(df["Intensity"])
            df = df.sort_values(by="BE", ascending=True).reset_index(drop=True)

            # Linear Baseline Subtraction
            baseline = np.linspace(df["Intensity"].iloc[0], df["Intensity"].iloc[-1], len(df))
            df["Corrected_Intensity"] = np.maximum(0, df["Intensity"] - baseline)

            x_data = df["BE"].values
            y_data = df["Corrected_Intensity"].values

            # Peak Auto-Detection
            prominence = st.sidebar.slider("Peak Sensitivity", 0.01, 1.0, 0.1)
            peak_indices, _ = find_peaks(y_data, prominence=prominence * np.max(y_data))
            detected_be = x_data[peak_indices]

            # Match against database
            identifications = []
            for be in detected_be:
                matched = "Unassigned"
                for label in selected_elements:
                    ref_be = XPS_DATABASE[label]
                    if abs(be - ref_be) <= energy_tol:
                        matched = f"{label} (~{ref_be} eV)"
                        break
                identifications.append({"Detected BE (eV)": round(be, 2), "Matched Core Level": matched})

            # Interactive Peak Fitting Inputs
            st.subheader("Gaussian Fit Parameters")
            initial_guesses, bounds_lower, bounds_upper = [], [], []
            fit_cols = st.columns(int(num_peaks))

            for idx in range(int(num_peaks)):
                with fit_cols[idx]:
                    st.markdown(f"**Peak {idx+1}**")
                    def_center = float(detected_be[idx]) if idx < len(detected_be) else float(np.mean(x_data))
                    center = st.number_input(f"Center BE (eV) #{idx+1}", value=round(def_center, 2))
                    amp = st.number_input(f"Amplitude #{idx+1}", value=float(np.max(y_data) / num_peaks))
                    sigma = st.number_input(f"Sigma #{idx+1}", value=1.0, min_value=0.1, step=0.1)

                    initial_guesses.extend([amp, center, sigma])
                    bounds_lower.extend([0, center - 5.0, 0.1])
                    bounds_upper.extend([np.max(y_data) * 2, center + 5.0, 10.0])

            # Perform Optimization
            popt, _ = curve_fit(multi_gaussian, x_data, y_data, p0=initial_guesses, bounds=(bounds_lower, bounds_upper))
            fit_y = multi_gaussian(x_data, *popt)

            # Plot XPS Profile
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(x_data, df["Intensity"], "k.", alpha=0.3, label="Raw Spectrum")
            ax.plot(x_data, fit_y + baseline, "r-", linewidth=2, label="Envelope Fit")
            ax.plot(x_data, baseline, "k--", label="Linear Baseline")

            peak_summary = []
            total_area = 0

            for i in range(0, len(popt), 3):
                amp, center, sigma = popt[i], popt[i+1], popt[i+2]
                component_y = gaussian(x_data, amp, center, sigma)
                area = np.trapezoid(component_y, x_data)
                total_area += area

                ax.plot(x_data, component_y + baseline, label=f"Peak @ {center:.2f} eV")
                ax.fill_between(x_data, baseline, component_y + baseline, alpha=0.2)

                peak_summary.append({
                    "Peak #": (i // 3) + 1,
                    "Center (eV)": round(center, 2),
                    "FWHM (eV)": round(2.35482 * sigma, 2),
                    "Area (a.u.·eV)": round(area, 2),
                })

            for p in peak_summary:
                p["Area (%)"] = round((p["Area (a.u.·eV)"] / total_area) * 100, 2) if total_area > 0 else 0

            ax.set_xlabel("Binding Energy (eV)")
            ax.set_ylabel("Intensity (a.u.)")
            ax.set_title(f"XPS Profile - {uploaded_file.name}")
            ax.invert_xaxis()  # XPS high BE to low BE convention
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper left")
            st.pyplot(fig)

            # Data Tables
            col_left, col_right = st.columns(2)
            with col_left:
                st.subheader("Identified Core Levels")
                st.dataframe(pd.DataFrame(identifications), use_container_width=True)

            with col_right:
                st.subheader("Peak Deconvolution Shares")
                df_peaks = pd.DataFrame(peak_summary)
                st.dataframe(df_peaks, use_container_width=True)

            # Excel Export with Plot Image Embedded
            plot_img_bytes = io.BytesIO()
            fig.savefig(plot_img_bytes, format="png", dpi=200, bbox_inches="tight")
            plot_img_bytes.seek(0)

            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                df_peaks.to_excel(writer, sheet_name="Peak Quantification", index=False)
                pd.DataFrame(identifications).to_excel(writer, sheet_name="Identifications", index=False)
                df.to_excel(writer, sheet_name="Raw Data", index=False)

                ws = writer.sheets["Peak Quantification"]
                img = OpenPyxlImage(plot_img_bytes)
                img.anchor = "F2"
                ws.add_image(img)

            st.download_button(
                label="📥 Download Full XPS Analysis Report (.xlsx)",
                data=excel_buffer.getvalue(),
                file_name=f"XPS_Report_{uploaded_file.name.rsplit('.', 1)[0]}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    except Exception as e:
        st.error(f"Error executing analysis: {e}")
else:
    st.info("Upload a `.vcm`, `.vms`, `.csv`, or `.xlsx` file to begin.")
