import io
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from openpyxl.drawing.image import Image as OpenPyxlImage
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import streamlit as st

st.set_page_config(
    page_title="XPS Peak Fitting & Analysis", layout="wide"
)
st.title("🔬 XPS Peak Fitting & Elemental Analysis")

# --- Known XPS Reference Binding Energies (eV) ---
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

# --- Helper Functions ---
def gaussian(x, amp, center, sigma):
    return amp * np.exp(-(((x - center) ** 2) / (2 * sigma**2)))


def multi_gaussian(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        amp = params[i]
        center = params[i + 1]
        sigma = params[i + 2]
        y += gaussian(x, amp, center, sigma)
    return y


# --- Sidebar Setup ---
st.sidebar.header("1. Upload & Setup")
uploaded_file = st.sidebar.file_uploader(
    "Upload XPS Data (.csv, .xlsx, .txt)", type=["csv", "xlsx", "xls", "txt"]
)

st.sidebar.header("2. Peak Auto-Identification")
selected_elements = st.sidebar.multiselect(
    "Select Elements / Core Levels",
    options=list(XPS_DATABASE.keys()),
    default=["C 1s", "O 1s", "Cu 2p3/2", "Zr 3d5/2"],
)

energy_tol = st.sidebar.slider(
    "Binding Energy Tolerance (± eV)", 0.5, 3.0, 1.5, 0.1
)

st.sidebar.header("3. Fitting Controls")
num_peaks = st.sidebar.number_input(
    "Number of Gaussian Peaks to Fit", min_value=1, max_value=6, value=2
)

# --- Main Logic ---
if uploaded_file is not None:
    try:
        file_ext = uploaded_file.name.split(".")[-1].lower()

        # Load file dynamically
        if file_ext in ["xlsx", "xls"]:
            raw_df = pd.read_excel(uploaded_file)
        else:
            delimiter = st.sidebar.selectbox(
                "Delimiter", [",", r"\s+", ";", r"\t"], index=0
            )
            raw_df = pd.read_csv(uploaded_file, delimiter=delimiter, engine="python")

        st.subheader("Data Selection")
        col_be, col_int = st.columns(2)
        be_col = col_be.selectbox("Select Binding Energy (eV) Column", raw_df.columns, index=0)
        int_col = col_int.selectbox("Select Intensity (a.u.) Column", raw_df.columns, index=min(1, len(raw_df.columns)-1))

        # Extract and sort by Binding Energy
        df = raw_df[[be_col, int_col]].dropna().copy()
        df.columns = ["BE", "Intensity"]
        df["BE"] = pd.to_numeric(df["BE"])
        df["Intensity"] = pd.to_numeric(df["Intensity"])
        df = df.sort_values(by="BE", ascending=True).reset_index(drop=True)

        # Baseline Subtraction (Linear)
        baseline = np.linspace(df["Intensity"].iloc[0], df["Intensity"].iloc[-1], len(df))
        df["Corrected_Intensity"] = np.maximum(0, df["Intensity"] - baseline)

        x_data = df["BE"].values
        y_data = df["Corrected_Intensity"].values

        # --- Peak Detection & Identification ---
        prominence = st.sidebar.slider("Peak Detection Sensitivity", 0.01, 1.0, 0.1)
        peak_indices, _ = find_peaks(y_data, prominence=prominence * np.max(y_data))
        detected_be = x_data[peak_indices]

        # Match detected peaks to database
        identifications = []
        for be in detected_be:
            matched = "Unassigned"
            for label in selected_elements:
                ref_be = XPS_DATABASE[label]
                if abs(be - ref_be) <= energy_tol:
                    matched = f"{label} (~{ref_be} eV)"
                    break
            identifications.append({"Detected BE (eV)": round(be, 2), "Matched Core Level": matched})

        # --- Peak Fitting Initialization ---
        st.subheader("Interactive Fitting Constraints")
        initial_guesses = []
        bounds_lower = []
        bounds_upper = []

        fit_cols = st.columns(int(num_peaks))

        for idx in range(int(num_peaks)):
            with fit_cols[idx]:
                st.markdown(f"**Peak {idx+1}**")
                def_center = float(detected_be[idx]) if idx < len(detected_be) else float(np.mean(x_data))
                center = st.number_input(f"Center BE (eV) #{idx+1}", value=round(def_center, 2))
                amp = st.number_input(f"Amplitude #{idx+1}", value=float(np.max(y_data) / num_peaks))
                sigma = st.number_input(f"Sigma (Width) #{idx+1}", value=1.0, min_value=0.1, step=0.1)

                initial_guesses.extend([amp, center, sigma])
                bounds_lower.extend([0, center - 5.0, 0.1])
                bounds_upper.extend([np.max(y_data) * 2, center + 5.0, 10.0])

        # --- Perform Fitting ---
        popt, _ = curve_fit(
            multi_gaussian, x_data, y_data, p0=initial_guesses, bounds=(bounds_lower, bounds_upper)
        )

        fit_y = multi_gaussian(x_data, *popt)

        # Plotting Results
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x_data, df["Intensity"], "k.", alpha=0.3, label="Raw Data")
        ax.plot(x_data, fit_y + baseline, "r-", linewidth=2, label="Envelope Fit")
        ax.plot(x_data, baseline, "k--", label="Linear Baseline")

        # Plot individual component peaks
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

        # High-resolution plot formatting
        ax.set_xlabel("Binding Energy (eV)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title(f"XPS Deconvolution Profile - {uploaded_file.name}")
        ax.invert_xaxis()  # XPS convention: higher binding energy on the left
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left")

        st.pyplot(fig)

        # Display Identification & Fitting Summary
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Auto-Identified Features")
            st.dataframe(pd.DataFrame(identifications), use_container_width=True)

        with col_right:
            st.subheader("Deconvoluted Peak Quantification")
            df_peaks = pd.DataFrame(peak_summary)
            st.dataframe(df_peaks, use_container_width=True)

        # --- Excel Export Generation ---
        plot_img_bytes = io.BytesIO()
        fig.savefig(plot_img_bytes, format="png", dpi=200, bbox_inches="tight")
        plot_img_bytes.seek(0)

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df_peaks.to_excel(writer, sheet_name="Peak Summary", index=False)
            pd.DataFrame(identifications).to_excel(writer, sheet_name="Identifications", index=False)
            df.to_excel(writer, sheet_name="Processed Data", index=False)

            ws = writer.sheets["Peak Summary"]
            img = OpenPyxlImage(plot_img_bytes)
            img.anchor = "F2"
            ws.add_image(img)

        st.download_button(
            label="📥 Download Full XPS Analysis Report (.xlsx)",
            data=excel_buffer.getvalue(),
            file_name=f"XPS_Analysis_{uploaded_file.name.rsplit('.', 1)[0]}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.error(f"Error executing fitting/analysis: {e}")
else:
    st.info("Please upload an XPS dataset (`.csv`, `.xlsx`, or `.txt`) to begin analysis.")