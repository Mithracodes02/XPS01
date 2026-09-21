# --- Updated VAMAS Parsing Logic ---
if file_ext in ["vcm", "vms"]:
    if not HAS_VAMAS:
        st.error("The `vamas` library is missing. Please ensure `vamas` is in requirements.txt.")
    else:
        bytes_data = uploaded_file.read()
        vms_obj = vamas.Vamas(bytes_data)
        block = vms_obj.blocks[0]
        
        # Robust attribute extraction for X (Binding Energy) and Y (Intensity)
        if hasattr(block, "abscissa"):
            x_vals = block.abscissa
        elif hasattr(block, "x"):
            x_vals = block.x
        elif hasattr(block, "x_array"):
            x_vals = block.x_array
        else:
            # Fallback calculating X from VAMAS header parameters
            x_start = getattr(block, "abscissa_start", 0)
            x_step = getattr(block, "abscissa_increment", 1)
            num_pts = getattr(block, "num_corresponding_variables", len(block.corresponding_variables[0].array))
            x_vals = [x_start + i * x_step for i in range(num_pts)]

        # Extract intensity array
        if hasattr(block, "corresponding_variables") and len(block.corresponding_variables) > 0:
            y_vals = block.corresponding_variables[0].array
        elif hasattr(block, "y"):
            y_vals = block.y
        else:
            y_vals = block.y_array

        df = pd.DataFrame({"BE": x_vals, "Intensity": y_vals})
