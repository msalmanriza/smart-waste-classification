import streamlit as st
import requests
import time
from PIL import Image
from waste_info import WASTE_INFO, DISPLAY_NAME

st.set_page_config(
    page_title="Smart Waste Classification",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ Smart Waste Classification")

with st.sidebar:
    st.title("♻️ Smart Waste AI")

    st.markdown("### Model")
    st.success("MobileNetV2")

    st.markdown("### Version")
    st.info("v1.0")

    st.divider()

    if st.button("Clear Result", use_container_width=True):
        st.session_state.clear()
        st.rerun()

st.markdown("""
Upload foto sampah atau ambil gambar langsung dari kamera.
Model akan memprediksi jenis sampah beserta tingkat kepercayaannya.
""")

tab1, tab2 = st.tabs(["Upload Gambar", "Kamera"])

with tab1:
    file = st.file_uploader(
        "Upload Gambar",
        type=["jpg", "jpeg", "png"]
    )

with tab2:
    camera = st.camera_input("Ambil Foto")

if camera:
    file = camera

if file:

    img = Image.open(file)

    st.image(
        img,
        caption="Gambar yang akan diprediksi",
        use_container_width=True
    )

    current_file = file.name

    if st.session_state.get("last_file") != current_file:

        st.session_state["last_file"] = current_file

        files = {
            "file": (
                file.name,
                file.getvalue(),
                file.type
            )
        }

        try:
            with st.spinner("AI sedang melakukan prediksi..."):

                time.sleep(3)

                res = requests.post(
                    "http://127.0.0.1:8000/predict",
                    files=files,
                    timeout=20
                )

                if res.status_code == 200:
                    st.session_state["result"] = res.json()
                else:
                    st.error("Prediksi gagal.")
                    st.stop()

        except requests.exceptions.RequestException:
            st.error("FastAPI belum berjalan.")
            st.stop()

    if "result" in st.session_state:

        data = st.session_state["result"]
        confidence = data["confidence"]

        st.success("Prediksi berhasil")
        st.subheader("📌 Hasil Prediksi")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Jenis Sampah",
                DISPLAY_NAME.get(data["class"], data["class"])
            )

        with col2:
            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

        st.progress(confidence)

        info = WASTE_INFO.get(data["class"])

        if info:

            st.subheader("♻️ Informasi Sampah")

            st.write("**Kategori**")
            st.info(info["category"])

            st.write("**Cara Pembuangan**")
            st.write(info["disposal"])

            st.write("**Tips**")
            st.write(info["tips"])

        st.subheader("💡 Informasi")

        if confidence >= 0.90:
            st.success("Model sangat yakin dengan hasil prediksi.")

        elif confidence >= 0.70:
            st.info(
                "Model cukup yakin, namun masih ada kemungkinan kelas lain yang mirip."
            )

        else:
            st.warning(
                "Confidence masih rendah. Coba gunakan pencahayaan yang lebih baik dan arahkan objek lebih dekat ke kamera."
            )