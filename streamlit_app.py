import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

st.set_page_config(page_title="Pelatihan Model Logistic Regression", layout="wide")
st.title("🧠 Pelatihan Model Logistic Regression")
st.caption("Upload dataset → latih model → evaluasi → download file .pkl")

# ---------------------------------------------------------
# STEP 1: Upload dataset
# ---------------------------------------------------------
st.header("1. Upload Dataset")
uploaded_file = st.file_uploader("Upload file CSV untuk pelatihan", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    st.dataframe(df.head())

    # -------------------------------------------------------
    # STEP 2: Pilih target dan fitur
    # -------------------------------------------------------
    st.header("2. Pilih Kolom Target dan Fitur")
    target_col = st.selectbox("Pilih kolom target (label)", df.columns, index=len(df.columns) - 1)
    feature_cols = st.multiselect(
        "Pilih kolom fitur",
        [c for c in df.columns if c != target_col],
        default=[c for c in df.columns if c != target_col],
    )

    if len(feature_cols) == 0:
        st.warning("Pilih minimal satu kolom fitur untuk melanjutkan.")
        st.stop()

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Encode kolom kategorikal pada fitur (jika ada)
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        st.info(f"Kolom kategorikal terdeteksi dan akan di-encode otomatis: {cat_cols}")
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Encode target jika bukan numerik
    label_encoder = None
    if y.dtype == "object" or y.dtype.name == "category":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        st.info(f"Target di-encode: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

    # -------------------------------------------------------
    # STEP 3: Pengaturan pelatihan
    # -------------------------------------------------------
    st.header("3. Pengaturan Pelatihan")
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Proporsi data uji (test size)", 0.1, 0.5, 0.2, 0.05)
        scale_features = st.checkbox("Standarisasi fitur (StandardScaler)", value=True)
    with col2:
        C_value = st.number_input("C (inverse regularization strength)", min_value=0.001, value=1.0, step=0.1)
        penalty = st.selectbox("Penalty", ["l2", "l1", "none"], index=0)
    with col3:
        solver = st.selectbox("Solver", ["liblinear", "lbfgs", "saga"], index=0)
        max_iter = st.number_input("Max iterations", min_value=50, value=100, step=50)

    random_state = st.number_input("Random state", min_value=0, value=42, step=1)

    # -------------------------------------------------------
    # STEP 4: Latih model
    # -------------------------------------------------------
    st.header("4. Latih Model")
    if st.button("🚀 Mulai Latih Model", type="primary"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        scaler = None
        if scale_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        with st.spinner("Melatih model..."):
            model = LogisticRegression(
                C=C_value,
                penalty=penalty,
                solver=solver,
                max_iter=int(max_iter),
                random_state=int(random_state),
            )
            model.fit(X_train, y_train)

        st.success("Pelatihan selesai!")

        # simpan ke session_state supaya tidak hilang saat rerun
        st.session_state["model"] = model
        st.session_state["scaler"] = scaler
        st.session_state["label_encoder"] = label_encoder
        st.session_state["feature_cols"] = list(X.columns)
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test

    # -------------------------------------------------------
    # STEP 5: Evaluasi model
    # -------------------------------------------------------
    if "model" in st.session_state:
        st.header("5. Evaluasi Model")
        model = st.session_state["model"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{acc:.3f}")
        m2.metric("Precision", f"{prec:.3f}")
        m3.metric("Recall", f"{rec:.3f}")
        m4.metric("F1-Score", f"{f1:.3f}")

        colA, colB = st.columns(2)
        with colA:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            im = ax.imshow(cm, cmap="Blues")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_xticks(range(cm.shape[1]))
            ax.set_yticks(range(cm.shape[0]))
            st.pyplot(fig)

        with colB:
            if y_proba is not None:
                st.subheader("ROC Curve")
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                fig2, ax2 = plt.subplots()
                ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
                ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
                ax2.set_xlabel("False Positive Rate")
                ax2.set_ylabel("True Positive Rate")
                ax2.legend()
                st.pyplot(fig2)

        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose())

        # -----------------------------------------------------
        # STEP 6: Koefisien model
        # -----------------------------------------------------
        st.header("6. Koefisien Model")
        coef_df = pd.DataFrame({
            "feature": st.session_state["feature_cols"],
            "coefficient": model.coef_[0]
        }).sort_values("coefficient", key=abs, ascending=False)
        st.dataframe(coef_df)
        st.bar_chart(coef_df.set_index("feature")["coefficient"])

        # -----------------------------------------------------
        # STEP 7: Download model
        # -----------------------------------------------------
        st.header("7. Simpan / Download Model")
        buffer = io.BytesIO()
        artifact = {
            "model": model,
            "scaler": st.session_state["scaler"],
            "label_encoder": st.session_state["label_encoder"],
            "feature_cols": st.session_state["feature_cols"],
        }
        joblib.dump(artifact, buffer)
        buffer.seek(0)

        st.download_button(
            label="⬇️ Download Model (.pkl)",
            data=buffer,
            file_name="logistic_regression_model_new.pkl",
            mime="application/octet-stream",
        )
        st.caption(
            "File berisi dict: {'model', 'scaler', 'label_encoder', 'feature_cols'}. "
            "Saat load kembali, gunakan `joblib.load()` lalu ambil masing-masing key."
        )
else:
    st.info("Silakan upload file CSV untuk memulai proses pelatihan.")
    with st.expander("ℹ️ Format dataset yang diharapkan"):
        st.markdown(
            """
            - File **CSV**, baris = data, kolom = fitur + 1 kolom target.
            - Kolom target boleh berisi label numerik (0/1) atau kategori (misal 'Yes'/'No').
            - Kolom kategorikal pada fitur akan otomatis di-*encode* (one-hot).
            - Tidak boleh ada nilai kosong (NaN) — bersihkan data terlebih dahulu jika perlu.
            """
        )