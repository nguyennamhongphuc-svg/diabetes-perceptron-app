import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. CẤU HÌNH GIAO DIỆN WEB APP (PHONG CÁCH HEALTHCARE SYSTEM)
st.set_page_config(
    page_title="Diabetes Diagnosis System",
    page_icon="🏥",
    layout="wide"
)

# Tiêu đề chính ứng dụng
st.markdown("<h1 style='text-align: center; color: #008080;'>🏥 Diabetes Diagnosis System using Perceptron Network</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #4682B4;'>Hệ thống Chẩn đoán Tiểu đường bằng Mạng Phân Loại Tuyến Tính Perceptron</h3>", unsafe_allow_html=True)
st.write("---")

# 2. HUẤN LUYỆN MÔ HÌNH VÀ LƯU VÀO BỘ NHỚ ĐỆM (CACHE) KHÔNG LO TRAIN LẠI KHI CLICK BUTTON
@st.cache_resource
def train_perceptron_model():
    # Đọc dữ liệu từ file cùng cấp thư mục
    try:
        df = pd.read_csv('diabetes.csv')
    except FileNotFoundError:
        # Dự phòng trường hợp tên file còn giữ nguyên ký tự tải về ban đầu
        df = pd.read_csv('diabetes (1).csv')
        
    # Tách thuộc tính đặc trưng (Features) và nhãn kết quả (Target)
    X = df[['Age', 'BMI', 'Glucose', 'BloodPressure']]
    y = df['Outcome']
    
    # Chia tập dữ liệu Train/Test theo tỷ lệ 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Chuẩn hóa dữ liệu đầu vào (Z-score Scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Khởi tạo và huấn luyện mạng Perceptron
    model = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Đánh giá các chỉ số kiểm thử
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    
    return model, scaler, acc, cm, cr

# Tải mô hình và các chỉ số đánh giá đã tính toán
model, scaler, accuracy, conf_matrix, class_report = train_perceptron_model()

# 3. THIẾT KẾ PHÂN PHỐI GIAO DIỆN (LAYOUT)
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 👨‍⚕️ Thông tin lâm sàng bệnh nhân")
    st.write("Vui lòng nhập chính xác các chỉ số sức khỏe đo được:")
    
    # Thanh trượt và các hộp nhập số liệu
    age = st.slider("Tuổi (Age)", min_value=1, max_value=120, value=35)
    bmi = st.number_input("Chỉ số khối cơ thể (BMI)", min_value=10.0, max_value=60.0, value=24.5, step=0.1)
    glucose = st.number_input("Nồng độ đường huyết (Glucose - mg/dL)", min_value=30, max_value=300, value=110)
    blood_pressure = st.number_input("Huyết áp tâm thu (Blood Pressure - mmHg)", min_value=40, max_value=240, value=120)
    
    st.write("")
    predict_btn = st.button("🔍 Tiến hành chẩn đoán", type="primary", use_container_width=True)

with col2:
    st.markdown("### 📋 Kết quả phân tích từ AI")
    
    if predict_btn:
        # Chuẩn bị dữ liệu đầu vào từ Form nhập liệu
        input_df = pd.DataFrame([[age, bmi, glucose, blood_pressure]], 
                                columns=['Age', 'BMI', 'Glucose', 'BloodPressure'])
        
        # Thực hiện chuẩn hóa dựa trên tham số bộ dữ liệu huấn luyện
        input_scaled = scaler.transform(input_df)
        
        # Dự đoán nhãn phân loại (0 hoặc 1)
        prediction = model.predict(input_scaled)[0]
        
        if prediction == 1:
            st.error("⚠️ CÓ NGUY CƠ MẮC BỆNH TIỂU ĐƯỜNG")
            st.markdown("""
            **Khuyến nghị y tế:** Hệ thống phát hiện các chỉ số lâm sàng của bạn có dấu hiệu trùng khớp với mô hình bệnh lý tiểu đường. 
            Bạn nên sớm đến các bệnh viện hoặc cơ sở chuyên khoa để thực hiện xét nghiệm chuyên sâu và nhận phác đồ điều trị kịp thời.
            """)
        else:
            st.success("✅ KHÔNG MẮC BỆNH TIỂU ĐƯỜNG")
            st.markdown("""
            **Khuyến nghị y tế:**
            Các chỉ số hiện tại của bạn đang nằm trong giới hạn an toàn theo mô hình tuyến tính của mạng Perceptron. 
            Hãy tiếp tục giữ vững chế độ dinh dưỡng cân bằng và rèn luyện thể chất đều đặn.
            """)
    else:
        st.info("Hệ thống đang sẵn sàng. Điền thông tin ở bảng bên trái và nhấn nút 'Tiến hành chẩn đoán' để xem kết quả phân tích.")

    st.write("---")
    st.markdown("### ⚙️ Thông số kỹ thuật mô hình (Model Analytics)")
    
    # Hiển thị độ chính xác dạng Metric
    st.metric(label="Độ chính xác tổng thể (Accuracy)", value=f"{accuracy * 100:.2f}%")
    
    # Trực quan hóa Ma trận nhầm lẫn bằng bảng dữ liệu cấu trúc
    st.write("**Ma trận nhầm lẫn (Confusion Matrix):**")
    cm_df = pd.DataFrame(
        conf_matrix, 
        index=["Thực tế: Khỏe mạnh (0)", "Thực tế: Tiểu đường (1)"], 
        columns=["Dự đoán: Khỏe mạnh (0)", "Dự đoán: Tiểu đường (1)"]
    )
    st.dataframe(cm_df, use_container_width=True)
    
    # Hiển thị báo cáo chi tiết Precision, Recall, F1-Score
    st.write("**Báo cáo chi tiết phân loại (Classification Report):**")
    cr_df = pd.DataFrame(class_report).transpose()
    st.dataframe(cr_df.style.format(precision=2), use_container_width=True)

st.write("---")
st.caption("Dự án được xây dựng phục vụ báo cáo môn học Trí tuệ nhân tạo & Học máy.")
