import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. CẤU HÌNH GIAO DIỆN WEB APP (PHONG CÁCH AI HEALTHCARE CLINIC)
st.set_page_config(
    page_title="Hệ thống dự đoán khả năng mắc bệnh tiểu đường dựa trên chỉ số cơ thể",
    page_icon="🩺",
    layout="wide"
)

# Thiết lập tiêu đề theo yêu cầu mới
st.markdown("<h1 style='text-align: center; color: #008080;'>🩺 Hệ thống dự đoán khả năng mắc bệnh tiểu đường dựa trên chỉ số cơ thể</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #5F9EA0; font-style: italic;'>Trợ lý Bác sĩ AI hỗ trợ phân tích và sàng lọc nguy cơ bệnh lý tuyến tiền phát</h4>", unsafe_allow_html=True)
st.write("---")

# 2. HUẤN LUYỆN MÔ HÌNH VÀ LƯU VÀO BỘ NHỚ ĐỆM (CACHE)
@st.cache_resource
def train_perceptron_model():
    try:
        df = pd.read_csv('diabetes.csv')
    except FileNotFoundError:
        df = pd.read_csv('diabetes (1).csv')
        
    X = df[['Age', 'BMI', 'Glucose', 'BloodPressure']]
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    
    return model, scaler, acc, cm, cr

model, scaler, accuracy, conf_matrix, class_report = train_perceptron_model()

# 3. THIẾT KẾ PHÂN PHỐI GIAO DIỆN (LAYOUT)
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 👨‍⚕️ Nhập hồ sơ sinh hiệu lâm sàng")
    st.write("Hãy cung cấp các chỉ số cơ thể hiện tại để Bác sĩ AI tiến hành phân tích:")
    
    age = st.slider("Tuổi của bạn (Age)", min_value=1, maximum=120, value=35)
    bmi = st.number_input("Chỉ số khối cơ thể (BMI - kg/m²)", min_value=10.0, max_value=60.0, value=24.5, step=0.1)
    glucose = st.number_input("Nồng độ đường huyết lúc đói (Glucose - mg/dL)", min_value=30, max_value=300, value=110)
    blood_pressure = st.number_input("Huyết áp tâm thu (Blood Pressure - mmHg)", min_value=40, max_value=240, value=120)
    
    st.write("")
    predict_btn = st.button("🩺 Yêu cầu Bác sĩ AI chẩn đoán", type="primary", use_container_width=True)

with col2:
    st.markdown("### 📋 Kết luận từ Bác sĩ AI")
    
    if predict_btn:
        input_df = pd.DataFrame([[age, bmi, glucose, blood_pressure]], 
                                columns=['Age', 'BMI', 'Glucose', 'BloodPressure'])
        
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        
        # Phần phản hồi sinh động, mang văn phong bác sĩ thực thụ
        if prediction == 1:
            st.error("⚠️ KẾT LUẬN: CÓ NGUY CƠ CAO MẮC BỆNH TIỂU ĐƯỜNG")
            
            st.markdown(f"""
            **Chào bạn, tôi là Trợ lý Bác sĩ AI.** Sau khi phân tích kỹ lưỡng các chỉ số sinh hiệu mà bạn cung cấp dựa trên mạng nơ-ron Perceptron, tôi nhận thấy cơ thể bạn đang phát ra những tín hiệu cảnh báo đáng chú ý:
            * Chỉ số đường huyết (**{glucose} mg/dL**) kết hợp với thể trạng BMI (**{bmi}**) đang nằm trong phân vùng rủi ro của mô hình bệnh lý tuyến tính.
            
            🩺 **Lời khuyên y khoa dành riêng cho bạn:**
            1.  **Thăm khám chuyên khoa:** Bạn cần sắp xếp thời gian đến gặp bác sĩ nội tiết sớm để thực hiện xét nghiệm **HbA1c** và nghiệm pháp dung nạp Glucose nhằm có kết quả khẳng định chính xác nhất.
            2.  **Kiểm soát chế độ ăn:** Cắt giảm ngay lập tức các loại tinh bột tinh chế, đường hấp thu nhanh, nước ngọt và đồ ăn nhiều mỡ động vật.
            3.  **Vận động tích cực:** Hãy duy trì việc đi bộ hoặc tập thể thao nhẹ nhàng ít nhất 30 phút mỗi ngày để tăng tính nhạy cảm của Insulin cơ thể.
            """)
        else:
            st.success("✅ KẾT LUẬN: CHƯA PHÁT HIỆN NGUY CƠ MẮC BỆNH TIỂU ĐƯỜNG")
            
            st.markdown(f"""
            **Chào bạn, tôi là Trợ lý Bác sĩ AI.**
            
            Hệ thống phân tích dữ liệu lâm sàng của tôi ghi nhận các chỉ số cơ thể hiện tại của bạn (Tuổi: {age}, BMI: {bmi}, Đường huyết: {glucose}, Huyết áp: {blood_pressure}) đều đang nằm trong phạm vi an toàn và có sự tương thích tốt. 
            
            🩺 **Lời khuyên duy trì sức khỏe từ Bác sĩ:**
            1.  **Duy trì phong độ:** Chỉ số của bạn rất đẹp, hãy tiếp tục duy trì chế độ dinh dưỡng giàu chất xơ, rau xanh và hạn chế tối đa việc ăn đêm.
            2.  **Tầm soát định kỳ:** Bệnh tiểu đường thường diễn biến âm thầm. Dù hiện tại an toàn, bạn vẫn nên kiểm tra đường huyết định kỳ mỗi 6 tháng đến 1 năm, đặc biệt là khi tuổi tác tăng lên.
            3.  **Lắng nghe cơ thể:** Nếu có bất kỳ triệu chứng đột ngột nào như nhanh khát nước, đi tiểu nhiều lần trong đêm hoặc sụt cân không rõ nguyên nhân, hãy liên hệ với cơ sở y tế ngay.
            """)
    else:
        st.info("Trợ lý Bác sĩ AI đã sẵn sàng. Vui lòng điền đầy đủ các thông số sinh hiệu ở bảng bên trái và nhấn nút 'Yêu cầu Bác sĩ AI chẩn đoán'.")

    # Giữ nguyên phần đánh giá kỹ thuật phục vụ báo cáo học thuật
    st.write("---")
    st.markdown("### ⚙️ Thông số kỹ thuật mô hình (Model Analytics)")
    
    st.metric(label="Độ chính xác tổng thể (Accuracy)", value=f"{accuracy * 100:.2f}%")
    
    with st.expander("Xem chi tiết cấu trúc thuật toán"):
        st.write("**Ma trận nhầm lẫn (Confusion Matrix):**")
        cm_df = pd.DataFrame(
            conf_matrix, 
            index=["Thực tế: Khỏe mạnh (0)", "Thực tế: Tiểu đường (1)"], 
            columns=["Dự đoán: Khỏe mạnh (0)", "Dự đoán: Tiểu đường (1)"]
        )
        st.dataframe(cm_df, use_container_width=True)
        
        st.write("**Báo cáo chi tiết phân loại (Classification Report):**")
        cr_df = pd.DataFrame(class_report).transpose()
        st.dataframe(cr_df.style.format(precision=2), use_container_width=True)

st.write("---")
st.caption("Ứng dụng thuộc đề tài nghiên cứu phân loại tuyến tính trong Y tế - Assignment Teamwork.")
