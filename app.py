import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

st.set_page_config(
    page_title="AI Doctor - Hệ thống dự đoán tiểu đường",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            font-family: 'Poppins', sans-serif !important;
        }
        .main-header {
            color: #00A8A8 !important; 
            text-align: center;
            font-weight: 700 !important;
            font-size: 2.5rem !important;
            margin-bottom: 5px;
        }
        .sub-header {
            color: #5F9EA0 !important;
            text-align: center;
            font-style: italic;
            font-weight: 400;
            margin-bottom: 30px;
        }
        .stButton>button {
            background-color: #00A8A8 !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 12px 24px !important;
            font-weight: 600 !important;
            transition: all 0.3s !important;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #008080 !important;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 128, 128, 0.3);
        }
        div[data-testid="stNumberInput"] label, div[data-testid="stSlider"] label {
            color: #008080 !important;
            font-weight: 600 !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Hệ thống chẩn đoán tiểu đường - Bác sĩ AI 👨‍⚕️</h1>', unsafe_allow_html=True)
st.markdown('<h4 class="sub-header">Cá nhân hóa phác đồ sinh hoạt chuẩn Y khoa của bạn</h4>', unsafe_allow_html=True)
st.write("---")

@st.cache_resource
def train_model():
    file_names = ['diabetes.csv', 'diabetes (1).csv']
    df = None
    for f in file_names:
        try:
            df = pd.read_csv(f, encoding='utf-8-sig')
            break
        except:
            continue
            
    if df is None:
        st.error("❌ Không tìm thấy file dữ liệu (diabetes.csv).")
        return None, None

    df.columns = df.columns.str.strip()
    features = ['Glucose', 'BloodPressure', 'BMI', 'Age']
    target = 'Outcome'
    
    X = df[features]
    y = df[target]
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

model, scaler = train_model()

if model is not None:
    col1, col2 = st.columns([2, 3], gap="large")

    with col1:
        st.markdown('### 👨‍⚕️ Khu vực Kiểm tra Lâm sàng')
        st.write("Hãy điền các chỉ số cơ thể hiện tại:")
        
        glucose = st.number_input("Nồng độ đường huyết (Glucose - mg/dL)", min_value=0, max_value=500, value=100)
        blood_pressure = st.number_input("Huyết áp tâm thu (Blood Pressure - mmHg)", min_value=0, max_value=300, value=120)
        bmi = st.number_input("Chỉ số khối cơ thể (BMI - kg/m²)", min_value=0.0, max_value=100.0, value=22.0, step=0.1)
        age = st.slider("Tuổi bệnh nhân (Age)", min_value=1, max_value=120, value=30)
        
        st.write("")
        predict_btn = st.button("🩺 Yêu cầu Bác sĩ AI chẩn đoán ngay", type="primary", use_container_width=True)

    with col2:
        st.markdown('### 📋 Chẩn đoán của Bác sĩ AI 🤖')
        
        if predict_btn:
            input_df = pd.DataFrame([[glucose, blood_pressure, bmi, age]], 
                                    columns=['Glucose', 'BloodPressure', 'BMI', 'Age'])
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            
            if prediction == 1:
                st.error("⚠️ **KẾT LUẬN LÂM SÀNG: CÓ NGUY CƠ CAO MẮC BỆNH TIỂU ĐƯỜNG**")
                st.write(f"**Chào bạn, tôi là Bác sĩ AI 🤖.** Dựa trên các chỉ số của bạn, đặc biệt là Đường huyết ở mức **{glucose}**, hệ thống phát hiện nguy cơ cao mắc bệnh tiểu đường. Dưới đây là phác đồ sinh hoạt tôi thiết kế dành riêng cho bạn:")
                
                st.warning("🌅 **Buổi sáng (06:00 - 07:00): Tập luyện Insulin-nhạy**\n\nĐi bộ nhanh, cardio nhẹ nhàng 30-45 phút giúp tế bào tiêu thụ đường hiệu quả hơn.")
                st.info("🥗 **Bữa sáng (07:30): Dinh dưỡng GI thấp**\n\nHạn chế tinh bột trắng. Ưu tiên yến mạch, gạo lứt và ít nhất 1 chén rau xanh.")
                st.success("💧 **Trưa & Chiều: Cân bằng Insulin**\n\nUống đủ 2 lít nước. Cắt giảm hoàn toàn đồ ngọt. Đứng dậy vận động nhẹ mỗi 1 tiếng làm việc.")
                st.error("🌙 **Bữa tối (18:30): Phục hồi Tuyến tụy**\n\nKết thúc bữa tối trước 20:00. Ưu tiên đồ hấp/luộc, hạn chế dầu mỡ và tinh bột.")
                
            else:
                st.success("✅ **KẾT LUẬN LÂM SÀNG: CHƯA PHÁT HIỆN NGUY CƠ MẮC BỆNH TIỂU ĐƯỜNG**")
                st.write("**Chào bạn, tôi là Bác sĩ AI 🤖.** Thật tuyệt vời! Các chỉ số cơ thể hiện tại của bạn đang nằm trong phân vùng an toàn lý tưởng. Để duy trì phong độ sức khỏe này, hãy thực hiện:")
                
                st.info("🏃‍♂️ **Thể thao: Duy trì độ dẻo dai**\n\nDuy trì thói quen tập luyện tối thiểu 150 phút/tuần với các môn bơi, chạy bộ hoặc gym.")
                st.success("🥑 **Dinh dưỡng: Nuôi dưỡng cơ thể**\n\nCân bằng 4 nhóm chất. Tăng cường rau xanh, trái cây ít đường và đạm sạch (cá, ức gà).")
                st.warning("⏰ **Thói quen: Chống mỡ nội tạng**\n\nTuyệt đối không bỏ bữa sáng. Tránh ăn đêm và hạn chế tối đa nước ngọt đóng chai.")
                st.info("🏥 **Sức khỏe: Kiểm soát chủ động**\n\nTiến hành tầm soát đường huyết và huyết áp định kỳ 6 tháng đến 1 năm một lần.")
                
        else:
            st.info("Trợ lý chẩn đoán đang sẵn sàng. Điền thông tin sinh hiệu bên trái và nhấn nút.")

st.write("---")
st.caption("Ứng dụng thuộc đề tài nghiên cứu phân loại tuyến tính trong Y tế - Assignment Teamwork.")
