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

# --- CẤU HÌNH CSS ĐỂ TẠO PHONG CÁCH "HOẠT HÌNH CHUYÊN NGHIỆP" ---
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        /* Tổng thể */
        * {
            font-family: 'Poppins', sans-serif !important;
            border-radius: 12px !important; /* Bo tròn mềm mại */
        }
        
        .main {
            background-color: #F0F9F9; /* Màu nền pastel nhẹ */
        }
        
        /* Ẩn Streamlit Menu và Footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Tiêu đề chính */
        .main-header {
            color: #00A8A8 !important; /* Cyan mềm mại */
            text-align: center;
            font-weight: 700 !important;
            font-size: 2.8rem !important;
            margin-bottom: -10px;
        }
        
        .sub-header {
            color: #5F9EA0 !important;
            text-align: center;
            font-style: italic;
            font-weight: 400;
            margin-bottom: 30px;
        }
        
        /* Cấu trúc Card (Hộp chứa) */
        .stContainer {
            background-color: #FFFFFF;
            padding: 30px;
            border-radius: 20px !important;
            box-shadow: 0 10px 25px rgba(0, 168, 168, 0.08); /* Bóng đổ mềm */
            margin-bottom: 20px;
        }
        
        .card-title {
            color: #008080;
            font-weight: 600;
            font-size: 1.5rem;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
        }
        
        /* Styling cho Button */
        .stButton>button {
            background-color: #00A8A8 !important;
            color: white !important;
            border: none !important;
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
        
        /* Kết quả */
        .stAlert {
            border-radius: 15px !important;
            padding: 15px !important;
        }
        
        /* Bác sĩ khuyên text */
        .dr-advice {
            color: #333;
            line-height: 1.6;
            margin-bottom: 15px;
        }
        
        /* --- CSS CHO BẢNG PHÁC ĐỒ HOẠT HÌNH --- */
        .custom-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            margin-top: 15px;
            overflow: hidden;
            border-radius: 15px !important;
            border: 1px solid #E0F2F2;
        }
        
        .custom-table thead th {
            background-color: #E0F7F7;
            color: #008080;
            font-weight: 600;
            text-align: left;
            padding: 12px 15px;
            border-bottom: 2px solid #B2DFDF;
        }
        
        .custom-table tbody td {
            background-color: #FFFFFF;
            padding: 12px 15px;
            border-bottom: 1px solid #E0F2F2;
            vertical-align: top;
        }
        
        .custom-table tbody tr:last-child td {
            border-bottom: none;
        }
        
        .custom-table tbody tr:hover td {
            background-color: #F8FFFF;
        }
    </style>
""", unsafe_allow_html=True)

# 1. TIÊU ĐỀ CHÍNH
st.markdown('<h1 class="main-header">Hệ thống chẩn đoán tiểu đường - Bác sĩ AI 👨‍⚕️</h1>', unsafe_allow_html=True)
st.markdown('<h4 class="sub-header">Cá nhân hóa phác đồ sinh hoạt chuẩn Y khoa của bạn</h4>', unsafe_allow_html=True)
st.write("---")

# 2. HUẤN LUYỆN MÔ HÌNH VÀ LƯU VÀO BỘ NHỚ ĐỆM
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

# 3. THIẾT KẾ PHÂN PHỐI GIAO DIỆN (TWO CARDS LAYOUT)
if model is not None:
    col1, col2 = st.columns([2, 3], gap="large")

    with col1:
        st.markdown('<div class="stContainer">', unsafe_allow_html=True)
        st.markdown('<div class="card-title"><span>👨‍⚕️ Khu vực Kiểm tra Lâm sàng</span></div>', unsafe_allow_html=True)
        st.write("Hãy điền các chỉ số cơ thể hiện tại:")
        
        glucose = st.number_input("Nồng độ đường huyết (Glucose - mg/dL)", min_value=0, max_value=500, value=100, key="glucose")
        blood_pressure = st.number_input("Huyết áp tâm thu (Blood Pressure - mmHg)", min_value=0, max_value=300, value=120, key="bp")
        bmi = st.number_input("Chỉ số khối cơ thể (BMI - kg/m²)", min_value=0.0, max_value=100.0, value=22.0, step=0.1, key="bmi")
        age = st.slider("Tuổi bệnh nhân (Age)", min_value=1, max_value=120, value=30, key="age")
        
        st.write("")
        predict_btn = st.button("🩺 Yêu cầu Bác sĩ AI chẩn đoán ngay", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="stContainer">', unsafe_allow_html=True)
        st.markdown('<div class="card-title"><span>📋 Chẩn đoán của Bác sĩ AI 🤖</span></div>', unsafe_allow_html=True)
        
        if predict_btn:
            input_df = pd.DataFrame([[glucose, blood_pressure, bmi, age]], 
                                    columns=['Glucose', 'BloodPressure', 'BMI', 'Age'])
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            
            if prediction == 1:
                st.markdown('<div class="stAlert stError">', unsafe_allow_html=True)
                st.error("⚠️ KẾT LUẬN LÂM SÀNG: CÓ NGUY CƠ CAO MẮC BỆNH TIỂU ĐƯỜNG")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown(f'<p class="dr-advice"><strong>Chào bạn, tôi là Bác sĩ AI 🤖.</strong><br>Dựa trên các chỉ số của bạn, đặc biệt là Đường huyết ({glucose}), hệ thống phát hiện nguy cơ cao mắc bệnh tiểu đường. Đây là phác đồ sinh hoạt chuẩn Y khoa để kiểm soát tình hình:</p>', unsafe_allow_html=True)
                
                high_risk_html = """
                <table class="custom-table">
                    <thead>
                        <tr>
                            <th>Khoảng thời gian</th>
                            <th>Hoạt động khuyến nghị</th>
                            <th>Chi tiết hướng dẫn từ Bác sĩ 🩺</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><b>Buổi sáng</b><br>(06:00 - 07:00)</td>
                            <td>Tập luyện Insulin-nhạy</td>
                            <td>Đi bộ nhanh, cardio nhẹ nhàng 30-45 phút giúp tế bào tiêu thụ đường hiệu quả hơn.</td>
                        </tr>
                        <tr>
                            <td><b>Bữa sáng</b><br>(07:30)</td>
                            <td>Dinh dưỡng GI thấp</td>
                            <td>Hạn chế tinh bột trắng. Ưu tiên yến mạch, gạo lứt và ít nhất 1 chén rau xanh.</td>
                        </tr>
                        <tr>
                            <td><b>Trưa & Chiều</b></td>
                            <td>Cân bằng Insulin</td>
                            <td>Uống 2 lít nước. Cắt giảm hoàn toàn đồ ngọt. Vận động nhẹ mỗi 1 tiếng.</td>
                        </tr>
                        <tr>
                            <td><b>Bữa tối</b><br>(18:30)</td>
                            <td>Phục hồi Tuyến tụy</td>
                            <td>Kết thúc bữa tối trước 20:00. Ưu tiên đồ hấp/luộc, hạn chế dầu mỡ.</td>
                        </tr>
                    </tbody>
                </table>
                """
                st.markdown(high_risk_html, unsafe_allow_html=True)
                
            else:
                st.markdown('<div class="stAlert stSuccess">', unsafe_allow_html=True)
                st.success("✅ KẾT LUẬN LÂM SÀNG: CHƯA PHÁT HIỆN NGUY CƠ MẮC BỆNH TIỂU ĐƯỜNG")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown(f'<p class="dr-advice"><strong>Chào bạn, tôi là Bác sĩ AI 🤖.</strong><br>Thật tuyệt vời! Các chỉ số cơ thể hiện tại của bạn đang nằm trong phân vùng an toàn lý tưởng. Để duy trì sức khỏe này, đây là phác đồ sinh hoạt tôi thiết kế dành riêng cho bạn:</p>', unsafe_allow_html=True)
                
                low_risk_html = """
                <table class="custom-table">
                    <thead>
                        <tr>
                            <th>Chế độ sinh hoạt</th>
                            <th>Mục tiêu hướng tới</th>
                            <th>Chi tiết hướng dẫn từ Bác sĩ 🩺</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><b>Thể thao</b></td>
                            <td>Duy trì thể trạng</td>
                            <td>Duy trì thói quen tập luyện tối thiểu 150 phút/tuần với các môn bơi, chạy bộ.</td>
                        </tr>
                        <tr>
                            <td><b>Dinh dưỡng</b></td>
                            <td>Nuôi dưỡng cơ thể</td>
                            <td>Cân bằng 4 nhóm chất. Tăng cường rau xanh, trái cây ít đường và đạm sạch.</td>
                        </tr>
                        <tr>
                            <td><b>Thói quen</b></td>
                            <td>Chống mỡ nội tạng</td>
                            <td>Tuyệt đối không bỏ bữa sáng. Tránh ăn đêm và uống nước ngọt định kỳ.</td>
                        </tr>
                        <tr>
                            <td><b>Sức khỏe</b></td>
                            <td>Kiểm soát chủ động</td>
                            <td>Tiến hành tầm soát đường huyết và huyết áp định kỳ 6 tháng đến 1 năm.</td>
                        </tr>
                    </tbody>
                </table>
                """
                st.markdown(low_risk_html, unsafe_allow_html=True)
                
        else:
            st.info("Trợ lý chẩn đoán đang sẵn sàng. Điền thông tin sinh hiệu bên trái và nhấn nút.")
        st.markdown('</div>', unsafe_allow_html=True)

st.write("---")
st.caption("Ứng dụng thuộc đề tài nghiên cứu phân loại tuyến tính trong Y tế - Assignment Teamwork.")
