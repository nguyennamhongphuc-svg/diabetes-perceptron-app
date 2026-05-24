import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

st.set_page_config(
    page_title="Hệ thống dự đoán tiểu đường - AI Healthcare System",
    page_icon="🩺",
    layout="wide"
)

st.markdown("<h1 style='text-align: center; color: #008080;'>Hệ thống dự đoán khả năng mắc bệnh tiểu đường dựa trên chỉ số cơ thể</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #5F9EA0; font-style: italic;'>Giải pháp công nghệ hỗ trợ sàng lọc lâm sàng và xây dựng phác đồ sinh hoạt cá nhân hóa</h4>", unsafe_allow_html=True)
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
        return None, None

    df.columns = df.columns.str.strip()
    features = ['Glucose', 'BloodPressure', 'BMI', 'Age']
    target = 'Outcome'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

model, scaler = train_model()

if model is not None:
    col1, col2 = st.columns([2, 3], gap="large")

    with col1:
        st.markdown("<h3 style='color: #008080;'>👨‍⚕️ Thông số Sinh hiệu Lâm sàng</h3>", unsafe_allow_html=True)
        st.write("Vui lòng cung cấp chính xác các chỉ số cơ thể hiện tại:")
        
        glucose = st.number_input("Nồng độ đường huyết (Glucose - mg/dL)", min_value=0, max_value=500, value=100)
        blood_pressure = st.number_input("Huyết áp tâm thu (Blood Pressure - mmHg)", min_value=0, max_value=300, value=120)
        bmi = st.number_input("Chỉ số khối cơ thể (BMI - kg/m²)", min_value=0.0, max_value=100.0, value=22.0, step=0.1)
        age = st.slider("Tuổi bệnh nhân (Age)", min_value=1, max_value=120, value=30)
        
        st.write("")
        predict_btn = st.button("🩺 Yêu cầu Bác sĩ AI phân tích", type="primary", use_container_width=True)

    with col2:
        st.markdown("<h3 style='color: #008080;'>📋 Đánh giá & Phác đồ từ Bác sĩ AI</h3>", unsafe_allow_html=True)
        
        if predict_btn:
            input_df = pd.DataFrame([[glucose, blood_pressure, bmi, age]], 
                                    columns=['Glucose', 'BloodPressure', 'BMI', 'Age'])
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            
            if prediction == 1:
                st.error("⚠️ KẾT LUẬN LÂM SÀNG: CÓ NGUY CƠ CAO MẮC BỆNH TIỂU ĐƯỜNG")
                st.markdown(f"""
                **Chào bạn, tôi là Trợ lý Bác sĩ AI.** Sau khi đối chiếu các thông số sinh hiệu của bạn với mô hình bệnh lý, hệ thống nhận thấy cơ thể bạn đang có những dấu hiệu rối loạn chuyển hóa đường rõ rệt. 
                
                Để kiểm soát đường huyết và ngăn chặn tiến triển bệnh, bạn cần tuân thủ nghiêm ngặt bảng hướng dẫn điều chỉnh lối sống dưới đây:
                """)
                
                high_risk_schedule = {
                    "Khoảng thời gian": ["Buổi sáng (06:00 - 07:00)", "Bữa sáng (07:30)", "Trưa & Chiều làm việc", "Bữa tối (18:30)", "Trước khi đi ngủ"],
                    "Hoạt động khuyến nghị": ["Vận động tiêu hao đường", "Dinh dưỡng hạ đường huyết", "Cân bằng chuyển hóa", "Kiểm soát năng lượng muộn", "Thư giãn tế bào thần kinh"],
                    "Chi tiết hướng dẫn từ Bác sĩ": [
                        "Đi bộ nhanh, đạp xe hoặc tập cardio nhẹ nhàng trong 30-45 phút giúp tăng tính nhạy cảm của Insulin.",
                        "Tuyệt đối không bỏ bữa. Ưu tiên rau xanh chiếm 50% khẩu phần, dùng tinh bột hấp thu chậm (gạo lứt, yến mạch).",
                        "Dùng đạm sạch (ức gà, cá, đậu phụ). Uống đủ 2 lít nước lọc, tuyệt đối không dùng nước ngọt hay trà sữa.",
                        "Ăn thanh đạm, ưu tiên đồ luộc/hấp và kết thúc bữa ăn trước 20:00 để tránh áp lực lên tuyến tụy.",
                        "Thực hiện thiền hoặc yoga nhẹ nhàng để giảm áp lực tâm lý (stress làm giải phóng cortisol gây tăng đường huyết)."
                    ]
                }
                st.table(pd.DataFrame(high_risk_schedule))
                
            else:
                st.success("✅ KẾT LUẬN LÂM SÀNG: CHƯA PHÁT HIỆN NGUY CƠ MẮC BỆNH TIỂU ĐƯỜNG")
                st.markdown(f"""
                **Chào bạn, tôi là Trợ lý Bác sĩ AI.** Thật vui mừng khi các chỉ số cơ thể hiện tại của bạn đang nằm trong phân vùng an toàn và chuyển hóa rất tốt.
                
                Nhằm duy trì trạng thái thể chất lý tưởng này và chủ động phòng ngừa rủi ro tiềm ẩn, tôi thiết kế cho bạn phác đồ sinh hoạt lành mạnh sau:
                """)
                
                low_risk_schedule = {
                    "Chế độ sinh hoạt": ["Hoạt động thể chất", "Chế độ dinh dưỡng", "Thói quen hằng ngày", "Tầm soát sức khỏe"],
                    "Mục tiêu hướng tới": ["Duy trì độ dẻo dai cơ bắp", "Nuôi dưỡng cơ thể lành mạnh", "Tránh tích tụ mỡ nội tạng", "Chủ động kiểm soát rủi ro"],
                    "Chi tiết hướng dẫn từ Bác sĩ": [
                        "Duy trì tập luyện thể thao tối thiểu 150 phút mỗi tuần (khoảng 30 phút mỗi ngày với các môn bơi, chạy bộ, thể hình).",
                        "Xây dựng thực đơn đa dạng chất xơ, hạn chế tiêu thụ quá mức đường tinh luyện và mỡ động vật.",
                        "Tránh ngồi làm việc liên tục quá 1 tiếng. Hãy đứng dậy đi lại nhẹ nhàng 5 phút để kích hoạt hệ tuần hoàn.",
                        "Lắng nghe các dấu hiệu thay đổi đột ngột của cơ thể và tiến hành kiểm tra đường huyết đói định kỳ mỗi 6 tháng."
                    ]
                }
                st.table(pd.DataFrame(low_risk_schedule))
        else:
            st.info("Hệ thống sàng lọc thông minh đang sẵn sàng. Vui lòng cung cấp đầy đủ thông số ở bảng bên trái và chọn 'Yêu cầu Bác sĩ AI phân tích'.")

st.write("---")
st.caption("Ứng dụng thuộc đề tài nghiên cứu phân loại tuyến tính trong Y tế - Assignment Teamwork.")
