import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Cấu hình trang
st.set_page_config(
    page_title="Hệ thống dự đoán tiểu đường - AI Doctor",
    page_icon="🩺",
    layout="wide"
)

# Giao diện Tiêu đề
st.markdown("<h1 style='text-align: center; color: #008080;'>🩺 Hệ thống dự đoán khả năng mắc bệnh tiểu đường dựa trên chỉ số cơ thể</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #5F9EA0; font-style: italic;'>Trợ lý Bác sĩ AI hỗ trợ phân tích và sàng lọc nguy cơ bệnh lý</h4>", unsafe_allow_html=True)
st.write("---")

# Hàm huấn luyện mô hình với xử lý lỗi dữ liệu
@st.cache_resource
def train_model():
    # Danh sách các tên file có thể có
    file_names = ['diabetes.csv', 'diabetes (1).csv']
    df = None
    
    for f in file_names:
        try:
            # Đọc file với encoding utf-8-sig để loại bỏ lỗi ký tự lạ ở đầu file
            df = pd.read_csv(f, encoding='utf-8-sig')
            break
        except:
            continue
            
    if df is None:
        st.error("❌ Không tìm thấy file dữ liệu (diabetes.csv). Vui lòng kiểm tra lại trên GitHub!")
        return None, None, 0, None, None

    # Xử lý xóa khoảng trắng thừa trong tên cột (nếu có)
    df.columns = df.columns.str.strip()
    
    # Định nghĩa các cột đầu vào dựa trên file của bạn
    features = ['Glucose', 'BloodPressure', 'BMI', 'Age']
    target = 'Outcome'
    
    # Kiểm tra xem các cột có tồn tại không
    if not all(col in df.columns for col in features + [target]):
        st.error(f"❌ File CSV sai cấu trúc cột. Cần có: {features + [target]}")
        return None, None, 0, None, None

    X = df[features]
    y = df[target]
    
    # Chia dữ liệu và Chuẩn hóa
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Huấn luyện Perceptron
    model = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Đánh giá
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    
    return model, scaler, acc, cm, cr

# Thực thi huấn luyện
model, scaler, accuracy, conf_matrix, class_report = train_model()

if model is not None:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 👨‍⚕️ Nhập hồ sơ sinh hiệu lâm sàng")
        st.write("Hãy cung cấp các chỉ số cơ thể hiện tại:")
        
        # Sắp xếp lại thứ tự nhập cho giống thực tế y tế
        glucose = st.number_input("Nồng độ đường huyết (Glucose - mg/dL)", min_value=0, max_value=500, value=100)
        blood_pressure = st.number_input("Huyết áp tâm thu (Blood Pressure - mmHg)", min_value=0, max_value=300, value=120)
        bmi = st.number_input("Chỉ số khối cơ thể (BMI - kg/m²)", min_value=0.0, max_value=100.0, value=22.0, step=0.1)
        age = st.slider("Tuổi của bạn (Age)", min_value=1, max_value=120, value=30)
        
        st.write("")
        predict_btn = st.button("🩺 Yêu cầu Bác sĩ AI chẩn đoán", type="primary", use_container_width=True)

    with col2:
        st.markdown("### 📋 Kết luận từ Bác sĩ AI")
        
        if predict_btn:
            # Tạo DataFrame đúng thứ tự cột khi train
            input_df = pd.DataFrame([[glucose, blood_pressure, bmi, age]], 
                                    columns=['Glucose', 'BloodPressure', 'BMI', 'Age'])
            
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            
            if prediction == 1:
                st.error("⚠️ KẾT LUẬN: CÓ NGUY CƠ CAO MẮC BỆNH TIỂU ĐƯỜNG")
                st.markdown(f"""
                **Chào bạn, tôi là Trợ lý Bác sĩ AI.** Dựa trên chỉ số Đường huyết ({glucose}) và BMI ({bmi}), hệ thống nhận thấy nguy cơ rối loạn chuyển hóa đường rất cao.
                
                🩺 **Lời khuyên y khoa:**
                1. **Khám chuyên khoa:** Bạn cần thực hiện xét nghiệm HbA1c ngay để xác định tình trạng bệnh.
                2. **Chế độ ăn:** Hạn chế tinh bột trắng, đồ ngọt và tăng cường chất xơ.
                3. **Vận động:** Đi bộ ít nhất 150 phút mỗi tuần.
                """)
            else:
                st.success("✅ KẾT LUẬN: CHƯA PHÁT HIỆN NGUY CƠ MẮC BỆNH TIỂU ĐƯỜNG")
                st.markdown(f"""
                **Chào bạn, tôi là Trợ lý Bác sĩ AI.**
                Chỉ số của bạn hiện tại nằm trong ngưỡng an toàn. Đây là kết quả rất tích cực!
                
                🩺 **Lời khuyên duy trì:**
                1. Duy trì chế độ ăn uống điều độ, tránh ăn tối quá muộn.
                2. Kiểm tra sức khỏe định kỳ mỗi 6 tháng để tầm soát sớm.
                """)
        else:
            st.info("Vui lòng nhập thông số và nhấn nút chẩn đoán.")

        with st.expander("⚙️ Xem thông số kỹ thuật (Dành cho báo cáo)"):
            st.write(f"Độ chính xác mô hình: **{accuracy*100:.2f}%**")
            st.write("Báo cáo phân loại:")
            st.dataframe(pd.DataFrame(class_report).transpose())

st.write("---")
st.caption("Ứng dụng hỗ trợ sàng lọc tiểu đường sử dụng thuật toán Perceptron.")
