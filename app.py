import streamlit as st
import pandas as pd
import pickle

# Load trained pipeline
with open('model_pipeline.pkl', 'rb') as f:
    model_pipeline = pickle.load(f)

# App title
st.set_page_config(page_title="Laptop Price Predictor", layout="centered")
st.title("💻 Laptop Price Predictor")
st.markdown("Fill in the laptop details to estimate its price.")

# Use columns to organize form layout
col1, col2 = st.columns(2)

with col1:
   Company = st.selectbox("Company", ['Apple', 'HP', 'Acer', 'Asus', 'Dell', 'Lenovo', 'Chuwi', 'MSI',
       'Microsoft', 'Toshiba', 'Huawei', 'Xiaomi', 'Vero', 'Razer',
       'Mediacom', 'Samsung', 'Google', 'Fujitsu', 'LG'])
   TypeName = st.selectbox("TypeName", ['Ultrabook', 'Notebook', 'Gaming', '2 in 1 Convertible','Workstation', 'Netbook'])
   Cpu = st.selectbox("CPU", ['Intel Core i5 2.3GHz', 'Intel Core i5 1.8GHz',
       'Intel Core i5 7200U 2.5GHz', 'Intel Core i7 2.7GHz',
       'Intel Core i5 3.1GHz', 'AMD A9-Series 9420 3GHz',
       'Intel Core i7 2.2GHz', 'Intel Core i7 8550U 1.8GHz',
       'Intel Core i5 8250U 1.6GHz', 'Intel Core i3 6006U 2GHz',
       'Intel Core i7 2.8GHz', 'Intel Core M m3 1.2GHz',
       'Intel Core i7 7500U 2.7GHz', 'Intel Core i7 2.9GHz',
       'Intel Core i3 7100U 2.4GHz', 'Intel Core i5 7300HQ 2.5GHz',
       'AMD E-Series E2-9000e 1.5GHz', 'Intel Core i5 1.6GHz',
       'Intel Core i7 8650U 1.9GHz', 'Intel Atom x5-Z8300 1.44GHz',
       'AMD E-Series E2-6110 1.5GHz', 'AMD A6-Series 9220 2.5GHz',
       'Intel Celeron Dual Core N3350 1.1GHz',
       'Intel Core i3 7130U 2.7GHz', 'Intel Core i7 7700HQ 2.8GHz',
       'Intel Core i5 2.0GHz', 'AMD Ryzen 1700 3GHz',
       'Intel Pentium Quad Core N4200 1.1GHz',
       'Intel Celeron Dual Core N3060 1.6GHz', 'Intel Core i5 1.3GHz',
       'AMD FX 9830P 3GHz', 'Intel Core i7 7560U 2.4GHz',
       'AMD E-Series 6110 1.5GHz', 'Intel Core i5 6200U 2.3GHz',
       'Intel Core M 6Y75 1.2GHz', 'Intel Core i5 7500U 2.7GHz',
       'Intel Core i3 6006U 2.2GHz', 'AMD A6-Series 9220 2.9GHz',
       'Intel Core i7 6920HQ 2.9GHz', 'Intel Core i5 7Y54 1.2GHz',
       'Intel Core i7 7820HK 2.9GHz', 'Intel Xeon E3-1505M V6 3GHz',
       'Intel Core i7 6500U 2.5GHz', 'AMD E-Series 9000e 1.5GHz',
       'AMD A10-Series A10-9620P 2.5GHz', 'AMD A6-Series A6-9220 2.5GHz',
       'Intel Core i5 2.9GHz', 'Intel Core i7 6600U 2.6GHz',
       'Intel Core i3 6006U 2.0GHz',
       'Intel Celeron Dual Core 3205U 1.5GHz',
       'Intel Core i7 7820HQ 2.9GHz', 'AMD A10-Series 9600P 2.4GHz',
       'Intel Core i7 7600U 2.8GHz', 'AMD A8-Series 7410 2.2GHz',
       'Intel Celeron Dual Core 3855U 1.6GHz',
       'Intel Pentium Quad Core N3710 1.6GHz',
       'AMD A12-Series 9720P 2.7GHz', 'Intel Core i5 7300U 2.6GHz',
       'AMD A12-Series 9720P 3.6GHz',
       'Intel Celeron Quad Core N3450 1.1GHz',
       'Intel Celeron Dual Core N3060 1.60GHz',
       'Intel Core i5 6440HQ 2.6GHz', 'Intel Core i7 6820HQ 2.7GHz',
       'AMD Ryzen 1600 3.2GHz', 'Intel Core i7 7Y75 1.3GHz',
       'Intel Core i5 7440HQ 2.8GHz', 'Intel Core i7 7660U 2.5GHz',
       'Intel Core i7 7700HQ 2.7GHz', 'Intel Core M m3-7Y30 2.2GHz',
       'Intel Core i5 7Y57 1.2GHz', 'Intel Core i7 6700HQ 2.6GHz',
       'Intel Core i3 6100U 2.3GHz', 'Intel Atom x5-Z8350 1.44GHz',
       'AMD A10-Series 9620P 2.5GHz', 'AMD E-Series 7110 1.8GHz',
       'Intel Celeron Dual Core N3350 2.0GHz',
       'AMD A9-Series A9-9420 3GHz', 'Intel Core i7 6820HK 2.7GHz',
       'Intel Core M 7Y30 1.0GHz', 'Intel Xeon E3-1535M v6 3.1GHz',
       'Intel Celeron Quad Core N3160 1.6GHz',
       'Intel Core i5 6300U 2.4GHz', 'Intel Core i3 6100U 2.1GHz',
       'AMD E-Series E2-9000 2.2GHz',
       'Intel Celeron Dual Core N3050 1.6GHz',
       'Intel Core M M3-6Y30 0.9GHz', 'AMD A9-Series 9420 2.9GHz',
       'Intel Core i5 6300HQ 2.3GHz', 'AMD A6-Series 7310 2GHz',
       'Intel Atom Z8350 1.92GHz', 'Intel Xeon E3-1535M v5 2.9GHz',
       'Intel Core i5 6260U 1.8GHz',
       'Intel Pentium Dual Core N4200 1.1GHz',
       'Intel Celeron Quad Core N3710 1.6GHz', 'Intel Core M 1.2GHz',
       'AMD A12-Series 9700P 2.5GHz', 'Intel Core i7 7500U 2.5GHz',
       'Intel Pentium Dual Core 4405U 2.1GHz',
       'AMD A4-Series 7210 2.2GHz', 'Intel Core i7 6560U 2.2GHz',
       'Intel Core M m7-6Y75 1.2GHz', 'AMD FX 8800P 2.1GHz',
       'Intel Core M M7-6Y75 1.2GHz', 'Intel Core i5 7200U 2.50GHz',
       'Intel Core i5 7200U 2.70GHz', 'Intel Atom X5-Z8350 1.44GHz',
       'Intel Core i5 7200U 2.7GHz', 'Intel Core M 1.1GHz',
       'Intel Atom x5-Z8550 1.44GHz',
       'Intel Pentium Dual Core 4405Y 1.5GHz',
       'Intel Pentium Quad Core N3700 1.6GHz', 'Intel Core M 6Y54 1.1GHz',
       'Intel Core i7 6500U 2.50GHz',
       'Intel Celeron Dual Core N3350 2GHz',
       'Samsung Cortex A72&A53 2.0GHz', 'AMD E-Series 9000 2.2GHz',
       'Intel Core M 6Y30 0.9GHz', 'AMD A9-Series 9410 2.9GHz'])
   Ram = st.selectbox("RAM", ['8GB', '16GB', '4GB', '2GB', '12GB', '64GB', '6GB', '32GB', '24GB','1GB'])

with col2:
    Gpu = st.selectbox("GPU", ['Intel Iris Plus Graphics 640', 'Intel HD Graphics 6000',
       'Intel HD Graphics 620', 'AMD Radeon Pro 455',
       'Intel Iris Plus Graphics 650', 'AMD Radeon R5',
       'Intel Iris Pro Graphics', 'Nvidia GeForce MX150',
       'Intel UHD Graphics 620', 'Intel HD Graphics 520',
       'AMD Radeon Pro 555', 'AMD Radeon R5 M430',
       'Intel HD Graphics 615', 'AMD Radeon Pro 560',
       'Nvidia GeForce 940MX', 'Nvidia GeForce GTX 1050', 'AMD Radeon R2',
       'AMD Radeon 530', 'Nvidia GeForce 930MX', 'Intel HD Graphics',
       'Intel HD Graphics 500', 'Nvidia GeForce 930MX ',
       'Nvidia GeForce GTX 1060', 'Nvidia GeForce 150MX',
       'Intel Iris Graphics 540', 'AMD Radeon RX 580',
       'Nvidia GeForce 920MX', 'AMD Radeon R4 Graphics', 'AMD Radeon 520',
       'Nvidia GeForce GTX 1070', 'Nvidia GeForce GTX 1050 Ti',
       'Intel HD Graphics 400', 'Nvidia GeForce MX130', 'AMD R4 Graphics',
       'Nvidia GeForce GTX 940MX', 'AMD Radeon RX 560',
       'Nvidia GeForce 920M', 'AMD Radeon R7 M445', 'AMD Radeon RX 550',
       'Nvidia GeForce GTX 1050M', 'Intel HD Graphics 515',
       'AMD Radeon R5 M420', 'Intel HD Graphics 505',
       'Nvidia GTX 980 SLI', 'AMD R17M-M1-70', 'Nvidia GeForce GTX 1080',
       'Nvidia Quadro M1200', 'Nvidia GeForce 920MX ',
       'Nvidia GeForce GTX 950M', 'AMD FirePro W4190M ',
       'Nvidia GeForce GTX 980M', 'Intel Iris Graphics 550',
       'Nvidia GeForce 930M', 'Intel HD Graphics 630',
       'AMD Radeon R5 430', 'Nvidia GeForce GTX 940M',
       'Intel HD Graphics 510', 'Intel HD Graphics 405',
       'AMD Radeon RX 540', 'Nvidia GeForce GT 940MX',
       'AMD FirePro W5130M', 'Nvidia Quadro M2200M', 'AMD Radeon R4',
       'Nvidia Quadro M620', 'AMD Radeon R7 M460',
       'Intel HD Graphics 530', 'Nvidia GeForce GTX 965M',
       'Nvidia GeForce GTX1080', 'Nvidia GeForce GTX1050 Ti',
       'Nvidia GeForce GTX 960M', 'AMD Radeon R2 Graphics',
       'Nvidia Quadro M620M', 'Nvidia GeForce GTX 970M',
       'Nvidia GeForce GTX 960<U+039C>', 'Intel Graphics 620',
       'Nvidia GeForce GTX 960', 'AMD Radeon R5 520',
       'AMD Radeon R7 M440', 'AMD Radeon R7', 'Nvidia Quadro M520M',
       'Nvidia Quadro M2200', 'Nvidia Quadro M2000M',
       'Intel HD Graphics 540', 'Nvidia Quadro M1000M', 'AMD Radeon 540',
       'Nvidia GeForce GTX 1070M', 'Nvidia GeForce GTX1060',
       'Intel HD Graphics 5300', 'AMD Radeon R5 M420X',
       'AMD Radeon R7 Graphics', 'Nvidia GeForce 920',
       'Nvidia GeForce 940M', 'Nvidia GeForce GTX 930MX',
       'AMD Radeon R7 M465', 'AMD Radeon R3', 'Nvidia GeForce GTX 1050Ti',
       'AMD Radeon R7 M365X', 'AMD Radeon R9 M385',
       'Intel HD Graphics 620 ', 'Nvidia Quadro 3000M',
       'Nvidia GeForce GTX 980 ', 'AMD Radeon R5 M330',
       'AMD FirePro W4190M', 'AMD FirePro W6150M', 'AMD Radeon R5 M315',
       'Nvidia Quadro M500M', 'AMD Radeon R7 M360',
       'Nvidia Quadro M3000M', 'Nvidia GeForce 960M', 'ARM Mali T860 MP4'])
    OpSys = st.selectbox("Operating System", ['macOS', 'No OS', 'Windows 10', 'Mac OS X', 'Linux',
       'Windows 10 S', 'Chrome OS', 'Windows 7', 'Android'])
    ScreenResolution = st.selectbox("Screen Resolution", ['IPS Panel Retina Display 2560x1600', '1440x900',
       'Full HD 1920x1080', 'IPS Panel Retina Display 2880x1800',
       '1366x768', 'IPS Panel Full HD 1920x1080',
       'IPS Panel Retina Display 2304x1440',
       'IPS Panel Full HD / Touchscreen 1920x1080',
       'Full HD / Touchscreen 1920x1080',
       'Touchscreen / Quad HD+ 3200x1800', 'Touchscreen 2256x1504',
       'Quad HD+ / Touchscreen 3200x1800', 'IPS Panel 1366x768',
       'IPS Panel 4K Ultra HD / Touchscreen 3840x2160',
       'IPS Panel Full HD 2160x1440',
       '4K Ultra HD / Touchscreen 3840x2160', '1600x900',
       'IPS Panel 4K Ultra HD 3840x2160', '4K Ultra HD 3840x2160',
       'Touchscreen 1366x768', 'Touchscreen 2560x1440',
       'IPS Panel Full HD 1366x768', 'IPS Panel 2560x1440',
       'IPS Panel Full HD 2560x1440',
       'IPS Panel Retina Display 2736x1824', 'Touchscreen 2400x1600',
       '2560x1440', 'IPS Panel Quad HD+ 2560x1440',
       'IPS Panel Quad HD+ 3200x1800',
       'IPS Panel Quad HD+ / Touchscreen 3200x1800',
       'IPS Panel Touchscreen 1366x768', '1920x1080',
       'IPS Panel Full HD 1920x1200',
       'IPS Panel Touchscreen / 4K Ultra HD 3840x2160',
       'IPS Panel Touchscreen 2560x1440',
       'Touchscreen / Full HD 1920x1080', 'Quad HD+ 3200x1800',
       'IPS Panel Touchscreen 1920x1200',
       'Touchscreen / 4K Ultra HD 3840x2160',
       'IPS Panel Touchscreen 2400x1600'])
    Memory = st.selectbox("Memory", ['128GB SSD', '128GB Flash Storage', '256GB SSD', '512GB SSD',
       '500GB HDD', '256GB Flash Storage', '1TB HDD',
       '128GB SSD +  1TB HDD', '256GB SSD +  256GB SSD',
       '64GB Flash Storage', '32GB Flash Storage', '256GB SSD +  1TB HDD',
       '256GB SSD +  2TB HDD', '32GB SSD', '2TB HDD', '64GB SSD',
       '1.0TB Hybrid', '512GB SSD +  1TB HDD', '1TB SSD',
       '256GB SSD +  500GB HDD', '128GB SSD +  2TB HDD',
       '512GB SSD +  512GB SSD', '16GB SSD', '16GB Flash Storage',
       '512GB SSD +  256GB SSD', '512GB SSD +  2TB HDD',
       '64GB Flash Storage +  1TB HDD', '180GB SSD', '1TB HDD +  1TB HDD',
       '32GB HDD', '1TB SSD +  1TB HDD', '?', '512GB Flash Storage',
       '128GB HDD', '240GB SSD', '8GB SSD', '508GB Hybrid', '1.0TB HDD',
       '512GB SSD +  1.0TB Hybrid', '256GB SSD +  1.0TB Hybrid'])
    Inches = st.number_input("Screen Size (Inches)", min_value=10.0, max_value=20.0, step=0.1, value=15.6)
    Weight = st.number_input("Weight (kg)", min_value=0.0, max_value=12.0, step=0.1, value=2.0)

# Input Dictionary
user_input = {
    'Company': Company,
    'TypeName': TypeName,
    'Cpu': Cpu,
    'Gpu': Gpu,
    'OpSys': OpSys,
    'ScreenResolution': ScreenResolution,
    'Memory': Memory,
    'Ram': Ram,
    'Inches ':Inches,
    'Weight': Weight
}

# Predict Button
if st.button("Predict Price"):
    input_df = pd.DataFrame([user_input])
    prediction = model_pipeline.predict(input_df)
    st.markdown(f"### 💰 Estimated Price: ₹{int(prediction[0]):,}")

st.markdown("""
<style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }

    .stSelectbox > div {
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)