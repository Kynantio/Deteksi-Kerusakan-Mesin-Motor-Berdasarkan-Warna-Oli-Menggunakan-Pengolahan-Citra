import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from src.model import ResNet50Transfer  # Pastikan path ini sesuai

# Konfigurasi halaman
st.set_page_config(
    page_title="Klasifikasi Jenis Oli",
    page_icon="🛢️",
    layout="wide"
)

st.title("🛢️ Sistem Klasifikasi Jenis Oli Mesin")
st.markdown("Upload gambar oli (belum dicrop), sistem akan otomatis crop sebelum klasifikasi.")

# Label mapping
label_map = {
    0: "🟢 Oli Baru",
    1: "⚫ Oli Bekas",
    2: "💧 Oli Tercampur Air",
    3: "🧪 Oli Tercampur Radiator"
}

# Load model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50Transfer(num_classes=4)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# Fungsi crop otomatis (berdasarkan posisi tetap)
def auto_crop(image):
    width, height = image.size
    # Koordinat crop, sesuaikan jika perlu
    left = int(width * 0.25)
    top = int(height * 0.35)
    right = int(width * 0.75)
    bottom = int(height * 0.80)
    return image.crop((left, top, right, bottom))

# Preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image.convert("RGB")).unsqueeze(0)

# Upload gambar
uploaded_file = st.file_uploader("📤 Pilih gambar oli...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    original_image = Image.open(uploaded_file).convert("RGB")
    st.image(original_image, caption="📸 Gambar Asli", use_column_width=False)

    # Crop otomatis
    cropped_image = auto_crop(original_image)
    st.image(cropped_image, caption="✂️ Gambar Hasil Crop", use_column_width=False)

    st.markdown("🔎 *Sedang menganalisis...*")

    # Preprocess dan prediksi
    input_tensor = preprocess_image(cropped_image).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        label_index = predicted.item()
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][label_index].item() * 100

    st.success(f"**Hasil Prediksi: {label_map[label_index]}**")
    st.markdown(f"📊 **Tingkat Keyakinan: {confidence:.2f}%**")

    # Penjelasan
    st.markdown("---")
    st.markdown("### Keterangan Label:")
    st.markdown("- 🟢 **Oli Baru** → Bersih & transparan")
    st.markdown("- ⚫ **Oli Bekas** → Hitam pekat karena hasil pembakaran")
    st.markdown("- 💧 **Tercampur Air** → Keruh, pucat, atau berbusa")
    st.markdown("- 🧪 **Tercampur Coolant** → Milky/cloudy, kehijauan atau putih")

else:
    st.info("📥 Silakan unggah gambar terlebih dahulu.")
