import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

st.set_page_config(page_title="Skin Lesion Classifier", layout="wide")

# ==========================================
# 1. Background Wallpaper Changer (Sidebar)
# ==========================================
st.sidebar.title("🎨 Appearance Settings")

wallpaper_options = {
    "Doctor / Clinic": "https://images.unsplash.com/photo-1622253692010-333f2da6031d?q=80&w=1920&auto=format&fit=crop",
    "Medical Laboratory": "https://images.unsplash.com/photo-1579154204601-01588f351e67?q=80&w=1920&auto=format&fit=crop",
    "Modern Dark Tech": "https://images.unsplash.com/photo-1518770660439-4636190af475?q=80&w=1920&auto=format&fit=crop",
    "Minimal Clean": "https://images.unsplash.com/photo-1507668077129-56e32842fceb?q=80&w=1920&auto=format&fit=crop"
}

selected_bg = st.sidebar.selectbox(
    "Choose Background Theme:",
    list(wallpaper_options.keys())
)

bg_url = wallpaper_options[selected_bg]

# Inject CSS for Glassmorphism & Background Styling
st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(0, 0, 0, 0.70), rgba(0, 0, 0, 0.80)), url("{bg_url}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .main-card {{
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #ffffff;
    }}
    h1, h2, h3, p, label {{
        color: #ffffff !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# 2. CNN Model Architecture
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        # conv.0 to conv.7 (Matches checkpoint perfectly)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),   # conv.0
            nn.BatchNorm2d(16),                           # conv.1
            nn.ReLU(),                                    # conv.2
            nn.MaxPool2d(2, 2),                           # conv.3
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # conv.4
            nn.BatchNorm2d(32),                           # conv.5
            nn.ReLU(),                                    # conv.6
            nn.MaxPool2d(2, 2)                            # conv.7
        )
        
        # fc.0 to fc.4 (Matches fc.1.weight and fc.4.weight perfectly)
        self.fc = nn.Sequential(
            nn.Flatten(),                                 # fc.0 (no parameters)
            nn.Linear(32 * 32 * 32, 64),                  # fc.1 (matches fc.1.weight/bias)
            nn.ReLU(),                                    # fc.2
            nn.Dropout(0.5),                              # fc.3 (matches missing fc.3 param)
            nn.Linear(64, num_classes)                    # fc.4 (matches fc.4.weight/bias)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

@st.cache_resource
def load_trained_model():
    checkpoint = torch.load("skin_cancer_cnn.pth", map_location=torch.device('cpu'))
    classes = checkpoint['classes']
    model = SimpleCNN(num_classes=len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, classes

model, classes = load_trained_model()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ==========================================
# 3. Main Interface & Detection Logic
# ==========================================
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.title("🩺 AI Skin Lesion Detection Assistant")
st.write("Upload a dermatoscopic skin image to test for Benign vs Malignant conditions.")

uploaded_file = st.file_uploader("Upload Lesion Image (JPG, PNG, JPEG):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
    with col2:
        with st.spinner("Analyzing image..."):
            tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                top_prob, top_idx = torch.topk(probs, 1)

            predicted_class = classes[top_idx.item()]
            confidence = top_prob.item() * 100

            st.write("### Diagnostic Result")
            
            if predicted_class == "Not_Skin_Lesion":
                st.error("⚠️ **Invalid Image Detected!**")
                st.warning("Yeh image skin lesion ki nahi lag rahi. Barah-e-karam skin spot/mole ki clear dermatoscopic picture upload karein.")
            else:
                if predicted_class == "Malignant":
                    st.error(f"⚠️ Prediction: **{predicted_class}** (Cancerous risk detected)")
                else:
                    st.success(f"✅ Prediction: **{predicted_class}** (Non-cancerous / Benign)")
                    
                st.info(f"Model Confidence: **{confidence:.2f}%**")
                
                with st.expander("View Probabilities Breakdown"):
                    for idx, cls_name in enumerate(classes):
                        st.write(f"- **{cls_name}:** {probs[idx].item() * 100:.2f}%")
                        
                st.caption("Note: This tool is for educational purposes only. Always consult a certified dermatologist.")

st.markdown('</div>', unsafe_allow_html=True)
