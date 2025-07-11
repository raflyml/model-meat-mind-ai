import os
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from PIL import Image

# ============ CLASS NAMES ============
FOOD_CLASS_NAMES = [
    "Apple pie", "Baby back ribs", "Baklava", "Beef carpaccio", "Beef tartare",
    "Beet salad", "Beignets", "Bibimbap", "Bread pudding", "Breakfast burrito",
    "Bruschetta", "Caesar salad", "Cannoli", "Caprese salad", "Carrot cake",
    "Ceviche", "Cheesecake", "Cheese plate", "Chicken curry", "Chicken quesadilla",
    "Chicken wings", "Chocolate cake", "Chocolate mousse", "Churros", "Clam chowder",
    "Club sandwich", "Crab cakes", "Creme brulee", "Croque madame", "Cup cakes",
    "Deviled eggs", "Donuts", "Dumplings", "Edamame", "Eggs benedict",
    "Escargots", "Falafel", "Filet mignon", "Fish and chips", "Foie gras",
    "French fries", "French onion soup", "French toast", "Fried calamari", "Fried rice",
    "Frozen yogurt", "Garlic bread", "Gnocchi", "Greek salad", "Grilled cheese sandwich",
    "Grilled salmon", "Guacamole", "Gyoza", "Hamburger", "Hot and sour soup",
    "Hot dog", "Huevos rancheros", "Hummus", "Ice cream", "Lasagna",
    "Lobster bisque", "Lobster roll sandwich", "Macaroni and cheese", "Macarons", "Miso soup",
    "Mussels", "Nachos", "Omelette", "Onion rings", "Oysters",
    "Pad thai", "Paella", "Pancakes", "Panna cotta", "Peking duck",
    "Pho", "Pizza", "Pork chop", "Poutine", "Prime rib",
    "Pulled pork sandwich", "Ramen", "Ravioli", "Red velvet cake", "Risotto",
    "Samosa", "Sashimi", "Scallops", "Seaweed salad", "Shrimp and grits",
    "Spaghetti bolognese", "Spaghetti carbonara", "Spring rolls", "Steak", "Strawberry shortcake",
    "Sushi", "Tacos", "Takoyaki", "Tiramisu", "Tuna tartare",
    "Waffles",
    # Indonesian food classes:
    "Ayam bakar", "Ayam goreng", "Bakso", "Bakwan", "Batagor", "Bihun", "Capcay", "Gado-gado",
    "Ikan goreng", "Kerupuk", "Martabak telor", "Mie", "Nasi goreng", "Nasi putih", "Nugget",
    "Opor ayam", "Pempek", "Rendang", "Roti", "Soto", "Steak", "Tahu", "Telur", "Tempe",
    "Terong balado", "Tumis kangkung", "Udang", "Sate", "Sosis"
]

FRUIT_CLASS_NAMES = [
    "Apple", "Apricot", "Avocado", "Banana", "Black Berry", "Blueberry", "Cherry", "Coconut",
    "Cranberry", "Dragonfruit", "Durian", "Grape", "Grapefruit", "Guava", "Jackfruit", "Kiwi",
    "Lemon", "Lime", "Lychee", "Mango", "Mangosteen", "Melon Pear", "Olive", "Orange", "Papaya",
    "Passion Fruit", "Raspberry", "Salak", "Sapodilla", "Strawberry", "Tomato", "Watermelon"
]

# ============ FASTAPI INIT ============
app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ LOAD MODELS ============
# Jika model di root folder:
FOOD_MODEL_PATH = "mobilenetv3_food41.keras"
FRUIT_MODEL_PATH = "final_fruit_mobilenetv3.keras"

# Jika model di dalam folder models:
# FOOD_MODEL_PATH = os.path.join("models", "mobilenetv3_food41.keras")
# FRUIT_MODEL_PATH = os.path.join("models", "final_fruit_mobilenetv3.keras")

print("Loading food model...")
food_model = load_model(FOOD_MODEL_PATH, compile=False)
print("Loading fruit model...")
fruit_model = load_model(FRUIT_MODEL_PATH, compile=False)

print("Model loaded.")

# ============ PREDICT FUNCTIONS ============
def predict_image(image_bytes, model, class_names):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image).astype("float32")
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        idx = int(np.argmax(pred[0]))
        conf = float(np.max(pred[0]))
        label = class_names[idx]
        return label, conf
    except Exception as e:
        print(f"Error in predict_image: {e}")
        return None, None

# ============ ENDPOINTS ============
@app.post("/predict/food")
async def predict_food(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        label, conf = predict_image(image_bytes, food_model, FOOD_CLASS_NAMES)
        if label is None:
            return {"error": "Failed to process image."}
        return {
            "class": label,
            "confidence": conf
        }
    except Exception as e:
        print(f"Error in predict_food endpoint: {e}")
        return {"error": str(e)}

@app.post("/predict/fruit")
async def predict_fruit(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        label, conf = predict_image(image_bytes, fruit_model, FRUIT_CLASS_NAMES)
        if label is None:
            return {"error": "Failed to process image."}
        return {
            "class": label,
            "confidence": conf
        }
    except Exception as e:
        print(f"Error in predict_fruit endpoint: {e}")
        return {"error": str(e)}

@app.get("/")
def root():
    return {
        "message": "MealMind API running!",
        "food_model": FOOD_MODEL_PATH,
        "fruit_model": FRUIT_MODEL_PATH,
        "endpoints": ["/predict/food", "/predict/fruit"]
    }