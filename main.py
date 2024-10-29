from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torchvision.transforms as transforms
import io

# Load the trained model
model = LogisticRegression(input_dim=28*28, output_dim=10)  # Ensure to match your model definition
model.load_state_dict(torch.load("path_to_your_model_weights.pth"))  # Replace with the path to your saved model weights
model.eval()  # Set the model to evaluation mode

# Initialize FastAPI
app = FastAPI()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize to 28x28
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize as done during training
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Load the image from the request
    image = Image.open(io.BytesIO(await file.read())).convert('L')  # Convert to grayscale

    # Preprocess the image
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(image)  # Forward pass
        _, predicted = torch.max(output.data, 1)  # Get the predicted class

    # Return the prediction
    return {"predicted_class": predicted.item()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

