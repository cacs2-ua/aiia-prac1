
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
# Set up credentials and client
endpoint = "https://aiia-ia.services.ai.azure.com/"
key = "4f4OExuyPq3BK2Zz0dc7QPiMSkpTHWFoQCsqkGPlq0ZTdOSbXXr1JQQJ99BJAC5T7U2XJ3w3AAAAACOGI6zB"
# Create an Image Analysis client for synchronous operations
client = ImageAnalysisClient(
 endpoint=endpoint,
 credential=AzureKeyCredential(key)
)
# Load image to analyze into a 'bytes' object
with open("frame.jpg", "rb") as f:
 image_data = f.read()
# Analyze image with PEOPLE feature
result = client.analyze(
 image_data=image_data,
 visual_features=[VisualFeatures.PEOPLE],
 language="en"
)
if result.people is not None:
    for person in result.people.list:
        confidence = person.confidence
        if confidence > 0.5:
            bounding_box = person.bounding_box
            print("Confidence:", confidence)
            print("Bounding Box:", bounding_box)
            print("Additional Information if available:")
            print()
else:
    print("Analysis failed.")
    print(" Error reason:", result.error.reason)
    print(" Error code:", result.error.code)
    print(" Error message:", result.error.message)