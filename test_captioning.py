from PIL import Image
from captioning import ImageCaptioner


# Load captioning model
captioner = ImageCaptioner()


# Load test image
image = Image.open("test.jpg")


# Generate caption
caption = captioner.generate_caption(image)


print("\nCaption:")
print(caption)