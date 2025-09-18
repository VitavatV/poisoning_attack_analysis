import os
from PIL import Image, ImageDraw, ImageFont

def generate_gif_from_folder(folder_path, output_path, duration=400, loop=0):
    # Get all image files in the folder (supports common formats)
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
    files.sort(key=lambda f: os.path.getctime(os.path.join(folder_path, f)))  # Sort files by creation time

    # Load images and add filename as text
    images = []
    for fname in files:
        img_path = os.path.join(folder_path, fname)
        img = Image.open(img_path).convert("RGBA")
        draw = ImageDraw.Draw(img)
        # Use a default font
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        text = fname
        # Draw a semi-transparent rectangle behind the text for readability
        try:
            # Pillow >= 8.0.0
            text_bbox = draw.textbbox((0, 0), text, font=font)
            rect_width = text_bbox[2] - text_bbox[0] + 8
            rect_height = text_bbox[3] - text_bbox[1] + 8
        except AttributeError:
            # Older Pillow
            text_size = font.getsize(text)
            rect_width = text_size[0] + 8
            rect_height = text_size[1] + 8
        draw.rectangle(
            [(0, 0), (rect_width, rect_height)],
            fill=(0, 0, 0, 128)
        )
        draw.text((4, 4), text, font=font, fill=(255, 255, 255, 255))
        images.append(img.convert("P"))  # Convert back to palette mode for GIF

    if not images:
        raise ValueError("No images found in the folder.")

    # Save as GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop
    )

if __name__ == "__main__":
    job_list = ['poisoned_0', 'poisoned_50', 'poisoned_100', 'all']
    for job in job_list:
        folder = os.path.join(r"D:\github\poisoning_attack_analysis\results\ML101_linear_CL_batch\uniform", job)
        output_gif = folder + ".gif"
        generate_gif_from_folder(folder, output_gif)
        print(f"GIF saved to {output_gif}")