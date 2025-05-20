
from utils import *
from glob import glob
from tqdm import tqdm

model = load_sr_model("models/super-resolution/swinir.pth", device="mps")
file = "495.png"

# for file in tqdm(files):
img = prepare_image(file, "mps", "ur")
with torch.no_grad():
    out = model(img)

    to_pil = transforms.ToPILImage()
    sr = to_pil(out.clamp(0,1).cpu().squeeze(0))
    name = file.split("/")[-1]
    sr.save(f"495-up.png")