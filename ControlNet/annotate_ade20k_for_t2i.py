import numpy as np
import pandas as pd
import torch
from PIL import Image
import skimage
from transformers import AutoProcessor, AutoModelForCausalLM
from tqdm.auto import tqdm
from pathlib import Path
import glob
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--in_path", type=str)
parser.add_argument("--out_path", type=str)
parser.add_argument("--model_type", type=str, default='base', choices=['base', 'large'])
parser.add_argument("--batch_size", type=int, default=4)

if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    batch_size = args.batch_size

    florence_model = AutoModelForCausalLM.from_pretrained(f"microsoft/Florence-2-{args.model_type}", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    florence_processor = AutoProcessor.from_pretrained(f"microsoft/Florence-2-{args.model_type}", trust_remote_code=True)
    task_prompt = "<CAPTION>"

    imagenet_caption_metadata_path = Path(args.out_path)
    imagenet_caption_metadata_temp_path = Path(f"{args.out_path}.temp")

    if imagenet_caption_metadata_path.exists():
        df = pd.read_csv(imagenet_caption_metadata_path)
        start_idx = len(df[df.caption.notna()])
        image_files = df[df.caption.isna()].path.to_list()
        print(f"Resuming from {imagenet_caption_metadata_path.as_posix()}, [{start_idx}/{len(df)}] already performed.")
    else:
        start_idx = 0
        image_files = list(glob.glob(f"{args.in_path}/*.jpg"))
        df = pd.DataFrame({"path": image_files})
        df['caption'] = np.nan
        df['height'] = np.nan
        df['width'] = np.nan



    for batch_idx in tqdm(range(0, len(image_files), batch_size)):
        batch_files = image_files[batch_idx:min(batch_idx + batch_size, len(image_files))]
        images = [Image.fromarray(skimage.io.imread(file)).convert('RGB') for file in batch_files]
        prompts = [task_prompt]*len(batch_files)
        inputs = florence_processor(text=prompts, images=images, return_tensors="pt").to(device, torch_dtype)

        generated_ids = florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
        )

        for i in range(len(batch_files)):
            generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[i]

            parsed_answer = florence_processor.post_process_generation(generated_text, task=task_prompt, image_size=(images[i].width, images[i].height))

            df.loc[start_idx+i+batch_idx, "caption"] = parsed_answer[task_prompt].replace("<pad>", "").strip()
            df.loc[start_idx+i+batch_idx, "height"] = images[i].height
            df.loc[start_idx+i+batch_idx, "width"] = images[i].width
        
        try:
            df.to_csv(imagenet_caption_metadata_temp_path, index=False)

            imagenet_caption_metadata_temp_path.rename(imagenet_caption_metadata_path)
        finally:
            imagenet_caption_metadata_temp_path.unlink(missing_ok=True)
        