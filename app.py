import gradio as gr
!npm install -g localtunnel
!pip3 install triton
vae_args = ""
 
root_dir = "/content"

%cd {root_dir}

def get_hypernetworks():
  !mkdir -p {root_dir}/stable-diffusion-webui/models/hypernetworks
  hypernetworks = ['anime_2.pt', 'anime.pt', 'anime_3.pt', 'furry_2.pt', 'furry_3.pt', 'furry_protogen.pt', 'furry_transformation.pt', 'furry_scalie.pt', 'pony.pt', 'aini.pt', 'furry.pt', 'furry_kemono.pt']
  for network in hypernetworks:
    !wget -c https://huggingface.co/acheong08/secretAI/resolve/main/stableckpt/modules/modules/{network} -O {root_dir}/stable-diffusion-webui/models/hypernetworks/{network}

def custom_model(url, checkpoint_name):
  user_token = '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'
  user_header = f"\"Authorization: Bearer {user_token}\""
  !wget -c --header={user_header} {url} -O {root_dir}/stable-diffusion-webui/models/Stable-diffusion/{checkpoint_name}.ckpt

def install_deps():
  %cd {root_dir}
  !git clone https://github.com/acheong08/stable-diffusion-webui
  #@markdown Choose the models you want
  use_hypernetworks = True #@param {'type':'boolean'}
  NovelAI = True #@param {'type':'boolean'}
  Stable_Diffusion = True #@param {'type':'boolean'}
  Waifu_Diffusion = True #@param {'type':'boolean'}
  H_Diffusion = True #@param {'type':'boolean'}
  SD_V1_5 = False #@param {'type':'boolean'}
  SD_v1_5_inpainting = False #@param {'type':'boolean'}
  if NovelAI:
    custom_model("https://huggingface.co/acheong08/secretAI/resolve/main/stableckpt/animefull-final-pruned/model.ckpt", "novelAI")
  if Stable_Diffusion:
    custom_model("https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt", "stable_diffusion")
  if Waifu_Diffusion:
    custom_model("https://huggingface.co/hakurei/waifu-diffusion-v1-3/resolve/main/wd-v1-3-float32.ckpt", "waifu_diffusion")
  if H_Diffusion:
    custom_model("https://huggingface.co/Deltaadams/Hentai-Diffusion/resolve/main/HD-16.ckpt", "h_diffusion")
  if SD_V1_5:
    custom_model("https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt", "sd-v1-5")
  if SD_v1_5_inpainting:
    custom_model("https://huggingface.co/runwayml/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.ckpt", "sd-v1-5-inpainting")
  if use_hypernetworks:
    get_hypernetworks()
    
  %cd {root_dir}/stable-diffusion-webui/extensions
  !git clone https://github.com/yfszzx/stable-diffusion-webui-images-browser
  %cd {root_dir}

def install_xformers():
  s = getoutput('nvidia-smi')
  if 'T4' in s:
    gpu = 'T4'
  elif 'P100' in s:
    gpu = 'P100'
  elif 'V100' in s:
    gpu = 'V100'
  elif 'A100' in s:
    gpu = 'A100'
  else:
    print("Your GPU sucks...")
    exit()

  if (gpu=='T4'):
    %pip install -q https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/T4/xformers-0.0.13.dev0-py3-none-any.whl
    
  elif (gpu=='P100'):
    %pip install -q https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/P100/xformers-0.0.13.dev0-py3-none-any.whl

  elif (gpu=='V100'):
    %pip install -q https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/V100/xformers-0.0.13.dev0-py3-none-any.whl

  elif (gpu=='A100'):
    %pip install -q https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/A100/xformers-0.0.13.dev0-py3-none-any.whl

def run_webui():
  animeVae = True #@param {'type':'boolean'}
  SDVae = False #@param {'type':'boolean'}
  if animeVae:
    !wget -c https://huggingface.co/acheong08/secretAI/resolve/main/stableckpt/animevae.pt -O {root_dir}/stable-diffusion-webui/models/Stable-diffusion/novelAI.vae.pt
    vae_args = "--vae-path " + root_dir + "/stable-diffusion-webui/models/Stable-diffusion/novelAI.vae.pt"
  if SDVae:
    !wget -c https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin -O {root_dir}/stable-diffusion-webui/models/Stable-diffusion/sd-v1-5.vae.pt
    vae_args = "--vae-path " + root_dir + "/stable-diffusion-webui/models/Stable-diffusion/sd-v1-5.vae.pt"

  %cd {root_dir}/stable-diffusion-webui/
  vram = "--medvram" #@param ["--medvram", "--lowvram", ""]
  other_args = "--share" #@param ["--share", ""]
  gradio_username = "webui" #@param {'type': 'string'}
  gradio_password = "diffusion" #@param {'type': 'string'}
  #use_xformers = False #@param {'type':'boolean'}
  #if use_xformers:
  #  other_args = "--xformers --share"
  #  install_xformers()
  !COMMANDLINE_ARGS="{other_args} {vae_args} {vram} --gradio-auth {gradio_username}:{gradio_password}" REQS_FILE="requirements.txt" python launch.py

install_deps()
run_webui()
#@markdown # Common issues
#@markdown ### Why am I getting low quality limages?
#@markdown NovelAI uses a different system for interpreting prompts than Stable Diffusion. Check out gelbooru.com's tags (NSFW)