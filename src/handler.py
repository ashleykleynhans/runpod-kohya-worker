'''
Handler for the generation of a fine tuned lora model.
'''

import os
import shutil
import subprocess
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import rp_download, upload_file_to_bucket

from rp_schema import INPUT_SCHEMA


def handler(job):
    # Clear content directories from previous runs
    shutil.rmtree('./training', ignore_errors=True)

    job_input = job['input']

    if 'errors' in (job_input := validate(job_input, INPUT_SCHEMA)):
        return {'error': job_input['errors']}
    job_input = job_input['validated_input']

    # Download the zip file
    downloaded_input = rp_download.file(job_input['zip_url'])

    steps = 520
    instance_name = "sks"

    if not os.path.exists('./training'):
        os.mkdir('./training')

    os.mkdir('./training/img')
    os.mkdir(f"./training/img/{steps}_{instance_name} {job_input['class_name']}")
    os.mkdir('./training/reg')
    os.mkdir(f"./training/reg/1_{job_input['class_name']}")
    os.mkdir('./training/model')
    os.mkdir('./training/logs')

    reg_path = './reg/woman'

    if os.path.exists(reg_path):
        for root, dirs, files in os.walk(reg_path):
            for file in files:
                src_file_path = os.path.join(root, file)
                shutil.copy(src_file_path, f"./training/reg/1_{job_input['class_name']}")

    # Make clean data directory
    allowed_extensions = [".jpg", ".jpeg", ".png"]
    flat_directory = f"./training/img/{steps}_{instance_name} {job_input['class_name']}"
    os.makedirs(flat_directory, exist_ok=True)

    for root, dirs, files in os.walk(downloaded_input['extracted_path']):
        # Skip __MACOSX folder
        if '__MACOSX' in root:
            continue

        for file in files:
            file_path = os.path.join(root, file)
            if os.path.splitext(file_path)[1].lower() in allowed_extensions:
                shutil.copy(
                    os.path.join(downloaded_input['extracted_path'], file_path),
                    flat_directory
                )

    subprocess.run(f"""accelerate launch --num_cpu_threads_per_process 1 --num_processes 1 --num_machines 1 --mixed_precision bf16 train_db.py \
            --bucket_no_upscale \
            --bucket_reso_steps=64 \
            --cache_latents \
            --cache_latents_to_disk \
            --clip_skip=1 \
            --enable_bucket \
            --gradient_accumulation_steps=1 \
            --gradient_checkpointing \
            --huber_c=0.1 \
            --huber_schedule="snr" \
            --learning_rate=7e-7 \
            --learning_rate_te=7e-7 \
            --logging_dir="./training/log/" \
            --loss_type="l2" \
            --lr_scheduler="constant" \
            --lr_scheduler_num_cycles=1 \
            --lr_scheduler_power=1 \
            --max_bucket_reso=2048 \
            --max_data_loader_n_workers=0 \
            --max_timestep=1000 \
            --max_token_length=150 \
            --max_train_steps=10400 \
            --min_bucket_reso=256 \
            --optimizer_args scale_parameter=False relative_step=False warmup_init=False weight_decay=0.01 \
            --optimizer_type="Adafactor" \
            --output_dir="./training/model" \
            --output_name="tier_1_quality_slow" \
            --pretrained_model_name_or_path="/model_cache/hyperRealism_30.safetensors" \
            --prior_loss_weight=1 \
            --reg_data_dir="./training/reg" \
            --resolution="768,768" \
            --save_every_n_epochs=1 \
            --save_every_n_steps=1000 \
            --save_model_as="safetensors" \
            --save_precision="float" \
            --train_data_dir="./training/img" \
            --train_batch_size=1""", shell=True, check=True)
    
    model_directory = "./training/model"
    for file_name in os.listdir(model_directory):
        file_path = os.path.join(model_directory, file_name)
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path)
            print(f"File: {file_name}, Size: {file_size} bytes")

    job_s3_config = job.get('s3Config')

    if job_s3_config is not None:
        uploaded_url = upload_file_to_bucket(
            file_name=f"{job['id']}.safetensors",
            file_location=f"./training/model/{job['id']}.safetensors",
            bucket_creds=job_s3_config,
            bucket_name=job_s3_config['bucketName'],
        )
        return {"url": uploaded_url}

    return {"url": "ok"}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
