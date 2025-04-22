#Unified Frequency-Aware Knowledge Distillation for Fast Text-Guided Image-to-Image Translation
This is the code of the paper "Unified Frequency-Aware Knowledge Distillation for Fast
Text-Guided Image-to-Image Translation" 

# Citation #


# Introduction
 <br>

The overall model architecture, please refer to the paper (coming soon) for more technical details.


# Environment
First, create a new conda virtual environment: <br>
<pre><code>
conda create -n FCDiffusion python=3.8
</code></pre>
Then, install pytorch related packages using conda: <br>
<pre><code>
conda activate FCDiffusion
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
</code></pre>
Last, install the required packages in the requirements.txt: <br>
<pre><code>
pip install -r requirements
</code></pre>

# Dataset
Since we do not train the large-scale latent diffusion model (LDM) from scratch but rather train a frequency-based control network of the pre-trained LDM, a small subset of LAION 5B is sufficient for our task. Therefore, we use **LAION Aesthetics 6.5+** which comprises 625K image-text pairs as the training set of our model. Download and put it in the **datasets** folder of the project as shown below:
<div style="padding-left: 4%; padding-right: 4%;">
                <div align="center">
                    <img src="img/training_set.png" width="70%"> <br>
		</div>
</div>

Then, run the Python script **training_data_prepare.py** to create the json file of the training dataset:
<pre><code>
python training_data_prepare.py
</code></pre>
A json file **training_data.json** wil be created under the **datasets** folder. It records the image path and the text prompt of each image-text pair of the training set, and is used in the training process.

# Download the required model
Our model is based on the pretrained text-to-image latent diffusion model. Specifically, we use **Stable Diffusion v2-1-base** model in our method. Download the model checkpoint file **v2-1_512-ema-pruned.ckpt** [here](https://huggingface.co/stabilityai/stable-diffusion-2-1-base/tree/main) and put it in the **models** folder of the project. Then, run the Python script **tool_add_control_sd21.py** to create our initialized model: 
<pre><code>
python tool_add_control_sd21.py ./models/v2-1_512-ema-pruned.ckpt ./models/FCDiffusion_ini.ckpt
</code></pre>
This script will create a ckpt file of our model with the parameters initialized from the pretrained Stable Diffusion v2-1-base. The created ckpt file named **FCDiffusion_ini.ckpt** will be in the **models** folder of the project, as shown below:
<div style="padding-left: 4%; padding-right: 4%;">
                <div align="center">
                    <img src="img/ckpt_file.png" width="70%"> <br>
		</div>
</div>
The training of the model will be started from the generated FCDiffusion_ini.ckpt. <br>

Besides, our method uses the pretrained OpenCLIP text encoder, download the **open_clip_pytorch_model.bin** file [here](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main) and put it in the **CLIP-ViT-H-14** folder of the project, as shown below:
<div style="padding-left: 4%; padding-right: 4%;">
                <div align="center">
                    <img src="img/open_clip_model.png" width="70%"> <br>
		</div>
</div>




# Teacher Model training
Before training, set the **control_mode** parameter in the model_config.yaml configuration file. The parameter must be one of "**mini_pass**", "**low_pass**", "**mid_pass**", and "**high_pass**". <be>
- The "mini_pass" mode realizes style-guided content creation with mini-frequency control. 
- The "low_pass" mode realizes image semantic manipulation with low-frequency control.
- The "mid_pass" mode realizes image scene translation with mid-frequency control.
- The "high-pass" mode realizes image style translation with high-frequency control.

Then, run the Python script  to start training single teacher model directly:
<pre><code>
python fcdiffusion_train.py
</code></pre>

Then, run the Python script  to start training multi-distillation model directly:
<pre><code>
python fcdiffusion_distill.py
</code></pre>

# Parameter setup
1 set the batch_size to fit for your machine.
2 set the path of teacher model ckpt.

# Model inference
Inference model for text-driven image-to-image translation by running the Python script :
<pre><code>
python test_distillation.py
</code></pre>

# Results display
<!-- <div style="padding-left: 4%; padding-right: 4%;">
                <div align="center">
                    <img src="img/style_guided_content_creation.jpg" width="80%"> <br>
                </div>
            <p style="line-height:180%">Figure 2. Results of style-guided content creation realized with mini-frequency control. The image content is recreated according to the text prompt while the style of the translated image is transferred from the source image.
	    </p>
	    </div>      
<br>

<div style="padding-left: 4%; padding-right: 4%;">
                <div align="center">
                    <img src="img/image semantic manipulation.jpg" width="80%"> <br>
                </div>
            <p style="line-height:180%">Figure 3. Results of image semantic manipulation realized with low-frequency control. The semantics of the source image is manipulated according to the text prompt while the image style and spatial structure are maintained.
	    </p>
	    </div>      
<br>

<div style="padding-left: 4%; padding-right: 4%;">
                <div align="center">
                    <img src="img/image style translation.jpg" width="80%"> <br>
                </div>
            <p style="line-height:180%">Figure 4. Results of image style translation realized with high-frequency control. The image style (appearance) is modified as per the text prompt while the main contours of the source image are preserved.
	    </p>
	    </div>      
<br>

<div style="padding-left: 4%; padding-right: 4%;">
                <div align="center">
                    <img src="img/image scene translation.jpg" width="80%"> <br>
                </div>
            <p style="line-height:180%">Figure 5. Results of image scene translation realized with mid-frequency control. The image scene is translated according to the text prompt. In this scenario, the layout of the source image is preserved while the lower-frequency image style and higher-frequency image contours are not restricted.
	    </p>
	    </div>       -->


