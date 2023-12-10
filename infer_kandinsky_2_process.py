import copy
from ikomia import core, dataprocess, utils
import torch
import numpy as np
import random
from diffusers import AutoPipelineForText2Image
import os

# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferKandinsky2Param(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        self.model_name = "kandinsky-community/kandinsky-2-2-decoder"
        self.prompt = "portrait of a young women, blue eyes, cinematic"
        self.cuda = torch.cuda.is_available()
        self.guidance_scale = 1.0
        self.prior_guidance_scale = 4.0
        self.negative_prompt = "low quality, bad quality"
        self.height = 768
        self.width = 768
        self.num_inference_steps = 100
        self.prior_num_inference_steps = 25
        self.seed = -1
        self.update = False

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.model_name = str(param_map["model_name"])
        self.prompt = str(param_map["prompt"])
        self.cuda = utils.strtobool(param_map["cuda"])
        self.prior_guidance_scale = float(param_map["prior_guidance_scale"])
        self.guidance_scale = float(param_map["guidance_scale"])
        self.negative_prompt = str(param_map["negative_prompt"])
        self.seed = int(param_map["seed"])
        self.height = int(param_map["height"])
        self.width = int(param_map["width"])
        self.num_inference_steps = int(param_map["num_inference_steps"])
        self.prior_num_inference_steps = int(param_map["prior_num_inference_steps"])
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        param_map["model_name"] = str(self.model_name)
        param_map["prompt"] = str(self.prompt)
        param_map["cuda"] = str(self.cuda)
        param_map["guidance_scale"] = str(self.guidance_scale)
        param_map["prior_guidance_scale"] = str(self.prior_guidance_scale)
        param_map["negative_prompt"] = str(self.negative_prompt)
        param_map["height"] = str(self.height)
        param_map["width"] = str(self.width)
        param_map["num_inference_steps"] = str(self.num_inference_steps)
        param_map["prior_num_inference_steps"] = str(self.prior_num_inference_steps)
        param_map["seed"] = str(self.seed)

        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferKandinsky2(dataprocess.CWorkflowTask):

    def __init__(self, name, param):
        core.CWorkflowTask.__init__(self, name)
        # Add input/output of the algorithm here
        self.add_output(dataprocess.CImageIO())
        # Create parameters object
        if param is None:
            self.set_param_object(InferKandinsky2Param())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.device = torch.device("cpu")
        self.pipe = None
        self.generator = None
        self.seed = None
        self.model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights")

    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def run(self):
        # Main function of your algorithm
        # Call begin_task_run() for initialization
        self.begin_task_run()

        # Get parameters
        param = self.get_param_object()

        # Load pipeline
        if param.update or self.pipe is None:
            self.device = torch.device("cuda") if param.cuda and torch.cuda.is_available() else torch.device("cpu")
            torch_tensor_dtype = torch.float16 if param.cuda and torch.cuda.is_available() else torch.float32
            # Load model weight
            try:
                self.pipe = AutoPipelineForText2Image.from_pretrained(
                    param.model_name,
                    torch_dtype=torch_tensor_dtype,
                    use_safetensors=True,
                    cache_dir=self.model_folder,
                    local_files_only=True
                    )
            except Exception as e:
                print(f"Failed with error: {e}. Trying without the local_files_only parameter...")
                self.pipe = AutoPipelineForText2Image.from_pretrained(
                    param.model_name,
                    torch_dtype=torch_tensor_dtype,
                    use_safetensors=True,
                    cache_dir=self.model_folder
                )
            # Pipe to device
            self.pipe.enable_model_cpu_offload()

            # Generate seed
            if param.seed == -1:
                self.seed = random.randint(0, 191965535)
            else:
                self.seed = param.seed

            self.generator = torch.Generator(self.device).manual_seed(self.seed)

        with torch.no_grad():
            result = self.pipe(prompt=param.prompt,
                          negative_prompt=param.negative_prompt,
                          guidance_scale=param.guidance_scale,
                          height=param.height,
                          width=param.width,
                          generator=self.generator,
                          num_inference_steps = param.num_inference_steps
                          ).images[0]

        print(f"Prompt:\t{param.prompt}\nSeed:\t{self.seed}")

        # Get and display output
        image = np.array(result)
        output_img = self.get_output(0)
        output_img.set_image(image)

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferKandinsky2Factory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_kandinsky_2"
        self.info.short_description = "Kandinsky 2.2 text2image diffusion model."
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.path = "Plugins/Python/Diffusion"
        self.info.version = "1.0.0"
        self.info.icon_path = "images/einstein.jpg"
        self.info.authors = "A. Shakhmatov, A. Razzhigaev, A. Nikolich, V. Arkhipkin, I. Pavlov, A. Kuznetsov, D. Dimitrov"
        self.info.article = "https://aclanthology.org/2023.emnlp-demo.25/"
        self.info.journal = "ACL Anthology"
        self.info.year = 2023
        self.info.license = "Apache 2.0 License"
        # URL of documentation
        self.info.documentation_link = "https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder"
        # Code source repository
        self.info.repository = "https://github.com/ai-forever/Kandinsky-2"
        # Keywords used for search
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "IMAGE_GENERATION"
        self.info.keywords = "Latent Diffusion,Hugging Face,Kandinsky,text2image,Generative"

    def create(self, param=None):
        # Create algorithm object
        return InferKandinsky2(self.info.name, param)
