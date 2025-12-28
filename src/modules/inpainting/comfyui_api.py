"""
ComfyUI API Client

Provides programmatic access to ComfyUI for running inpainting workflows.
"""

import json
import uuid
import urllib.request
import urllib.parse
import websocket
import io
import numpy as np
from PIL import Image
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class ComfyUIConfig:
    """ComfyUI server configuration."""
    host: str = "127.0.0.1"
    port: int = 8188
    use_https: bool = False
    timeout: int = 300  # seconds
    
    @property
    def base_url(self) -> str:
        protocol = "https" if self.use_https else "http"
        return f"{protocol}://{self.host}:{self.port}"
    
    @property
    def ws_url(self) -> str:
        protocol = "wss" if self.use_https else "ws"
        return f"{protocol}://{self.host}:{self.port}/ws"


class ComfyUIClient:
    """
    Client for interacting with ComfyUI server.
    
    Supports:
    - Uploading images
    - Running workflows
    - Retrieving results
    - Progress monitoring via WebSocket
    """
    
    def __init__(self, config: Optional[ComfyUIConfig] = None):
        self.config = config or ComfyUIConfig()
        self.client_id = str(uuid.uuid4())
    
    def _request(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[bytes] = None,
        headers: Optional[Dict] = None
    ) -> Any:
        """Make HTTP request to ComfyUI server."""
        url = f"{self.config.base_url}/{endpoint}"
        
        req_headers = headers or {}
        if data and "Content-Type" not in req_headers:
            req_headers["Content-Type"] = "application/json"
        
        request = urllib.request.Request(
            url,
            data=data,
            headers=req_headers,
            method=method
        )
        
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                return json.loads(response.read())
        except urllib.error.URLError as e:
            logger.error(f"Request failed: {e}")
            raise ConnectionError(f"Failed to connect to ComfyUI: {e}")
    
    def queue_prompt(self, workflow: Dict) -> str:
        """
        Queue a workflow for execution.
        
        Args:
            workflow: ComfyUI workflow dictionary
            
        Returns:
            Prompt ID for tracking execution
        """
        data = json.dumps({
            "prompt": workflow,
            "client_id": self.client_id
        }).encode('utf-8')
        
        result = self._request("prompt", method="POST", data=data)
        return result.get("prompt_id")
    
    def get_history(self, prompt_id: str) -> Dict:
        """Get execution history for a prompt."""
        return self._request(f"history/{prompt_id}")
    
    def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        """
        Retrieve an image from ComfyUI.
        
        Args:
            filename: Image filename
            subfolder: Subfolder within output directory
            folder_type: "output", "input", or "temp"
            
        Returns:
            Image data as bytes
        """
        params = urllib.parse.urlencode({
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type
        })
        
        url = f"{self.config.base_url}/view?{params}"
        
        with urllib.request.urlopen(url, timeout=self.config.timeout) as response:
            return response.read()
    
    def upload_image(
        self,
        image: np.ndarray,
        name: str = "input.png",
        overwrite: bool = True
    ) -> Dict:
        """
        Upload an image to ComfyUI.
        
        Args:
            image: Image as numpy array (H, W, C) or (H, W)
            name: Filename for uploaded image
            overwrite: Whether to overwrite existing file
            
        Returns:
            Upload response with filename info
        """
        # Convert to PIL Image
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        
        if image.ndim == 2:
            pil_image = Image.fromarray(image, mode='L')
        elif image.shape[2] == 4:
            pil_image = Image.fromarray(image, mode='RGBA')
        else:
            pil_image = Image.fromarray(image, mode='RGB')
        
        # Save to buffer
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        image_data = buffer.getvalue()
        
        # Create multipart form data
        boundary = uuid.uuid4().hex
        
        body = (
            f'--{boundary}\r\n'
            f'Content-Disposition: form-data; name="image"; filename="{name}"\r\n'
            f'Content-Type: image/png\r\n\r\n'
        ).encode('utf-8')
        
        body += image_data
        body += f'\r\n--{boundary}\r\n'.encode('utf-8')
        body += f'Content-Disposition: form-data; name="overwrite"\r\n\r\n{"true" if overwrite else "false"}'.encode('utf-8')
        body += f'\r\n--{boundary}--\r\n'.encode('utf-8')
        
        headers = {
            'Content-Type': f'multipart/form-data; boundary={boundary}'
        }
        
        return self._request("upload/image", method="POST", data=body, headers=headers)
    
    def upload_mask(
        self,
        mask: np.ndarray,
        name: str = "mask.png"
    ) -> Dict:
        """Upload a mask image (converted to grayscale)."""
        # Ensure single channel
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        
        # Normalize to 0-255
        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
        
        return self.upload_image(mask, name)
    
    def wait_for_completion(
        self,
        prompt_id: str,
        callback: Optional[callable] = None,
        poll_interval: float = 0.5
    ) -> Dict:
        """
        Wait for a prompt to complete execution.
        
        Args:
            prompt_id: ID from queue_prompt
            callback: Optional callback for progress updates
            poll_interval: Seconds between status checks
            
        Returns:
            Execution history/results
        """
        try:
            # Try WebSocket for real-time updates
            return self._wait_websocket(prompt_id, callback)
        except Exception as e:
            logger.warning(f"WebSocket connection failed, falling back to polling: {e}")
            return self._wait_polling(prompt_id, callback, poll_interval)
    
    def _wait_websocket(self, prompt_id: str, callback: Optional[callable] = None) -> Dict:
        """Wait for completion using WebSocket."""
        ws = websocket.WebSocket()
        ws.connect(f"{self.config.ws_url}?clientId={self.client_id}")
        
        try:
            while True:
                message = ws.recv()
                if isinstance(message, str):
                    data = json.loads(message)
                    
                    if data.get("type") == "progress":
                        progress = data.get("data", {})
                        if callback:
                            callback(progress)
                    
                    elif data.get("type") == "executing":
                        exec_data = data.get("data", {})
                        if exec_data.get("node") is None and exec_data.get("prompt_id") == prompt_id:
                            # Execution complete
                            break
                    
                    elif data.get("type") == "execution_error":
                        error = data.get("data", {})
                        raise RuntimeError(f"Execution error: {error}")
        finally:
            ws.close()
        
        return self.get_history(prompt_id)
    
    def _wait_polling(
        self,
        prompt_id: str,
        callback: Optional[callable] = None,
        poll_interval: float = 0.5
    ) -> Dict:
        """Wait for completion by polling history endpoint."""
        start_time = time.time()
        
        while True:
            if time.time() - start_time > self.config.timeout:
                raise TimeoutError(f"Execution timed out after {self.config.timeout}s")
            
            history = self.get_history(prompt_id)
            
            if prompt_id in history:
                prompt_history = history[prompt_id]
                
                if "outputs" in prompt_history:
                    return history
            
            time.sleep(poll_interval)
    
    def get_output_images(self, history: Dict, prompt_id: str) -> List[np.ndarray]:
        """
        Extract output images from execution history.
        
        Args:
            history: History dict from wait_for_completion
            prompt_id: The prompt ID
            
        Returns:
            List of output images as numpy arrays
        """
        images = []
        
        if prompt_id not in history:
            return images
        
        outputs = history[prompt_id].get("outputs", {})
        
        for node_id, node_output in outputs.items():
            if "images" in node_output:
                for img_info in node_output["images"]:
                    img_data = self.get_image(
                        img_info["filename"],
                        img_info.get("subfolder", ""),
                        img_info.get("type", "output")
                    )
                    
                    # Convert to numpy
                    pil_image = Image.open(io.BytesIO(img_data))
                    np_image = np.array(pil_image)
                    images.append(np_image)
        
        return images


class FluxInpaintWorkflow:
    """
    Wrapper for the Flux 2 inpainting workflow.
    
    Provides a simple interface for vehicle inpainting using the
    Flux 2 model with reference images and ControlNet conditioning.
    """
    
    def __init__(
        self,
        client: Optional[ComfyUIClient] = None,
        workflow_path: Optional[str] = None
    ):
        self.client = client or ComfyUIClient()
        self.workflow_template = self._load_workflow(workflow_path)
    
    def _load_workflow(self, path: Optional[str]) -> Dict:
        """Load workflow template from file or use default."""
        if path:
            with open(path, 'r') as f:
                return json.load(f)
        else:
            # Return minimal workflow structure
            # In practice, load from your actual workflow JSON
            return self._get_default_workflow()
    
    def _get_default_workflow(self) -> Dict:
        """Get default Flux inpainting workflow structure."""
        # This is a simplified placeholder
        # You would replace this with your actual workflow
        return {
            "6": {  # CLIP Text Encode (Prompt)
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "",
                    "clip": ["38", 0]
                }
            },
            # ... other nodes would go here
        }
    
    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str,
        reference_image: Optional[np.ndarray] = None,
        negative_prompt: str = "",
        steps: int = 20,
        cfg_scale: float = 1.2,
        seed: int = -1,
        depth_map: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Run Flux inpainting.
        
        Args:
            image: Background image
            mask: Inpainting mask
            prompt: Text prompt
            reference_image: Optional reference image for vehicle
            negative_prompt: Negative prompt
            steps: Sampling steps
            cfg_scale: CFG scale
            seed: Random seed (-1 for random)
            depth_map: Optional depth map for ControlNet
            
        Returns:
            Inpainted image
        """
        # Upload images
        bg_info = self.client.upload_image(image, "background.png")
        mask_info = self.client.upload_mask(mask, "mask.png")
        
        # Build workflow
        workflow = self._build_workflow(
            background_filename=bg_info.get("name", "background.png"),
            mask_filename=mask_info.get("name", "mask.png"),
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed
        )
        
        # Upload reference if provided
        if reference_image is not None:
            ref_info = self.client.upload_image(reference_image, "reference.png")
            # Update workflow to use reference
            # (depends on your specific workflow structure)
        
        # Upload depth if provided
        if depth_map is not None:
            depth_info = self.client.upload_image(depth_map, "depth.png")
            # Update workflow for ControlNet
        
        # Execute
        prompt_id = self.client.queue_prompt(workflow)
        history = self.client.wait_for_completion(prompt_id)
        
        # Get result
        images = self.client.get_output_images(history, prompt_id)
        
        if not images:
            raise RuntimeError("No output images from workflow")
        
        return images[0]
    
    def _build_workflow(
        self,
        background_filename: str,
        mask_filename: str,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 20,
        cfg_scale: float = 1.2,
        seed: int = -1
    ) -> Dict:
        """Build workflow with given parameters."""
        workflow = json.loads(json.dumps(self.workflow_template))  # Deep copy
        
        # Update nodes with parameters
        # This depends on your specific workflow structure
        # Below is a generic example
        
        for node_id, node in workflow.items():
            if node.get("class_type") == "CLIPTextEncode":
                if "Positive" in node.get("title", "") or node.get("_meta", {}).get("title", "") == "Positive":
                    node["inputs"]["text"] = prompt
                elif "Negative" in node.get("title", ""):
                    node["inputs"]["text"] = negative_prompt
            
            elif node.get("class_type") == "KSampler":
                node["inputs"]["steps"] = steps
                node["inputs"]["cfg"] = cfg_scale
                if seed >= 0:
                    node["inputs"]["seed"] = seed
            
            elif node.get("class_type") == "LoadImage":
                # Update image filename based on node purpose
                pass
        
        return workflow


def create_inpaint_function(
    comfyui_config: Optional[ComfyUIConfig] = None,
    workflow_path: Optional[str] = None,
    default_prompt: str = "Insert vehicle naturally into the scene"
) -> callable:
    """
    Create an inpainting function compatible with ScaleAwareInpainter.
    
    Args:
        comfyui_config: ComfyUI server configuration
        workflow_path: Path to workflow JSON
        default_prompt: Default prompt if not specified
        
    Returns:
        Function with signature (image, mask, **kwargs) -> image
    """
    client = ComfyUIClient(comfyui_config)
    workflow = FluxInpaintWorkflow(client, workflow_path)
    
    def inpaint_fn(
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str = None,
        reference: np.ndarray = None,
        depth: np.ndarray = None,
        **kwargs
    ) -> np.ndarray:
        return workflow.inpaint(
            image=image,
            mask=mask,
            prompt=prompt or default_prompt,
            reference_image=reference,
            depth_map=depth,
            **kwargs
        )
    
    return inpaint_fn
