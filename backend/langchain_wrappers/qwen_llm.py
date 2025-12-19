"""
Custom LangChain LLM wrapper for Qwen Vision-Language Model.
Enables seamless integration of Qwen VLM with LangChain chains.
"""

from typing import Any, Dict, Iterator, List, Optional, Union
from io import BytesIO

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk

from PIL import Image


class QwenLLM(LLM):
    """
    LangChain LLM wrapper for Qwen Vision-Language Model.
    
    Supports both text-only and multimodal (text + image) generation.
    
    Example:
        >>> llm = QwenLLM()
        >>> response = llm.invoke("What is the capital of France?")
        >>> print(response)
        
        # With images via run config
        >>> response = llm.invoke("Describe this image", config={"images": [image_bytes]})
    """
    
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    device: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 1024
    
    # Internal state (not serialized)
    _qwen_model: Any = None
    _model_loaded: bool = False
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True
        extra = 'forbid'
    
    def __init__(self, **kwargs):
        """Initialize QwenLLM wrapper."""
        super().__init__(**kwargs)
        self._qwen_model = None
        self._model_loaded = False
    
    @property
    def _llm_type(self) -> str:
        """Return identifier of LLM type."""
        return "qwen-vlm"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
        }
    
    def _load_model(self) -> None:
        """Lazy load the Qwen model."""
        if not self._model_loaded:
            from models.qwen_vlm import load_qwen_model
            print(f"Loading Qwen VLM model: {self.model_name}")
            self._qwen_model = load_qwen_model(prefer_vision=True)
            self._model_loaded = True
    
    def _extract_images(self, kwargs: Dict) -> Optional[List[bytes]]:
        """Extract images from kwargs or config."""
        # Check for images in various places
        images = kwargs.get('images')
        if images is None:
            config = kwargs.get('config', {})
            if isinstance(config, dict):
                images = config.get('images')
        return images
    
    def _extract_context(self, kwargs: Dict) -> Optional[str]:
        """Extract context from kwargs."""
        return kwargs.get('context')
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Execute the LLM on the given prompt.
        
        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating.
            run_manager: Callback manager for the run.
            **kwargs: Additional arguments (images, context, etc.)
            
        Returns:
            The generated text output.
        """
        self._load_model()
        
        # Extract optional parameters
        images = self._extract_images(kwargs)
        context = self._extract_context(kwargs)
        
        # Generate response
        response = self._qwen_model.generate(
            query=prompt,
            context=context,
            images=images,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        
        return response
    
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """
        Stream the LLM on the given prompt.
        
        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating.
            run_manager: Callback manager for the run.
            **kwargs: Additional arguments.
            
        Yields:
            Generated text chunks.
        """
        self._load_model()
        
        # Extract optional parameters
        images = self._extract_images(kwargs)
        context = self._extract_context(kwargs)
        
        # Stream response
        for token in self._qwen_model.generate_stream(
            query=prompt,
            context=context,
            images=images,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        ):
            chunk = GenerationChunk(text=token)
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk
    
    def is_vision_available(self) -> bool:
        """Check if vision capabilities are available."""
        self._load_model()
        return self._qwen_model.is_vision_available()
    
    def generate_with_images(
        self,
        prompt: str,
        images: Optional[List[Union[str, bytes, Image.Image]]] = None,
        context: Optional[str] = None,
    ) -> str:
        """
        Convenience method for multimodal generation.
        
        Args:
            prompt: Text prompt
            images: Optional list of images
            context: Optional context from RAG retrieval
            
        Returns:
            Generated response
        """
        self._load_model()
        return self._qwen_model.generate(
            query=prompt,
            context=context,
            images=images,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
