"""
Multimodal embedding module using CLIP for aligned text and image embeddings.
Creates embeddings in the same vector space for both text and images.
"""

import os
from typing import List, Optional, Union, Tuple
from pathlib import Path
import numpy as np

import torch
from PIL import Image

# Try to import open_clip, fallback gracefully
try:
    import open_clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: open-clip-torch not installed. Run: pip install open-clip-torch")


class MultimodalEmbedder:
    """
    Multimodal embedding model using CLIP.
    Creates aligned embeddings for both text and images in the same vector space.
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: Optional[str] = None
    ):
        """
        Initialize CLIP-based multimodal embedder.
        
        Args:
            model_name: CLIP model architecture (e.g., 'ViT-B-32', 'ViT-L-14')
            pretrained: Pretrained weights to use
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        if not CLIP_AVAILABLE:
            raise ImportError("open-clip-torch is required. Install with: pip install open-clip-torch")
        
        self.model_name = model_name
        self.pretrained = pretrained
        
        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Loading CLIP model '{model_name}' ({pretrained}) on {self.device}...")
        
        # Load CLIP model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device
        )
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        # Get embedding dimension
        with torch.no_grad():
            dummy_text = self.tokenizer(["test"])
            dummy_features = self.model.encode_text(dummy_text.to(self.device))
            self.embedding_dim = dummy_features.shape[1]
        
        print(f"CLIP model loaded. Embedding dimension: {self.embedding_dim}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text string
            
        Returns:
            Embedding vector as numpy array
        """
        with torch.no_grad():
            tokens = self.tokenizer([text]).to(self.device)
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy()[0]
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings, shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                tokens = self.tokenizer(batch).to(self.device)
                text_features = self.model.encode_text(tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                all_embeddings.append(text_features.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def embed_image(self, image: Union[str, Image.Image, bytes]) -> np.ndarray:
        """
        Generate embedding for a single image.
        
        Args:
            image: Image path, PIL Image, or bytes
            
        Returns:
            Embedding vector as numpy array
        """
        # Load image
        if isinstance(image, str):
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, bytes):
            from io import BytesIO
            pil_image = Image.open(BytesIO(image)).convert('RGB')
        elif isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Preprocess and embed
        with torch.no_grad():
            image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy()[0]
    
    def embed_images(self, images: List[Union[str, Image.Image, bytes]], batch_size: int = 16) -> np.ndarray:
        """
        Generate embeddings for multiple images.
        
        Args:
            images: List of image paths, PIL Images, or bytes
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings, shape (n_images, embedding_dim)
        """
        if not images:
            return np.array([])
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                
                # Load and preprocess images
                image_tensors = []
                for img in batch:
                    if isinstance(img, str):
                        pil_image = Image.open(img).convert('RGB')
                    elif isinstance(img, bytes):
                        from io import BytesIO
                        pil_image = Image.open(BytesIO(img)).convert('RGB')
                    else:
                        pil_image = img.convert('RGB')
                    
                    image_tensors.append(self.preprocess(pil_image))
                
                image_batch = torch.stack(image_tensors).to(self.device)
                image_features = self.model.encode_image(image_batch)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                all_embeddings.append(image_features.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def embed(self, content: Union[str, Image.Image, bytes], content_type: str = 'auto') -> np.ndarray:
        """
        Generate embedding for either text or image.
        
        Args:
            content: Text string, image path, PIL Image, or bytes
            content_type: 'text', 'image', or 'auto' (auto-detect)
            
        Returns:
            Embedding vector as numpy array
        """
        if content_type == 'auto':
            if isinstance(content, str):
                # Check if it's a file path
                if os.path.exists(content) and content.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                    content_type = 'image'
                else:
                    content_type = 'text'
            elif isinstance(content, (Image.Image, bytes)):
                content_type = 'image'
            else:
                content_type = 'text'
        
        if content_type == 'text':
            return self.embed_text(content)
        else:
            return self.embed_image(content)
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        return float(np.dot(embedding1, embedding2))
    
    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return self.embedding_dim
    
    @staticmethod
    def is_available() -> bool:
        """Check if CLIP is available."""
        return CLIP_AVAILABLE


class HybridEmbedder:
    """
    Hybrid embedder that uses:
    - CLIP for images (true visual embeddings)
    - Sentence-transformers for text (better text understanding)
    
    Both are projected to the same dimension for unified search.
    """
    
    def __init__(
        self,
        text_model_name: str = "all-MiniLM-L6-v2",
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "laion2b_s34b_b79k",
        device: Optional[str] = None
    ):
        """
        Initialize hybrid embedder.
        
        Args:
            text_model_name: Sentence-transformer model for text
            clip_model_name: CLIP model for images
            clip_pretrained: CLIP pretrained weights
            device: Device to use
        """
        from sentence_transformers import SentenceTransformer
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Loading hybrid embedder on {self.device}...")
        
        # Load text embedder
        self.text_model = SentenceTransformer(text_model_name, device=self.device)
        self.text_dim = self.text_model.get_sentence_embedding_dimension()
        
        # Load CLIP for images
        self.clip_available = CLIP_AVAILABLE
        if CLIP_AVAILABLE:
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                clip_model_name,
                pretrained=clip_pretrained,
                device=self.device
            )
            self.clip_model.eval()
            
            # Get CLIP dimension
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224).to(self.device)
                clip_features = self.clip_model.encode_image(dummy)
                self.clip_dim = clip_features.shape[1]
            
            # Create projection layer if dimensions differ
            if self.clip_dim != self.text_dim:
                self.projection = torch.nn.Linear(self.clip_dim, self.text_dim).to(self.device)
                # Initialize as identity-like
                torch.nn.init.xavier_uniform_(self.projection.weight)
                torch.nn.init.zeros_(self.projection.bias)
            else:
                self.projection = None
            
            print(f"CLIP loaded. Projecting {self.clip_dim} -> {self.text_dim} dimensions")
        else:
            print("CLIP not available. Images will use VLM descriptions instead.")
        
        self.embedding_dim = self.text_dim
        print(f"Hybrid embedder ready. Output dimension: {self.embedding_dim}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed text using sentence-transformer."""
        return self.text_model.encode(text, convert_to_numpy=True)
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed multiple texts."""
        return self.text_model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
    
    def embed_image(self, image: Union[str, Image.Image, bytes]) -> np.ndarray:
        """Embed image using CLIP, projected to text embedding space."""
        if not self.clip_available:
            raise RuntimeError("CLIP not available for image embedding")
        
        # Load image
        if isinstance(image, str):
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, bytes):
            from io import BytesIO
            pil_image = Image.open(BytesIO(image)).convert('RGB')
        else:
            pil_image = image.convert('RGB')
        
        # Get CLIP embedding
        with torch.no_grad():
            image_tensor = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
            image_features = self.clip_model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Project to text dimension if needed
            if self.projection is not None:
                image_features = self.projection(image_features)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy()[0]
    
    def embed(self, text: str) -> np.ndarray:
        """
        Embed text (alias for embed_text for compatibility).
        
        Args:
            text: Input text string
            
        Returns:
            Embedding vector as numpy array
        """
        return self.embed_text(text)
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed multiple texts (alias for embed_texts for compatibility).
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings
        """
        return self.embed_texts(texts, batch_size)
    
    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return self.embedding_dim
    
    def can_embed_images(self) -> bool:
        """Check if image embedding is available."""
        return self.clip_available


def detect_content_type(file_path: str) -> str:
    """
    Detect if a file is text-based or image-based.
    
    Args:
        file_path: Path to the file
        
    Returns:
        'image', 'text', or 'mixed' (for PDFs with images)
    """
    ext = Path(file_path).suffix.lower()
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff'}
    text_extensions = {'.txt', '.docx', '.doc', '.rtf'}
    
    if ext in image_extensions:
        return 'image'
    elif ext in text_extensions:
        return 'text'
    elif ext == '.pdf':
        # PDFs can be mixed - need to analyze
        return 'mixed'
    else:
        return 'text'  # Default to text


def has_images_in_pdf(file_path: str) -> bool:
    """
    Check if a PDF contains images.
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        True if PDF contains images
    """
    import fitz
    
    try:
        doc = fitz.open(file_path)
        for page in doc:
            images = page.get_images(full=True)
            if images:
                doc.close()
                return True
        doc.close()
        return False
    except Exception:
        return False
